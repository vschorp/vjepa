# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    # -- WARNING: IF DOING DISTRIBUTED TRAINING ON A NON-SLURM CLUSTER, MAKE
    # --          SURE TO UPDATE THIS TO GET LOCAL-RANK ON NODE, OR ENSURE
    # --          THAT YOUR JOBS ARE LAUNCHED WITH ONLY 1 DEVICE VISIBLE
    # --          TO EACH PROCESS
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["SLURM_LOCALID"]
except Exception:
    pass

import logging
import pprint

import numpy as np

import torch
import torch.multiprocessing as mp
import torchvision.transforms as transforms

from torch.nn.parallel import DistributedDataParallel

from timm.data import create_transform as timm_make_transforms

import src.models.vision_transformer as vit_encoder
import src.models.predictor as vit_predictor
from src.models.attentive_pooler import AttentiveClassifier
from src.models.utils.multimask import MultiMaskWrapper, PredictorMultiMaskWrapper
from src.models.mae_models import MaeDecoder
from src.datasets.data_manager import init_data
from src.utils.distributed import init_distributed, AllReduce
from src.utils.schedulers import (
    WarmupCosineSchedule,
    CosineWDSchedule,
)
from src.utils.logging import AverageMeter, CSVLogger
from app.vjepa.utils import load_checkpoint, init_video_model
from evals.image_reconstruction_frozen.utils import make_dataloader, MaskGenerator

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

pp = pprint.PrettyPrinter(indent=4)


def main(args_eval, resume_preempt=False):

    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # image reconstruction with frozen encode and predict models. Train a decoder.
    # Dataset is ImageNet1k with 224x224 resolution images masked with 16x16 patches.

    # -- PRETRAIN
    args_pretrain = args_eval.get("pretrain")
    encoder_checkpoint_key = args_pretrain.get("encoder_checkpoint_key")
    predictor_checkpoint_key = args_pretrain.get("predictor_checkpoint_key")
    model_name = args_pretrain.get("model_name", None)
    patch_size = args_pretrain.get("patch_size", None)
    pretrain_folder = args_pretrain.get("folder", None)
    ckp_fname = args_pretrain.get("checkpoint", None)
    tag = args_pretrain.get("write_tag", None)
    use_sdpa = args_pretrain.get("use_sdpa", True)
    use_SiLU = args_pretrain.get("use_silu", False)
    tight_SiLU = args_pretrain.get("tight_silu", True)
    uniform_power = args_pretrain.get("uniform_power", False)
    pretrained_model_fpath = os.path.join(pretrain_folder, ckp_fname)
    pred_depth = args_pretrain.get("pred_depth")
    pred_embed_dim = args_pretrain.get("pred_embed_dim")
    use_mask_tokens = args_pretrain.get("use_mask_tokens", True)
    zero_init_mask_tokens = args_pretrain.get("zero_init_mask_tokens", True)

    # -- MASKING
    cfgs_mask = args_eval.get("mask")

    # -- DATA
    args_data = args_eval.get("data")
    dataset_name = args_data.get("dataset_name")
    num_classes = args_data.get("num_classes")
    root_path = args_data.get("root_path", None)
    image_folder = args_data.get("image_folder", None)
    resolution = args_data.get("resolution", 224)
    frames_per_clip = args_data.get("frames_per_clip")
    tubelet_size = args_pretrain.get("tubelet_size", 2)

    # -- OPTIMIZATION
    args_opt = args_eval.get("optimization")
    batch_size = args_opt.get("batch_size")
    num_epochs = args_opt.get("num_epochs")
    wd = args_opt.get("weight_decay")
    start_lr = args_opt.get("start_lr")
    lr = args_opt.get("lr")
    final_lr = args_opt.get("final_lr")
    warmup = args_opt.get("warmup")
    use_bfloat16 = args_opt.get("use_bfloat16")

    # -- EXPERIMENT-ID/TAG (optional)
    resume_checkpoint = args_eval.get("resume_checkpoint", False) or resume_preempt
    eval_tag = args_eval.get("tag", None)
    eval_name = args_eval.get("eval_name", None)

    # ----------------------------------------------------------------------- #

    try:
        mp.set_start_method("spawn")
    except Exception:
        pass

    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

    world_size, rank = init_distributed()
    logger.info(f"Initialized (rank/world-size) {rank}/{world_size}")

    # -- log/checkpointing paths
    folder = os.path.join(pretrain_folder, eval_name)
    if eval_tag is not None:
        folder = os.path.join(folder, eval_tag)
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    log_file = os.path.join(folder, f"{tag}_r{rank}.csv")
    latest_path = os.path.join(folder, f"{tag}-latest.pth.tar")

    # -- make csv_logger
    if rank == 0:
        csv_logger = CSVLogger(log_file, ("%d", "epoch"), ("%.5f", "loss"), ("%.5f", "acc"))

    train_loader = make_dataloader(
        dataset_name=dataset_name,
        root_path=root_path,
        resolution=resolution,
        image_folder=image_folder,
        batch_size=batch_size,
        world_size=world_size,
        rank=rank,
        training=True,
        logger=logger,
    )
    val_loader = make_dataloader(
        dataset_name=dataset_name,
        root_path=root_path,
        resolution=resolution,
        image_folder=image_folder,
        batch_size=batch_size,
        world_size=world_size,
        rank=rank,
        training=False,
        logger=logger,
    )
    mask_generator = MaskGenerator(resolution, frames_per_clip, patch_size, tubelet_size, mask_ratio=0.9)
    ipe = len(train_loader)
    logger.info(f"Dataloader created... iterations per epoch: {ipe}")

    # test train_loader
    for itr, data in enumerate(train_loader):
        if itr > 10:
            break
        img_batch = data[0]
        img_batch = torch.unsqueeze(img_batch, 1)
        clip_batch = torch.cat([img_batch, img_batch], dim=1)
        B, T, C, H, W = clip_batch.shape
        mask = mask_generator.get_random_masks(
            B
        )  # TODO fix masks generator to use randomBlock3d (might want to just import _MaskGenerator)
        # apply src.masks.utils.apply_masks to clip_batch to see if it works

        logger.info(f"Data shape: {clip_batch.shape}")
        # Question: do we need to keep frame count = 16 to match learned pos-temporal embeddings?
        # initialize encoder and predictor models
        # initialize decoder model
        # initialize
        # add training loop with pixel wise loss of masked pixel predictions
