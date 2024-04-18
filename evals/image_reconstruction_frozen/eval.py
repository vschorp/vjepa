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
import time
from tqdm import tqdm
import numpy as np

import torch
import torch.multiprocessing as mp
import torchvision.transforms as transforms

from torch.nn.parallel import DistributedDataParallel

import wandb

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
from evals.image_reconstruction_frozen.utils import (
    make_dataloader,
    load_checkpoint,
    init_video_model,
    init_opt,
    load_decoder_checkpoint,
    save_checkpoint,
    save_img_batch,
    reconstruct_masked_img,
)
from src.masks.multiblock3d import MaskCollator as MB3DMaskCollator
from src.utils.tensors import repeat_interleave_batch

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

pp = pprint.PrettyPrinter(indent=4)

os.environ["WANDB__SERVICE_WAIT"] = "300"


def main(args_eval, resume_preempt=False):

    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # image reconstruction with frozen encode and predict models. Train a decoder.
    # Dataset is ImageNet1k with 224x224 resolution images masked with 16x16 patches.

    # -- FOLDER PATHS
    data_path = args_eval.get("data_path")

    # -- PRETRAIN
    args_pretrain = args_eval.get("pretrain")
    encoder_checkpoint_key = args_pretrain.get("encoder_checkpoint_key")
    predictor_checkpoint_key = args_pretrain.get("predictor_checkpoint_key")
    model_name = args_pretrain.get("model_name", None)
    patch_size = args_pretrain.get("patch_size", None)
    ckpt_folder = args_pretrain.get("folder", None)
    ckp_fname = args_pretrain.get("checkpoint", None)
    tag = args_pretrain.get("write_tag", None)
    use_sdpa = args_pretrain.get("use_sdpa", True)
    use_SiLU = args_pretrain.get("use_silu", False)
    tight_SiLU = args_pretrain.get("tight_silu", True)
    uniform_power = args_pretrain.get("unifo,rm_power", False)
    pretrained_model_fpath = os.path.join(data_path, ckpt_folder, ckp_fname)
    pred_depth = args_pretrain.get("pred_depth")
    pred_embed_dim = args_pretrain.get("pred_embed_dim")
    use_mask_tokens = args_pretrain.get("use_mask_tokens", True)
    zero_init_mask_tokens = args_pretrain.get("zero_init_mask_tokens", True)
    frames_per_clip = args_pretrain.get("frames_per_clip")
    tubelet_size = args_pretrain.get("tubelet_size", 2)

    # -- MASKING
    cfgs_mask = args_eval.get("mask")

    # -- DATA
    args_data = args_eval.get("data")
    dataset_name = args_data.get("dataset_name")
    num_classes = args_data.get("num_classes")
    image_folder = args_data.get("image_folder", None)
    image_path = os.path.join(data_path, image_folder)
    resolution = args_data.get("resolution", 224)

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

    # -- Image transforms
    args_img_transforms = args_eval.get("image_transforms", None)
    args_img_normalization = args_img_transforms.get("normalization", None)
    if args_img_normalization is not None:
        img_normalization_mean = args_img_normalization.get("mean")
        img_normalization_std = args_img_normalization.get("std")

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

    # Image normalization
    img_normalization_mean = torch.tensor(img_normalization_mean, device=device)
    img_normalization_std = torch.tensor(img_normalization_std, device=device)

    world_size, rank = init_distributed()
    logger.info(f"Initialized (rank/world-size) {rank}/{world_size}")

    if rank == 0:
        wandb.init(
            entity="vjepa-excavator",
            project=eval_name,
            name=eval_tag,
            dir=data_path,
        )

    # -- log/checkpointing pathslatest
    decoder_ckpt_folder = os.path.join(data_path, ckpt_folder, eval_name)
    if eval_tag is not None:
        decoder_ckpt_folder = os.path.join(decoder_ckpt_folder, eval_tag)
    if not os.path.exists(decoder_ckpt_folder):
        os.makedirs(decoder_ckpt_folder, exist_ok=True)
    log_file = os.path.join(decoder_ckpt_folder, f"{tag}_r{rank}.csv")
    decoder_ckpt_latest_path = os.path.join(decoder_ckpt_folder, f"{tag}-latest.pth.tar")

    # -- initialize mask generator
    mask_collator = MB3DMaskCollator(
        crop_size=resolution,
        num_frames=frames_per_clip,
        patch_size=patch_size,
        tubelet_size=tubelet_size,
        cfgs_mask=cfgs_mask,
    )

    # -- initialize encoder and predictor models
    encoder, predictor = init_video_model(
        uniform_power=uniform_power,
        use_mask_tokens=use_mask_tokens,
        num_mask_tokens=len(cfgs_mask),
        zero_init_mask_tokens=zero_init_mask_tokens,
        device=device,
        patch_size=patch_size,
        num_frames=frames_per_clip,
        tubelet_size=tubelet_size,
        model_name=model_name,
        crop_size=resolution,
        pred_depth=pred_depth,
        pred_embed_dim=pred_embed_dim,
        use_sdpa=use_sdpa,
    )
    encoder = DistributedDataParallel(encoder, static_graph=True)
    predictor = DistributedDataParallel(predictor, static_graph=True)
    encoder, predictor = load_checkpoint(
        pretrained_model_fpath, encoder, predictor, encoder_checkpoint_key, predictor_checkpoint_key
    )
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    predictor.eval()
    for p in predictor.parameters():
        p.requires_grad = False

    # -- initialize decoder model
    mae_decoder = MaeDecoder(
        patch_size=16,
        in_chans=3,
        embed_dim=1024,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
    ).to(device)
    mae_decoder.initialize_weights()
    mae_decoder = DistributedDataParallel(mae_decoder, static_graph=True)  # TODO what is this?

    # -- load training checkpoint
    start_epoch = 0
    if resume_checkpoint:
        mae_decoder, optimizer, scaler, start_epoch = load_decoder_checkpoint(
            r_path=decoder_ckpt_latest_path, decoder=mae_decoder, opt=optimizer, scaler=scaler
        )
        for _ in range(start_epoch * ipe):
            scheduler.step()
            wd_scheduler.step()

    train_loader = make_dataloader(
        dataset_name=dataset_name,
        root_path=data_path,
        resolution=resolution,
        image_folder=image_folder,
        batch_size=batch_size,
        world_size=world_size,
        rank=rank,
        training=True,
        logger=logger,
        collator=mask_collator,
        normalization=(img_normalization_mean, img_normalization_std),
    )
    val_loader = make_dataloader(
        dataset_name=dataset_name,
        root_path=data_path,
        resolution=resolution,
        image_folder=image_folder,
        batch_size=batch_size,
        world_size=world_size,
        rank=rank,
        training=False,
        logger=logger,
        collator=mask_collator,
        normalization=(img_normalization_mean, img_normalization_std),
    )
    ipe = len(train_loader)
    logger.info(f"Dataloader created... iterations per epoch: {ipe}")

    # -- optimizer and scheduler
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        decoder=mae_decoder,
        wd=wd,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        iterations_per_epoch=ipe,
        warmup=warmup,
        num_epochs=num_epochs,
        use_bfloat16=use_bfloat16,
    )

    n_tokens_per_frame = (resolution // patch_size) ** 2

    # -- TRAINING LOOP
    for epoch in range(start_epoch, num_epochs):
        logger.info("Epoch %d" % (epoch + 1))

        train_loss_meter = AverageMeter()
        val_loss_meter = AverageMeter()

        # -- TRAINING
        run_one_epoch(
            device=device,
            training=True,
            encoder=encoder,
            predictor=predictor,
            mae_decoder=mae_decoder,
            scaler=scaler,
            optimizer=optimizer,
            scheduler=scheduler,
            wd_scheduler=wd_scheduler,
            data_loader=train_loader,
            use_bfloat16=use_bfloat16,
            frames_per_clip=frames_per_clip,
            batch_size=batch_size,
            loss_meter=train_loss_meter,
            n_tokens_per_frame=n_tokens_per_frame,
            resolution=resolution,
            patch_size=patch_size,
            n_channels=3,
            img_normalization_mean=img_normalization_mean,
            img_normalization_std=img_normalization_std,
            epoch=epoch,
            save_img_every_n=1000,
            rank=rank,
        )

        save_checkpoint(
            mae_decoder, optimizer, scaler, epoch, batch_size, world_size, lr, rank, decoder_ckpt_latest_path
        )

        # -- VALIDATION
        run_one_epoch(
            device=device,
            training=False,
            encoder=encoder,
            predictor=predictor,
            mae_decoder=mae_decoder,
            scaler=scaler,
            optimizer=optimizer,
            scheduler=scheduler,
            wd_scheduler=wd_scheduler,
            data_loader=val_loader,
            use_bfloat16=use_bfloat16,
            frames_per_clip=frames_per_clip,
            batch_size=batch_size,
            loss_meter=train_loss_meter,
            n_tokens_per_frame=n_tokens_per_frame,
            resolution=resolution,
            patch_size=patch_size,
            n_channels=3,
            img_normalization_mean=img_normalization_mean,
            img_normalization_std=img_normalization_std,
            epoch=epoch,
            save_img_every_n=1000,
            rank=rank,
        )

        logger.info("Epoch %d: train loss %.3f, val loss %.3f" % (epoch + 1, train_loss_meter.avg, val_loss_meter.avg))


def run_one_epoch(
    device,
    training,
    encoder,
    predictor,
    mae_decoder,
    scaler,
    optimizer,
    scheduler,
    wd_scheduler,
    data_loader,
    use_bfloat16,
    frames_per_clip,
    batch_size,
    loss_meter,
    n_tokens_per_frame,
    resolution,
    patch_size,
    n_channels,
    img_normalization_mean,
    img_normalization_std,
    epoch,
    save_img_every_n,
    rank,
):
    mae_decoder.train(training)
    mode = "train" if training else "val"
    logger.info("Training") if training else logger.info("Validation")
    logger.info(f"Epoch has {len(data_loader)} iterations")
    for itr, data in tqdm(enumerate(data_loader)):

        img_batch, masks_enc, masks_pred = data
        img_batch = img_batch[0]  # ignore labels
        current_batch_size = img_batch.shape[0]
        if current_batch_size != batch_size:
            logger.info(f"Batch size mismatch, skipping batch...")
            continue

        img_batch = torch.unsqueeze(img_batch, 2)
        clip_batch = torch.cat(frames_per_clip * [img_batch], dim=2)

        # create target
        img_target = img_batch.squeeze(2)
        img_target_patches = (
            img_target.unfold(1, 3, 3).unfold(2, 16, 16).unfold(3, 16, 16)
        )  # [B, 1, n_path, n_path, 3, patch_size, patch_size]
        img_target_patches = img_target_patches.flatten(1, 3)  # [B, n_path * n_path, 3, patch_size, patch_size]
        img_target_patches = img_target_patches.flatten(2, 4)  # [B, n_path * n_path, 3 * patch_size * patch_size]

        # get first frame mask indices
        first_frame_max_pred_token_idx = []
        targets = []
        valid_masks = True
        for i in range(len(masks_pred)):
            first_frame_masks = masks_pred[i] < n_tokens_per_frame
            first_frame_tokens_min_count = first_frame_masks.sum(dim=1).min()
            first_frame_max_pred_token_idx.append(first_frame_tokens_min_count)
            if first_frame_tokens_min_count < 1:
                valid_masks = False
                break
            target_tokens_idx = masks_pred[i][:, :first_frame_tokens_min_count]
            target_tokens = img_target_patches[torch.arange(batch_size)[:, None], target_tokens_idx]
            targets.append(target_tokens.to(device, non_blocking=True))

        if not valid_masks:
            logger.info("Invalid masks, skipping batch...")
            continue

        def load_clips():
            # Put clip batch on the GPU
            clips = clip_batch.to(device, non_blocking=True)

            # Put each mask-enc/mask-pred pair on the GPU and reuse the
            # same mask pair for each clip
            _masks_enc, _masks_pred = [], []
            for _me, _mp in zip(masks_enc, masks_pred):
                _me = _me.to(device, non_blocking=True)
                _mp = _mp.to(device, non_blocking=True)
                _me = repeat_interleave_batch(_me, batch_size, repeat=1)
                _mp = repeat_interleave_batch(_mp, batch_size, repeat=1)
                _masks_enc.append(_me)
                _masks_pred.append(_mp)

            return (clips, _masks_enc, _masks_pred)

        clips, masks_enc, masks_pred = load_clips()
        target_placeholder = len(masks_enc) * [None]
        first_frame_pred_indices_list = []
        for i in range(len(masks_pred)):
            first_frame_pred_indices = masks_pred[i] < n_tokens_per_frame
            first_frame_pred_indices_list.append(first_frame_pred_indices.to(device, non_blocking=True))

        def step():
            if training:
                _new_lr = scheduler.step()
                _new_wd = wd_scheduler.step()

            def pixel_loss(pred, target, mask=None, is_norm_pix_loss=False):
                if is_norm_pix_loss:
                    mean = target.mean(dim=-1, keepdim=True)
                    var = target.var(dim=-1, keepdim=True)
                    target = (target - mean) / (var + 1.0e-6) ** 0.5

                pixel_loss = (pred - target) ** 2
                # TODO fix
                pixel_loss = torch.mean(pixel_loss)

                if mask is not None:
                    pixel_loss = (pixel_loss * mask).sum() / mask.sum()  # mean loss on removed patches
                return pixel_loss

            # Step 1. Forward
            with torch.cuda.amp.autocast(dtype=torch.float16, enabled=use_bfloat16):
                with torch.no_grad():
                    z_enc = encoder(clips, masks_enc)
                    z_pred = predictor(z_enc, target_placeholder, masks_enc, masks_pred)

                loss = 0.0
                imgs_pred = []
                for i in range(len(masks_enc)):
                    first_frame_pred_tokens = z_pred[i][
                        :, : first_frame_max_pred_token_idx[i]
                    ]  # only feed the tokens for first frame to decoder

                    if training:
                        img_pred = mae_decoder(first_frame_pred_tokens)
                        loss += pixel_loss(img_pred, targets[i])
                    else:
                        with torch.no_grad():
                            img_pred = mae_decoder(first_frame_pred_tokens)
                            loss += pixel_loss(img_pred, targets[i])
                    imgs_pred.append(img_pred)

            if training:
                # Step 2. Backward & step
                # _enc_norm, _pred_norm = 0.0, 0.0
                if use_bfloat16:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                else:
                    loss.backward()

                if use_bfloat16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad()

            # Image logging
            if save_img_every_n > 0 and itr % save_img_every_n == 0 and rank == 0:
                orig_img = clips[:, :, 0, :, :]
                save_img_batch(orig_img, img_normalization_mean, img_normalization_std, f"orig_imgs_{mode}")
                for i in range(len(masks_pred)):
                    target_img_batch = reconstruct_masked_img(
                        targets[i],
                        masks_pred[i],
                        n_channels,
                        patch_size,
                        n_tokens_per_frame,
                        first_frame_max_pred_token_idx[i],
                        resolution,
                    )
                    save_img_batch(
                        target_img_batch, img_normalization_mean, img_normalization_std, f"target_imgs_mask_{i}_{mode}"
                    )
                    pred_img_batch = reconstruct_masked_img(
                        imgs_pred[i],
                        masks_pred[i],
                        n_channels,
                        patch_size,
                        n_tokens_per_frame,
                        first_frame_max_pred_token_idx[i],
                        resolution,
                    )
                    save_img_batch(
                        pred_img_batch, img_normalization_mean, img_normalization_std, f"pred_imgs_mask_{i}_{mode}"
                    )

            return float(loss)

        loss = step()
        loss_meter.update(loss)
        if rank == 0:
            wandb.log(
                {
                    f"{mode}_epoch": epoch,
                    f"{mode}_iteration": itr,
                    f"{mode}_loss": loss_meter.avg,
                    "gpu-mem": torch.cuda.max_memory_allocated() / 1024.0**2,
                }
            )
