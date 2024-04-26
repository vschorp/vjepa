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
from src.masks.multiblock3d import MaskCollator as MB3DMaskCollator
from src.masks.random_tube import MaskCollator as TubeMaskCollator
from src.utils.tensors import repeat_interleave_batch
from app.vjepa.transforms import make_transforms
from evals.video_reconstruction_frozen.utils import (
    init_video_model,
    init_opt,
    load_pretrained_checkpoint,
    load_decoder_checkpoint,
    save_decoder_checkpoint,
    log_images,
)

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

    # video reconstruction with frozen encode and predict models. Train a decoder.
    # Dataset is ImageNet1k with 224x224 resolution images masked with 16x16 patches.

    # -- FOLDER PATHS
    data_path = args_eval.get("data_path")

    # -- PRETRAIN
    args_pretrain = args_eval.get("pretrain")
    encoder_checkpoint_key = args_pretrain.get("encoder_checkpoint_key")
    predictor_checkpoint_key = args_pretrain.get("predictor_checkpoint_key")
    model_name = args_pretrain.get("model_name", None)
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

    # -- MASKING
    cfgs_mask = args_eval.get("mask")
    mask_type = args_eval.get("mask_type", "multiblock3d")

    # -- DATA
    args_data = args_eval.get("data")
    dataset_type = args_data.get("dataset_type")
    dataset_train = args_data.get("dataset_train")
    dataset_val = args_data.get("dataset_val")
    decode_one_clip = args_data.get("decode_one_clip")
    num_clips = args_data.get("num_clips")
    frames_per_clip = args_data.get("num_frames")
    tubelet_size = args_data.get("tubelet_size")
    sampling_rate = args_data.get("sampling_rate")
    resolution = args_data.get("crop_size")
    patch_size = args_data.get("patch_size")
    pin_mem = args_data.get("pin_mem")
    num_workers = args_data.get("num_workers")
    filter_short_videos = args_data.get("filter_short_videos")
    clip_duration = args_data.get("clip_duration")

    # -- DATA AUGMENTATION
    args_data_aug = args_eval.get("data_aug")
    use_aa = args_data_aug.get("auto_augment")
    motion_shift = args_data_aug.get("motion_shift")
    ar_range = args_data_aug.get("random_resize_aspect_ratio")
    rr_scale = args_data_aug.get("random_resize_scale")
    reprob = args_data_aug.get("reprob")

    # -- OPTIMIZATION
    args_opt = args_eval.get("optimization")
    num_epochs = args_opt.get("num_epochs")
    batch_size = args_opt.get("batch_size")
    wd = args_opt.get("weight_decay")
    lr = args_opt.get("lr")
    start_lr = args_opt.get("start_lr")
    final_lr = args_opt.get("final_lr")
    warmup = args_opt.get("warmup")
    use_bfloat16 = args_opt.get("use_bfloat16")

    # -- EXPERIMENT-ID/TAG (optional)
    resume_checkpoint = args_eval.get("resume_checkpoint", False) or resume_preempt
    eval_tag = args_eval.get("tag", None)
    eval_name = args_eval.get("eval_name", None)

    # -- LOGGING
    args_logging = args_eval.get("logging")
    log_folder = args_logging.get("folder")
    log_path = os.path.join(data_path, log_folder)

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

    if rank == 0:
        wandb.init(
            entity="vjepa-excavator",
            project=eval_name,
            name=eval_tag,
            dir=data_path,
        )

    # -- log/checkpointing paths
    decoder_ckpt_folder = os.path.join(data_path, ckpt_folder, eval_name)
    if eval_tag is not None:
        decoder_ckpt_folder = os.path.join(decoder_ckpt_folder, eval_tag)
    if not os.path.exists(decoder_ckpt_folder):
        os.makedirs(decoder_ckpt_folder, exist_ok=True)
    decoder_ckpt_latest_path = os.path.join(decoder_ckpt_folder, f"{tag}-latest.pth.tar")

    # -- make data transforms
    if mask_type == "multiblock3d":
        logger.info("Initializing basic multi-block mask")
        mask_collator = MB3DMaskCollator(
            crop_size=resolution,
            num_frames=frames_per_clip,
            patch_size=patch_size,
            tubelet_size=tubelet_size,
            cfgs_mask=cfgs_mask,
        )
    else:
        logger.info("Initializing random tube mask")
        mask_collator = TubeMaskCollator(
            crop_size=resolution,
            num_frames=frames_per_clip,
            patch_size=patch_size,
            tubelet_size=tubelet_size,
            cfgs_mask=cfgs_mask,
        )
    transform = make_transforms(
        random_horizontal_flip=True,
        random_resize_aspect_ratio=ar_range,
        random_resize_scale=rr_scale,
        reprob=reprob,
        auto_augment=use_aa,
        motion_shift=motion_shift,
        crop_size=resolution,
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
    encoder, predictor = load_pretrained_checkpoint(
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
        tubelet_size=2,
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
        # for _ in range(start_epoch * ipe):
        #     scheduler.step()
        #     wd_scheduler.step()

    # -- initialize data loaders
    (unsupervised_train_loader, _) = init_data(
        data=dataset_type,
        root_path=dataset_train,
        batch_size=batch_size,
        training=True,
        clip_len=frames_per_clip,
        frame_sample_rate=sampling_rate,
        filter_short_videos=filter_short_videos,
        decode_one_clip=decode_one_clip,
        duration=clip_duration,
        num_clips=num_clips,
        transform=transform,
        datasets_weights=None,
        collator=mask_collator,
        num_workers=num_workers,
        world_size=world_size,
        pin_mem=pin_mem,
        rank=rank,
        log_dir=None,
    )
    try:
        _dlen = len(unsupervised_train_loader)
    except Exception:  # Different interface for webdataset
        _dlen = unsupervised_train_loader.num_batches
    train_ipe = _dlen
    logger.info(f"train iterations per epoch/dataset length: {train_ipe}/{_dlen}")

    (unsupervised_val_loader, _) = init_data(
        data=dataset_type,
        root_path=dataset_val,
        batch_size=batch_size,
        training=False,
        clip_len=frames_per_clip,
        frame_sample_rate=sampling_rate,
        filter_short_videos=filter_short_videos,
        decode_one_clip=decode_one_clip,
        duration=clip_duration,
        num_clips=num_clips,
        transform=transform,
        datasets_weights=None,
        collator=mask_collator,
        num_workers=num_workers,
        world_size=world_size,
        pin_mem=pin_mem,
        rank=rank,
        log_dir=None,
    )

    # -- optimizer and scheduler
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        decoder=mae_decoder,
        wd=wd,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        iterations_per_epoch=train_ipe,
        warmup=warmup,
        num_epochs=num_epochs,
        use_bfloat16=use_bfloat16,
    )

    n_tokens_per_clip = (frames_per_clip // tubelet_size) * (resolution // patch_size) ** 2
    save_img_every_n = 1000

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
            data_loader=unsupervised_train_loader,
            use_bfloat16=use_bfloat16,
            frames_per_clip=frames_per_clip,
            num_clips=num_clips,
            batch_size=batch_size,
            loss_meter=train_loss_meter,
            n_tokens_per_clip=n_tokens_per_clip,
            resolution=resolution,
            patch_size=patch_size,
            n_channels=3,
            tubelet_size=tubelet_size,
            epoch=epoch,
            save_img_every_n=save_img_every_n,
            rank=rank,
        )

        save_decoder_checkpoint(
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
            data_loader=unsupervised_val_loader,
            use_bfloat16=use_bfloat16,
            frames_per_clip=frames_per_clip,
            num_clip=num_clips,
            batch_size=batch_size,
            loss_meter=train_loss_meter,
            n_tokens_per_clip=n_tokens_per_clip,
            resolution=resolution,
            patch_size=patch_size,
            n_channels=3,
            tubelet_size=tubelet_size,
            epoch=epoch,
            save_img_every_n=save_img_every_n,
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
    num_clips,
    batch_size,
    loss_meter,
    n_tokens_per_clip,
    resolution,
    patch_size,
    n_channels,
    tubelet_size,
    epoch,
    save_img_every_n,
    rank,
):
    loader = iter(data_loader)
    mae_decoder.train(training)
    if not training:
        mae_decoder.eval()
    mode = "train" if training else "val"
    logger.info("Training") if training else logger.info("Validation")
    logger.info(f"Epoch has {len(data_loader)} iterations")
    ipe = len(data_loader)
    n_clips_to_save = min(4, batch_size)
    if save_img_every_n > 0:
        save_img_every_n = ipe // 10
        logger.info(f"Saving images every {save_img_every_n} iterations.")
    for itr in tqdm(range(ipe)):
        try:
            udata, masks_enc, masks_pred = next(loader)
        except Exception:
            logger.info("Exhausted data loaders. Refreshing...")
            loader = iter(data_loader)
            udata, masks_enc, masks_pred = next(loader)
        assert len(masks_enc) == len(masks_pred), "Currently require num encoder masks = num predictor masks"

        # udata[0][0] = B, C, T, H, W
        # masks_pred[0] = B, N_Patches in clip
        clip_batch = udata[0][0]  # [B, C, T, H, W]
        clip_batch_patches = (
            clip_batch.unfold(1, 3, 3).unfold(2, 2, 2).unfold(3, 16, 16).unfold(4, 16, 16)
        )  # [B, 1, n_tubelets, n_path, n_path, 3, patch_size, patch_size]
        clip_batch_patches = clip_batch_patches.flatten(
            1, 4
        )  # [B, n_tubelets * n_patch * n_patch, 3, tubelet_size, patch_size, patch_size]
        clip_batch_patches = clip_batch_patches.flatten(
            2, 5
        )  # [B, n_tubelets * n_patch * n_patch, 3 * tubelet_size * patch_size * patch_size]

        clip_batch_patches = clip_batch_patches.to(device, non_blocking=True)

        targets = []
        for mask_pred in masks_pred:
            target_tokens = clip_batch_patches[torch.arange(batch_size)[:, None], mask_pred]
            targets.append(target_tokens.to(device, non_blocking=True))

        def load_clips():
            # -- unsupervised video clips
            # Put each clip on the GPU and concatenate along batch
            # dimension
            clips = torch.cat([u.to(device, non_blocking=True) for u in udata[0]], dim=0)

            # Put each mask-enc/mask-pred pair on the GPU and reuse the
            # same mask pair for each clip
            _masks_enc, _masks_pred = [], []
            for _me, _mp in zip(masks_enc, masks_pred):
                _me = _me.to(device, non_blocking=True)
                _mp = _mp.to(device, non_blocking=True)
                _me = repeat_interleave_batch(_me, batch_size, repeat=num_clips)
                _mp = repeat_interleave_batch(_mp, batch_size, repeat=num_clips)
                _masks_enc.append(_me)
                _masks_pred.append(_mp)

            return (clips, _masks_enc, _masks_pred)

        clips, masks_enc, masks_pred = load_clips()
        target_placeholder = len(masks_enc) * [None]

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
                clips_pred = []
                for i in range(len(masks_enc)):
                    if training:
                        clip_pred = mae_decoder(z_pred[i])
                        loss += pixel_loss(clip_pred, targets[i])
                    else:
                        with torch.no_grad():
                            clip_pred = mae_decoder(z_pred[i])
                            loss += pixel_loss(clip_pred, targets[i])
                    clips_pred.append(clip_pred)

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
                log_images(
                    clips,
                    clip_batch_patches,
                    targets,
                    clips_pred,
                    masks_enc,
                    masks_pred,
                    frames_per_clip,
                    n_channels,
                    tubelet_size,
                    patch_size,
                    n_tokens_per_clip,
                    resolution,
                    mode,
                    device,
                    batch_size,
                    n_clips_to_save,
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
