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

import src.models.vision_transformer as vit
from src.models.attentive_pooler import AttentiveClassifier
from src.models.mae_models import MAEDecoder
from src.datasets.data_manager import (
    init_data,
)
from src.utils.distributed import init_distributed, AllReduce
from src.utils.schedulers import (
    WarmupCosineSchedule,
    CosineWDSchedule,
)
from src.utils.logging import AverageMeter, CSVLogger
from app.vjepa.utils import load_checkpoint, init_video_model

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

    # -- PRETRAIN
    args_pretrain = args_eval.get("pretrain")
    checkpoint_key = args_pretrain.get("checkpoint_key", "target_encoder")
    model_name = args_pretrain.get("model_name", None)
    patch_size = args_pretrain.get("patch_size", None)
    pretrain_folder = args_pretrain.get("folder", None)
    ckp_fname = args_pretrain.get("checkpoint", None)
    tag = args_pretrain.get("write_tag", None)
    use_sdpa = args_pretrain.get("use_sdpa", True)
    use_SiLU = args_pretrain.get("use_silu", False)
    tight_SiLU = args_pretrain.get("tight_silu", True)
    uniform_power = args_pretrain.get("uniform_power", False)
    pretrained_path = os.path.join(pretrain_folder, ckp_fname)
    use_mask_tokens = cfgs_model.get("use_mask_tokens", True)
    cfgs_mask = args.get("mask")
    zero_init_mask_tokens = cfgs_model.get("zero_init_mask_tokens", True)
    num_frames = cfgs_data.get("num_frames")
    crop_size = cfgs_data.get("crop_size", 224)
    pred_depth = cfgs_model.get("pred_depth")
    pred_embed_dim = cfgs_model.get("pred_embed_dim")
    # Optional [for Video model]:
    tubelet_size = args_pretrain.get("tubelet_size", 2)
    frames_per_clip = args_pretrain.get("frames_per_clip", 1)

    # -- DATA
    args_data = args_eval.get("data")
    dataset_name = args_data.get("dataset_name")
    num_classes = args_data.get("num_classes")
    root_path = args_data.get("root_path", None)
    image_folder = args_data.get("image_folder", None)
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
    folder = os.path.join(pretrain_folder, "image_classification_frozen/")
    if eval_tag is not None:
        folder = os.path.join(folder, eval_tag)
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    log_file = os.path.join(folder, f"{tag}_r{rank}.csv")
    latest_path = os.path.join(folder, f"{tag}-latest.pth.tar")

    # -- make csv_logger
    if rank == 0:
        csv_logger = CSVLogger(log_file, ("%d", "epoch"), ("%.5f", "loss"), ("%.5f", "acc"))

    # Initialize model
    encoder, predictor = init_video_model(
        uniform_power=uniform_power,
        use_mask_tokens=use_mask_tokens,
        num_mask_tokens=len(cfgs_mask),
        zero_init_mask_tokens=zero_init_mask_tokens,
        device=device,
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        model_name=model_name,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_embed_dim=pred_embed_dim,
        use_sdpa=use_sdpa,
    )
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    predictor.eval()
    for p in predictor.parameters():
        p.requires_grad = False

    # -- init mae decoder
    mae_decoder = MAEDecoder(
        patch_size=16,
        in_chans=3,
        embed_dim=1024,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
    ).to(device)

    train_loader = make_dataloader(
        dataset_name=dataset_name,
        root_path=root_path,
        resolution=resolution,
        image_folder=image_folder,
        batch_size=batch_size,
        world_size=world_size,
        rank=rank,
        training=True,
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
    mae_decoder = DistributedDataParallel(mae_decoder, static_graph=True)  # TODO what is this?

    # -- load training checkpoint
    start_epoch = 0
    if resume_checkpoint:
        mae_decoder, optimizer, scaler, start_epoch = load_decoder_checkpoint(
            device=device, r_path=latest_path, decoder=mae_decoder, opt=optimizer, scaler=scaler
        )
        for _ in range(start_epoch * ipe):
            scheduler.step()
            wd_scheduler.step()

    def save_checkpoint(epoch):
        save_dict = {
            "mae_decoder": mae_decoder.state_dict(),
            "opt": optimizer.state_dict(),
            "scaler": None if scaler is None else scaler.state_dict(),
            "epoch": epoch,
            "batch_size": batch_size,
            "world_size": world_size,
            "lr": lr,
        }
        if rank == 0:
            torch.save(save_dict, latest_path)

    # TRAIN LOOP
    for epoch in range(start_epoch, num_epochs):
        logger.info("Epoch %d" % (epoch + 1))
        train_acc = run_one_epoch(
            device=device,
            training=True,
            encoder=encoder,
            predictor=predictor,
            decoder=mae_decoder,
            scaler=scaler,
            optimizer=optimizer,
            scheduler=scheduler,
            wd_scheduler=wd_scheduler,
            data_loader=train_loader,
            use_bfloat16=use_bfloat16,
        )

        val_acc = run_one_epoch(
            device=device,
            training=False,
            encoder=encoder,
            predictor=predictor,
            decoder=mae_decoder,
            scaler=scaler,
            optimizer=optimizer,
            scheduler=scheduler,
            wd_scheduler=wd_scheduler,
            data_loader=val_loader,
            use_bfloat16=use_bfloat16,
        )

        logger.info("[%5d] train: %.3f%% val: %.3f%%" % (epoch + 1, train_acc, val_acc))
        if rank == 0:
            csv_logger.log(epoch + 1, train_acc, val_acc)
        save_checkpoint(epoch + 1)


def run_one_epoch(
    device,
    training,
    encoder,
    predictor,
    decoder,
    scaler,
    optimizer,
    scheduler,
    wd_scheduler,
    data_loader,
    use_bfloat16,
):

    decoder.train(mode=training)
    criterion = pixel_loss
    top1_meter = AverageMeter()
    for itr, data in enumerate(data_loader):

        if training:
            scheduler.step()
            wd_scheduler.step()

        with torch.cuda.amp.autocast(dtype=torch.float16, enabled=use_bfloat16):

            imgs = data[0].to(device)
            with torch.no_grad():
                outputs = encoder(imgs)
                if not training:
                    outputs = classifier(outputs)
            if training:
                outputs = classifier(outputs)

        loss = criterion(outputs, labels)
        top1_acc = 100.0 * outputs.max(dim=1).indices.eq(labels).sum() / len(imgs)
        top1_acc = float(AllReduce.apply(top1_acc))
        top1_meter.update(top1_acc)

        if training:
            if use_bfloat16:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
                optimizer.step()
            optimizer.zero_grad()

        if itr % 20 == 0:
            logger.info(
                "[%5d] %.3f%% (loss: %.3f) [mem: %.2e]"
                % (itr, top1_meter.avg, loss, torch.cuda.max_memory_allocated() / 1024.0**2)
            )

    return top1_meter.avg


def pixel_loss(pred, target, mask=None, is_norm_pix_loss=False):
    if is_norm_pix_loss:
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + 1.0e-6) ** 0.5

    loss = (pred - target) ** 2
    # TODO fix
    loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

    loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
    return loss


def load_decoder_checkpoint(device, r_path, classifier, opt, scaler):
    try:
        checkpoint = torch.load(r_path, map_location=torch.device("cpu"))
        epoch = checkpoint["epoch"]

        # -- loading encoder
        pretrained_dict = checkpoint["classifier"]
        msg = classifier.load_state_dict(pretrained_dict)
        logger.info(f"loaded pretrained classifier from epoch {epoch} with msg: {msg}")

        # -- loading optimizer
        opt.load_state_dict(checkpoint["opt"])
        if scaler is not None:
            scaler.load_state_dict(checkpoint["scaler"])
        logger.info(f"loaded optimizers from epoch {epoch}")
        logger.info(f"read-path: {r_path}")
        del checkpoint

    except Exception as e:
        logger.info(f"Encountered exception when loading checkpoint {e}")
        epoch = 0

    return classifier, opt, scaler, epoch


def load_encoder_checkpoint(
    checkpoint_fpath,
    encoder,
    predictor,
):
    try:
        checkpoint = torch.load(checkpoint_fpath, map_location=torch.device("cpu"))
    except Exception as e:
        logger.info(f"Encountered exception when loading checkpoint {e}")

    try:
        epoch = checkpoint["epoch"]

        # -- loading encoder
        pretrained_dict = checkpoint["encoder"]
        msg = encoder.load_state_dict(pretrained_dict)
        logger.info(f"loaded pretrained encoder from epoch {epoch} with msg: {msg}")

        # -- loading predictor
        pretrained_dict = checkpoint["predictor"]
        msg = predictor.load_state_dict(pretrained_dict)
        logger.info(f"loaded pretrained predictor from epoch {epoch} with msg: {msg}")

        logger.info(f"read checkpoint from path: {checkpoint_fpath}")
        del checkpoint

    except Exception as e:
        logger.info(f"Encountered exception when loading checkpoint {e}")

    return (encoder, predictor)


def make_dataloader(
    dataset_name,
    root_path,
    image_folder,
    batch_size,
    world_size,
    rank,
    resolution=224,
    training=False,
    subset_file=None,
):
    normalization = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    if training:
        logger.info("implementing auto-agument strategy")
        transform = timm_make_transforms(
            input_size=resolution,
            is_training=training,
            auto_augment="original",
            interpolation="bicubic",
            re_prob=0.25,
            re_mode="pixel",
            re_count=1,
            mean=normalization[0],
            std=normalization[1],
        )
    else:
        transform = transforms.Compose(
            [
                transforms.Resize(size=int(resolution * 256 / 224)),
                transforms.CenterCrop(size=resolution),
                transforms.ToTensor(),
                transforms.Normalize(normalization[0], normalization[1]),
            ]
        )

    data_loader, _ = init_data(
        data=dataset_name,
        transform=transform,
        batch_size=batch_size,
        world_size=world_size,
        rank=rank,
        root_path=root_path,
        image_folder=image_folder,
        training=training,
        copy_data=False,
        drop_last=False,
        subset_file=subset_file,
    )
    return data_loader


def init_opt(
    decoder,
    iterations_per_epoch,
    start_lr,
    ref_lr,
    warmup,
    num_epochs,
    wd=1e-6,
    final_wd=1e-6,
    final_lr=0.0,
    use_bfloat16=False,
):
    param_groups = [
        {"params": (p for n, p in decoder.named_parameters() if ("bias" not in n) and (len(p.shape) != 1))},
        {
            "params": (p for n, p in decoder.named_parameters() if ("bias" in n) or (len(p.shape) == 1)),
            "WD_exclude": True,
            "weight_decay": 0,
        },
    ]

    logger.info("Using AdamW")
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=int(warmup * iterations_per_epoch),
        start_lr=start_lr,
        ref_lr=ref_lr,
        final_lr=final_lr,
        T_max=int(num_epochs * iterations_per_epoch),
    )
    wd_scheduler = CosineWDSchedule(
        optimizer, ref_wd=wd, final_wd=final_wd, T_max=int(num_epochs * iterations_per_epoch)
    )
    scaler = torch.cuda.amp.GradScaler() if use_bfloat16 else None
    return optimizer, scaler, scheduler, wd_scheduler
