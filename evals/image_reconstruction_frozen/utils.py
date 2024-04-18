import torchvision.transforms as transforms
from timm.data import create_transform as timm_make_transforms
import numpy as np
import torch
import logging
import sys
from torchvision.utils import save_image, make_grid
import wandb

from src.datasets.data_manager import init_data
import src.models.vision_transformer as video_vit
import src.models.predictor as vit_pred
from src.models.utils.multimask import MultiMaskWrapper, PredictorMultiMaskWrapper
from src.utils.tensors import trunc_normal_
from src.utils.schedulers import (
    WarmupCosineSchedule,
    CosineWDSchedule,
)

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def make_dataloader(
    dataset_name,
    root_path,
    image_folder,
    batch_size,
    world_size,
    rank,
    logger,
    collator,
    normalization,
    resolution=224,
    training=False,
    subset_file=None,
):
    # normalization = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    # if training:
    #     logger.info("implementing auto-agument strategy")
    #     transform = timm_make_transforms(
    #         input_size=resolution,
    #         is_training=training,
    #         auto_augment="original",
    #         interpolation="bicubic",
    #         re_prob=0.25,
    #         re_mode="pixel",
    #         re_count=1,
    #         mean=normalization[0],
    #         std=normalization[1],
    #     )
    # else:
    #     transform = transforms.Compose(
    #         [
    #             transforms.Resize(size=int(resolution * 256 / 224)),
    #             transforms.CenterCrop(size=resolution),
    #             transforms.ToTensor(),
    #             transforms.Normalize(normalization[0], normalization[1]),
    #         ]
    #     )
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
        collator=collator,
    )
    return data_loader


def init_video_model(
    device,
    patch_size=16,
    num_frames=16,
    tubelet_size=2,
    model_name="vit_base",
    crop_size=224,
    pred_depth=6,
    pred_embed_dim=384,
    uniform_power=False,
    use_mask_tokens=False,
    num_mask_tokens=2,
    zero_init_mask_tokens=True,
    use_sdpa=False,
):
    encoder = video_vit.__dict__[model_name](
        img_size=crop_size,
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        uniform_power=uniform_power,
        use_sdpa=use_sdpa,
    )
    encoder = MultiMaskWrapper(encoder)
    predictor = vit_pred.__dict__["vit_predictor"](
        img_size=crop_size,
        use_mask_tokens=use_mask_tokens,
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        embed_dim=encoder.backbone.embed_dim,
        predictor_embed_dim=pred_embed_dim,
        depth=pred_depth,
        num_heads=encoder.backbone.num_heads,
        uniform_power=uniform_power,
        num_mask_tokens=num_mask_tokens,
        zero_init_mask_tokens=zero_init_mask_tokens,
        use_sdpa=use_sdpa,
    )
    predictor = PredictorMultiMaskWrapper(predictor)

    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    for m in encoder.modules():
        init_weights(m)

    for m in predictor.modules():
        init_weights(m)

    encoder.to(device)
    predictor.to(device)
    logger.info(encoder)
    logger.info(predictor)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"Encoder number of parameters: {count_parameters(encoder)}")
    logger.info(f"Predictor number of parameters: {count_parameters(predictor)}")

    return encoder, predictor


def load_checkpoint(r_path, encoder, predictor, encoder_checkpoint_key, predictor_checkpoint_key):
    try:
        checkpoint = torch.load(r_path, map_location=torch.device("cpu"))
    except Exception as e:
        logger.info(f"Encountered exception when loading checkpoint {e}")
        return None, None

    # -- loading encoder
    pretrained_dict = checkpoint[encoder_checkpoint_key]
    msg = encoder.load_state_dict(pretrained_dict)
    logger.info(f"loaded pretrained encoder with msg: {msg}")

    # -- loading predictor
    pretrained_dict = checkpoint[predictor_checkpoint_key]
    msg = predictor.load_state_dict(pretrained_dict)
    logger.info(f"loaded pretrained predictor with msg: {msg}")

    return encoder, predictor


def load_decoder_checkpoint(r_path, decoder, opt, scaler):
    try:
        checkpoint = torch.load(r_path, map_location=torch.device("cpu"))
        epoch = checkpoint["epoch"]

        # -- loading encoder
        pretrained_dict = checkpoint["decoder"]
        msg = decoder.load_state_dict(pretrained_dict)
        logger.info(f"loaded pretrained decoder from epoch {epoch} with msg: {msg}")

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

    return decoder, opt, scaler, epoch


def save_checkpoint(decoder, optimizer, scaler, epoch, batch_size, world_size, lr, rank, latest_path):
    save_dict = {
        "decoder": decoder.state_dict(),
        "opt": optimizer.state_dict(),
        "scaler": None if scaler is None else scaler.state_dict(),
        "epoch": epoch,
        "batch_size": batch_size,
        "world_size": world_size,
        "lr": lr,
    }
    if rank == 0:
        torch.save(save_dict, latest_path)


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


def save_img_batch(img: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, fname: str = "img_batch", use_wandb=True):
    ### img as [B, C, H, W]
    batch_size = img.shape[0]
    saved_batch_size = min(4, batch_size)
    img = img * std[None, :, None, None] + mean[None, :, None, None]
    img = img.clip(0, 1)
    img_grid = make_grid(img[:saved_batch_size], nrow=4)
    if use_wandb:
        wandb.log({fname: wandb.Image(img_grid)})
    else:
        save_image(img_grid, f"{fname}.png")


def reconstruct_masked_img(
    target_batch, mask_pred, n_channels, patch_size, n_tokens_per_frame, first_frame_max_pred_token_idx, resolution=224
):
    batch_size = target_batch.shape[0]
    n_patches_per_dim = resolution // patch_size
    img_patches = torch.zeros(
        (batch_size, n_tokens_per_frame, n_channels * patch_size * patch_size),
        dtype=target_batch.dtype,
        device=target_batch.device,
    )  # [B, n_tokens, n_channels * patch_size * patch_size]
    target_token_idx = mask_pred[:, :first_frame_max_pred_token_idx]
    img_patches[torch.arange(batch_size).unsqueeze(1), target_token_idx] = target_batch
    img_patches = img_patches.unflatten(2, (n_channels, patch_size, patch_size)).unflatten(
        1, (1, n_patches_per_dim, n_patches_per_dim)
    )  # [B, 1, n_patches_per_dim, n_patches_per_dim, n_channels, patch_size, patch_size]
    img_patches = img_patches.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
    img_batch = img_patches.view(batch_size, n_channels, resolution, resolution)
    return img_batch
