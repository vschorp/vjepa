import torchvision.transforms as transforms
from timm.data import create_transform as timm_make_transforms
import numpy as np
import torch

from src.datasets.data_manager import init_data
from src.masks.random_tube import MaskCollator as TubeMaskCollator
from src.masks.multiblock3d import MaskCollator as MB3DMaskCollator


def make_dataloader(
    dataset_name,
    root_path,
    image_folder,
    batch_size,
    world_size,
    rank,
    logger,
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


class MaskGenerator:
    def __init__(
        self,
        crop_size=224,
        num_frames=16,
        patch_size=16,
        tubelet_size=2,
        mask_raito=0.9,
    ):
        self.num_patches_spatial = (crop_size // patch_size) ** 2
        self.duration = num_frames // tubelet_size
        self.num_keep_spatial = int(self.num_patches_spatial * (1 - mask_raito))

    def _sample_block_size(self, temporal_scale, spatial_scale, aspect_ratio_scale):
        # -- Sample temporal block mask scale
        _rand = torch.rand(1).item()
        min_t, max_t = temporal_scale
        temporal_mask_scale = min_t + _rand * (max_t - min_t)
        t = max(1, int(self.duration * temporal_mask_scale))

        # -- Sample spatial block mask scale
        _rand = torch.rand(1).item()
        min_s, max_s = spatial_scale
        spatial_mask_scale = min_s + _rand * (max_s - min_s)
        spatial_num_keep = int(self.height * self.width * spatial_mask_scale)

        # -- Sample block aspect-ratio
        _rand = torch.rand(1).item()
        min_ar, max_ar = aspect_ratio_scale
        aspect_ratio = min_ar + _rand * (max_ar - min_ar)

        # -- Compute block height and width (given scale and aspect-ratio)
        h = int(round(math.sqrt(spatial_num_keep * aspect_ratio)))
        w = int(round(math.sqrt(spatial_num_keep / aspect_ratio)))
        h = min(h, self.height)
        w = min(w, self.width)

        return (t, h, w)

    def _sample_block_mask(self, b_size):
        t, h, w = b_size
        top = torch.randint(0, self.height - h + 1, (1,))
        left = torch.randint(0, self.width - w + 1, (1,))
        start = torch.randint(0, self.duration - t + 1, (1,))

        mask = torch.ones((self.duration, self.height, self.width), dtype=torch.int32)
        mask[start : start + t, top : top + h, left : left + w] = 0

        # Context mask will only span the first X frames
        # (X=self.max_context_frames)
        if self.max_context_duration < self.duration:
            mask[self.max_context_duration :, :, :] = 0

        # --
        return mask

    def get_random_masks(self, batch_size):
        keep_mask_indices_collated = []
        p_size = self._sample_block_size(
            temporal_scale=self.temporal_pred_mask_scale,
            spatial_scale=self.spatial_pred_mask_scale,
            aspect_ratio_scale=self.aspect_ratio,
        )
        for _ in range(batch_size):
            mask_e = torch.ones((self.duration, self.height, self.width), dtype=torch.int32)
            for _ in range(self.npred):
                mask_e *= self._sample_block_mask(p_size)
            mask_e = mask_e.flatten()
            keep_mask_indices = torch.nonzero(mask_e).squeeze()
            keep_mask_indices_collated.append(keep_mask_indices)
        keep_mask_indices_collated = torch.utils.data.default_collate(keep_mask_indices_collated)
        return keep_mask_indices_collated
