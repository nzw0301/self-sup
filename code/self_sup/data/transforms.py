from typing import List

import numpy as np
import torch
import torchvision.transforms
from omegaconf import OmegaConf
from torchvision import transforms


class RandomGaussianBlur:
    """
    https://github.com/facebookresearch/swav/blob/master/src/multicropdataset.py
    """

    @staticmethod
    def __call__(img):
        import cv2

        if np.random.rand() > 0.5:
            return img
        sigma = np.random.uniform(0.1, 2.0)
        return cv2.GaussianBlur(
            np.asarray(img), (23, 23), sigma
        )  # 23 is for imagenet that has size of 224 x 224.


def create_simclr_data_augmentation(strength: float, size: int) -> transforms.Compose:
    """
    Create SimCLR's data augmentation.

    :param strength: strength parameter for colorjiter.
    :param size: `RandomResizedCrop`'s size parameter.
    :return: Compose of transforms.

    """
    color_jitter = transforms.ColorJitter(
        brightness=0.8 * strength,
        contrast=0.8 * strength,
        saturation=0.8 * strength,
        hue=0.2 * strength,
    )

    rnd_color_jitter = transforms.RandomApply(transforms=[color_jitter], p=0.8)

    common_transforms = [
        transforms.RandomResizedCrop(size=size),
        transforms.RandomHorizontalFlip(p=0.5),
        # the following two are `color_distort`
        rnd_color_jitter,
        transforms.RandomGrayscale(0.2),
        # end of color_distort
    ]
    if size == 224:  # ImageNet-1K or pet dataset's shape.
        common_transforms.append(RandomGaussianBlur())
    elif size == 32:
        pass
    else:
        raise ValueError("`size` must be either `32` or `224`.")
    common_transforms.append(transforms.ToTensor())

    return transforms.Compose(common_transforms)


class SimCLRTransforms:
    def __init__(
        self, strength: float = 0.5, size: int = 32, num_views: int = 2
    ) -> None:
        # Definition is from Appendix A. of SimCLRv1 paper:
        # https://arxiv.org/pdf/2002.05709.pdf

        self.transform = create_simclr_data_augmentation(strength, size)

        if num_views <= 1:
            raise ValueError("`num_views` must be greater than 1.")

        self._num_views = num_views

    def __call__(self, x) -> List[torch.Tensor]:
        return [self.transform(x) for _ in range(self._num_views)]


def get_data_augmentation(cfg: OmegaConf) -> torchvision.transforms.Compose:
    aug_name = cfg["name"]
    assert aug_name in {"simclr_data_aug", "simclr_data_aug_for_linear_eval"}

    if aug_name == "simclr_data_aug":
        return create_simclr_data_augmentation(cfg["strength"], size=cfg["size"])
    elif aug_name == "simclr_data_aug_for_linear_eval":
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(size=cfg["size"]),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
            ]
        )
