from typing import TYPE_CHECKING, Union

import torch
from torchvision.models import ResNet

if TYPE_CHECKING:
    # To avoid circular import error.
    # See https://docs.python.org/3/library/typing.html#typing.TYPE_CHECKING.
    from .classifier import SupervisedModel


def modify_resnet_by_simclr_for_cifar(
    model: Union["SupervisedModel", ResNet]
) -> Union["SupervisedModel", ResNet]:
    """By following SimCLR v1 paper, this function replaces a few layers for CIFAR-10 experiments.

    Args:
        model: Instance of `SupervisedModel`.

    Returns:
        SupervisedModel: Modified `SupervisedModel`.
    """

    # Replace the first conv2d with smaller conv and remove the first pooling.
    conv = torch.nn.Conv2d(
        in_channels=3, out_channels=64, stride=1, kernel_size=3, padding=3, bias=False,
    )
    identity = torch.nn.Identity()
    if isinstance(model, ResNet):
        model.conv1 = conv
        model.maxpool = identity
    else:
        model.f.conv1 = conv
        model.f.maxpool = identity

    return model
