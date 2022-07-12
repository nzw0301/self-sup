import torch
from torchvision.models import resnet18, resnet34, resnet50
from typing import Union, Optional
from torchvision.models import ResNet
from .classifier import SupervisedModel
from .head import ProjectionHead


class ContrastiveModel(torch.nn.Module):
    def __init__(
        self,
        base_cnn: str = "resnet18",
        head: Optional[ProjectionHead] = None,
        is_cifar: bool = True,
    ):
        """
        CNN architecture used in SimCLR v1.

        :param base_cnn: The backbone's model name. resnet18, resnet34, or resnet50.
        :param head: The projection head
        :param is_cifar: If it is `True`, network is modified by following SimCLR's CIFAR-10 experiments.
        """

        assert base_cnn in {"resnet18", "resnet34", "resnet50"}
        super(ContrastiveModel, self).__init__()

        if base_cnn == "resnet50":
            self.f = resnet50()
        elif base_cnn == "resnet34":
            self.f = resnet34()
        else:  # resnet18
            # TODO(nzw0301): should apply to the other resnet too?
            if is_cifar:
                self.f = modify_resnet_by_simclr_for_cifar(resnet18())
            else:
                self.f = resnet18()

        # drop the last classification layer
        self.f.fc = torch.nn.Identity()

        # projection head
        if head is not None:
            self.g = head
        else:
            self.g = torch.nn.Identity()

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        return features before projection head.
        :param inputs: FloatTensor that contains images.
        :return: feature representations.
        """

        return self.f(inputs)  # N x num_last_hidden_units

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:

        h = self.encode(inputs)
        z = self.g(h)
        return z  # N x d


def modify_resnet_by_simclr_for_cifar(
    model: Union[SupervisedModel, ResNet]
) -> Union[SupervisedModel, ResNet]:
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
    if isinstance(model, SupervisedModel):
        model.f.conv1 = conv
        model.f.maxpool = identity
    elif isinstance(model, ResNet):
        model.conv1 = conv
        model.maxpool = identity

    return model
