from typing import Optional

import torch
from omegaconf import OmegaConf
from torchvision.models import resnet18, resnet34, resnet50

from .head import ProjectionHead
from .utils import modify_resnet_by_simclr_for_cifar


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


def get_contrastive_model(
    cfg: OmegaConf, local_rank: int, pre_train_weight_path: Optional[str] = None
) -> ContrastiveModel:
    model = ContrastiveModel(
        base_cnn=cfg["backbone"]["name"],
        head=ProjectionHead(
            input_dim=2048 if cfg["backbone"]["name"] == "resnet50" else 512,
            latent_dim=cfg["projection_head"]["d"],
            num_non_linear_blocks=cfg["projection_head"]["num_hidden_layer"],
        ),
        is_cifar="cifar" in cfg["dataset"]["name"],
    )
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(local_rank)

    if pre_train_weight_path is None:
        return model

    state_dict = torch.load(pre_train_weight_path)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False, map_location=local_rank)
    return model
