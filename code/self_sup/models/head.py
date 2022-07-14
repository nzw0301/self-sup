from collections import OrderedDict

import torch


class ProjectionHead(torch.nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, num_non_linear_blocks) -> None:
        """
        :param input_dim: the dimensionality of the encoder's output representation.
        :param latent_dim: the dimensionality of the latent and feature vectors.
        :param num_non_linear_blocks: the number of non-linear blocks.

        """
        super(ProjectionHead, self).__init__()

        last_dim = input_dim
        head = []

        # non-linear components.
        for i in range(num_non_linear_blocks):
            head.append((f"linear{i}", torch.nn.Linear(last_dim, latent_dim)))
            head.append((f"bn{i}", torch.nn.BatchNorm1d(latent_dim)))
            head.append((f"relu{i}", torch.nn.ReLU()))
            last_dim = latent_dim

        # final one
        head.append(
            ("last_linear", torch.nn.Linear(latent_dim, latent_dim, bias=False))
        )
        self.projection_head = torch.nn.Sequential(OrderedDict(head))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.projection_head(inputs)
