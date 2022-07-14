import torch


class NT_Xent(torch.nn.Module):
    """
    Normalised Temperature-scaled cross-entropy loss.
    """

    def __init__(
        self,
        temperature: float = 0.1,
        reduction: str = "mean",
        device: torch.device = torch.device("cpu"),
    ):
        """
        :param temperature: Temperature parameter. The value must be positive.
        :param reduction: Same to PyTorch's `reduction` in losses.
        :param device: PyTorch's device instance.
        """

        reduction = reduction.lower()

        if temperature <= 0.0:
            raise ValueError("`temperature` must be positive. {}".format(temperature))

        if reduction not in {"none", "mean", "sum"}:
            raise ValueError(
                f'`reduction` must be in `{"none", "mean", "sum"}`. Not {reduction}'
            )

        super(NT_Xent, self).__init__()
        self.cross_entropy = torch.nn.functional.cross_entropy
        self.temperature = temperature
        self.reduction = reduction
        self.device = device

    def forward(self, view_0: torch.Tensor, view_1: torch.Tensor) -> torch.Tensor:
        """
        SimCLR's InfoNCE loss, namely, NTXent loss.
        The number of augmentation can be larger than 2.

        :param
            views_0: feature representation. The shape is (N, D), where
            `N` is the size of mini-batches (or the number of seed image, or K+1 in the paper),
            and `D` is the dimensionality of features.
            views_1: feature representation. The shape is (N, D). the first axis data should be
            corresponding to the fisrt axis's view_0.
        :return: Loss value. The shape depends on `reduction`: (2, N) or a scalar.
        """

        B = len(view_0)  # == N
        assert view_0.size() == view_1.size()

        # normalisation
        view_0 = torch.nn.functional.normalize(view_0, p=2, dim=1)  # N x D
        view_1 = torch.nn.functional.normalize(view_1, p=2, dim=1)  # N x D

        sim_00 = torch.matmul(view_0, view_0.t()) / self.temperature
        sim_01 = torch.matmul(view_0, view_1.t()) / self.temperature
        sim_11 = torch.matmul(view_1, view_1.t()) / self.temperature

        # remove its own similarity
        sim_00 = sim_00.flatten()[1:].view(B - 1, B + 1)[:, :-1].reshape(B, B - 1)
        sim_11 = sim_11.flatten()[1:].view(B - 1, B + 1)[:, :-1].reshape(B, B - 1)

        targets = torch.arange(B, device=self.device)
        loss_01 = self.cross_entropy(
            torch.hstack([sim_01, sim_00]), targets, reduction=self.reduction
        )
        loss_10 = self.cross_entropy(
            torch.hstack([sim_01.t(), sim_11]), targets, reduction=self.reduction
        )
        if self.reduction == "sum":
            return loss_01 + loss_10
        elif self.reduction == "mean":
            return 0.5 * (loss_01 + loss_10)
        else:  # None
            return torch.vstack([loss_01, loss_10])
