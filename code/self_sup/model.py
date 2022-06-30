from collections import OrderedDict

import torch
from torchvision.models import resnet18, resnet34, resnet50


class NonLinearClassifier(torch.nn.Module):
    def __init__(
        self, num_features: int = 128, num_hidden: int = 128, num_classes: int = 10
    ):
        super(NonLinearClassifier, self).__init__()

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(num_features, num_hidden),
            torch.nn.BatchNorm1d(num_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(num_hidden, num_classes),
        )

    def forward(self, inputs: torch.FloatTensor) -> torch.FloatTensor:
        """
        Return Unnormalized probabilities

        :param inputs: Mini-batches of feature representation.
        :return: Unnormalized probabilities.
        """

        return self.classifier(inputs)  # N x num_classes


class NormalisedLinear(torch.nn.Linear):
    """
    Linear module with normalized weights.
    """

    def forward(self, input) -> torch.FloatTensor:
        w = torch.nn.functional.normalize(self.weight, dim=1, p=2)
        return torch.nn.functional.linear(input, w, self.bias)


class LinearClassifier(torch.nn.Module):
    def __init__(
        self, num_features: int = 128, num_classes: int = 10, normalize: bool = True
    ):
        """
        Linear classifier for linear evaluation protocol.

        :param num_features: The dimensionality of feature representation
        :param num_classes: The number of supervised class
        :param normalize: Whether feature is normalized or not.
        """

        super(LinearClassifier, self).__init__()
        if normalize:
            self.classifier = NormalisedLinear(num_features, num_classes, bias=False)
        else:
            self.classifier = torch.nn.Linear(num_features, num_classes)

    def forward(self, inputs: torch.FloatTensor) -> torch.FloatTensor:
        return self.classifier(inputs)  # N x num_classes


class CentroidClassifier(torch.nn.Module):
    def __init__(self, weights: torch.FloatTensor):
        """
        :param weights: The pre-computed weights of the classifier.
        """
        super(CentroidClassifier, self).__init__()
        self.weights = weights  # d x num_classes

    def forward(self, inputs) -> torch.FloatTensor:
        return torch.matmul(inputs, self.weights)  # N x num_classes

    @staticmethod
    def create_weights(data_loader, num_classes: int) -> torch.FloatTensor:
        """
        :param data_loader: Data loader of feature representation to create weights.
        :param num_classes: The number of classes.
        :return: FloatTensor contains weights.
        """

        X = data_loader.data
        Y = data_loader.targets

        weights = []
        for k in range(num_classes):
            ids = torch.where(Y == k)[0]
            weights.append(torch.mean(X[ids], dim=0))

        weights = torch.stack(weights, dim=1)  # d x num_classes
        return weights


class ProjectionHead(torch.nn.Module):
    def __init__(self, num_last_hidden_units: int, d: int):
        """
        :param num_last_hidden_units: the dimensionality of the encoder's output representation.
        :param d: the dimensionality of output.

        """
        super(ProjectionHead, self).__init__()

        self.projection_head = torch.nn.Sequential(
            OrderedDict(
                [
                    (
                        "linear1",
                        torch.nn.Linear(num_last_hidden_units, num_last_hidden_units),
                    ),
                    ("bn1", torch.nn.BatchNorm1d(num_last_hidden_units)),
                    ("relu1", torch.nn.ReLU()),
                    ("linear2", torch.nn.Linear(num_last_hidden_units, d, bias=False)),
                ]
            )
        )

    def forward(self, inputs: torch.FloatTensor) -> torch.FloatTensor:
        return self.projection_head(inputs)


class ContrastiveModel(torch.nn.Module):
    def __init__(self, base_cnn: str = "resnet18", d: int = 128, is_cifar: bool = True):
        """
        :param base_cnn: The backbone's model name. resnet18 or resnet50.
        :param d: The dimensionality of the output feature.
        :param is_cifar:
            model is for CIFAR10/100 or not.
            If it is `True`, network is modified by following SimCLR's experiments.
        """

        assert base_cnn in {"resnet18", "resnet50"}
        super(ContrastiveModel, self).__init__()

        if base_cnn == "resnet50":
            self.f = resnet50()
            num_last_hidden_units = 2048
        elif base_cnn == "resnet18":
            self.f = resnet18()
            num_last_hidden_units = 512

            if is_cifar:
                # replace the first conv2d with smaller conv
                self.f.conv1 = torch.nn.Conv2d(
                    in_channels=3,
                    out_channels=64,
                    stride=1,
                    kernel_size=3,
                    padding=3,
                    bias=False,
                )

                # remove the first max pool
                self.f.maxpool = torch.nn.Identity()
        else:
            raise ValueError(
                "`base_cnn` must be either `resnet18` or `resnet50`. `{}` is unsupported.".format(
                    base_cnn
                )
            )

        # drop the last classification layer
        self.f.fc = torch.nn.Identity()

        # non-linear projection head
        self.g = ProjectionHead(num_last_hidden_units, d)

    def encode(self, inputs: torch.FloatTensor) -> torch.FloatTensor:
        """
        return features before projection head.
        :param inputs: FloatTensor that contains images.
        :return: feature representations.
        """

        return self.f(inputs)  # N x num_last_hidden_units

    def forward(self, inputs: torch.FloatTensor) -> torch.FloatTensor:

        h = self.encode(inputs)
        z = self.g(h)
        return z  # N x d


class SupervisedModel(torch.nn.Module):
    def __init__(self, base_cnn: str = "resnet18", num_classes: int = 10):
        """Instantiate ResNet-{18,34,50} as a supervised classifier.

        Args:
            base_cnn (str): name of backbone model. Defaults to "resnet18".
            num_classes (int): the number of supervised classes. Defaults to 10.
        """

        assert base_cnn in {"resnet18", "resnet34", "resnet50"}
        super(SupervisedModel, self).__init__()

        if base_cnn == "resnet50":
            self.f = resnet50()
            num_last_hidden_units = 2048
        elif base_cnn == "resnet34":
            self.f = resnet34()
            pass
            # num_last_hidden_units =
        else:
            self.f = resnet18()
            num_last_hidden_units = 512

        self.f.fc = torch.nn.Linear(num_last_hidden_units, num_classes)

    def forward(self, inputs: torch.FloatTensor) -> torch.FloatTensor:

        return self.f(inputs)


def modify_resnet_by_simclr_for_cifar(model: SupervisedModel) -> SupervisedModel:
    """By following SimCLR v1 paper, this function replaces a few layers for CIFAR-10 experiments.

    Args:
        model: Instance of `SupervisedModel`.

    Returns:
        SupervisedModel: Modified `SupervisedModel`.
    """

    # replace the first conv2d with smaller conv
    model.f.conv1 = torch.nn.Conv2d(
        in_channels=3, out_channels=64, stride=1, kernel_size=3, padding=3, bias=False,
    )

    # remove the first max pool
    model.f.maxpool = torch.nn.Identity()
    return model
