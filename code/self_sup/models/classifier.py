import torch
from torchvision.models import resnet18, resnet34, resnet50
from typing import Union
from contrastive import ContrastiveModel


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

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
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

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
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

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(inputs)  # N x num_classes


class CentroidClassifier(torch.nn.Module):
    def __init__(self, weights: torch.Tensor):
        """
        :param weights: The pre-computed weights of the classifier.
        """
        super(CentroidClassifier, self).__init__()
        self.weights = weights  # d x num_classes

    def forward(self, inputs) -> torch.Tensor:
        return torch.matmul(inputs, self.weights)  # N x num_classes

    @staticmethod
    def create_weights(data_loader, num_classes: int) -> torch.Tensor:
        """
        :param data_loader: Data loader of feature representation to create weights.
        :param num_classes: The number of classes.
        :return: Tensor contains weights.
        """

        X = data_loader.data
        Y = data_loader.targets

        weights = []
        for k in range(num_classes):
            ids = torch.where(Y == k)[0]
            weights.append(torch.mean(X[ids], dim=0))

        weights = torch.stack(weights, dim=1)  # d x num_classes
        return weights


class SupervisedModel(torch.nn.Module):
    def __init__(self, base_cnn: str = "resnet18", num_classes: int = 10) -> None:
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
            num_last_hidden_units = 512
        else:
            self.f = resnet18()
            num_last_hidden_units = 512

        self.f.fc = torch.nn.Linear(num_last_hidden_units, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:

        return self.f(inputs)


class ClassifierWithFeatureExtractor(torch.nn.Module):
    def __init__(
        self,
        feature_extractor: Union[SupervisedModel, ContrastiveModel],
        predictor: Union[LinearClassifier, NonLinearClassifier, NormalisedLinear],
    ) -> None:
        super(ClassifierWithFeatureExtractor, self).__init__()
        self.feature_extractor = feature_extractor
        self.predictor = predictor

    def make_ddp(self):
        pass

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(inputs)
        return self.predictor(features)
