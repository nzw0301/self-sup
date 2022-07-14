import numpy
import pytest
import torch

from self_sup.models.classifier import (
    ClassifierWithFeatureExtractor,
    LinearClassifier,
    NonLinearClassifier,
    SupervisedModel,
)
from self_sup.models.contrastive import ContrastiveModel


@pytest.mark.parametrize(
    "model", [LinearClassifier, NonLinearClassifier, SupervisedModel]
)
def test_initialize_independent_classifier_correctly(model):
    model()


def update_params(model, data):
    optimizer = torch.optim.SGD(params=model.parameters(), lr=1.0)
    for _ in range(10):
        optimizer.zero_grad()
        output = model(data)
        loss = torch.nn.MSELoss()(output, torch.zeros_like(output))
        loss.backward()
        optimizer.step()


def eval_param(model, data):
    with torch.inference_mode():
        for _ in range(10):
            model(data)


def test_classifier_with_feature_extractor_frozen():
    mini_batch = torch.rand(2, 3, 32, 32)

    model = ClassifierWithFeatureExtractor(
        feature_extractor=ContrastiveModel(),
        predictor=LinearClassifier(512),
        frozen_feature_extractor=True,
    )
    model.train()
    # representative of unchanged_wights in feature extractor
    original_weight = numpy.array(
        model.feature_extractor.f.conv1.weight.detach().numpy()
    )
    original_moving_average = numpy.array(model.feature_extractor.f.bn1.running_mean)
    # representative of changeable weights in predictor
    original_predictor_weight = numpy.array(
        model.predictor.classifier.weight.detach().numpy()
    ).flatten()

    update_params(model, mini_batch)

    unchanged_weight = model.feature_extractor.f.conv1.weight.detach().numpy()
    unchanged_moving_average = (
        model.feature_extractor.f.bn1.running_mean.detach().numpy()
    )
    changed_predictor_weight = (
        model.predictor.classifier.weight.detach().numpy().flatten()
    )

    numpy.testing.assert_array_equal(original_weight, unchanged_weight)
    numpy.testing.assert_array_equal(original_moving_average, unchanged_moving_average)
    assert all(original_predictor_weight != changed_predictor_weight)


def test_classifier_with_feature_extractor_fine_tune():
    mini_batch = torch.rand(2, 3, 32, 32)

    model = ClassifierWithFeatureExtractor(
        feature_extractor=ContrastiveModel(),
        predictor=LinearClassifier(512),
        frozen_feature_extractor=False,
    )
    model.train()
    original_weight = numpy.array(
        model.feature_extractor.f.conv1.weight.detach().numpy()
    ).flatten()
    original_moving_average = numpy.array(
        model.feature_extractor.f.bn1.running_mean
    ).flatten()
    original_predictor_weight = numpy.array(
        model.predictor.classifier.weight.detach().numpy()
    ).flatten()

    update_params(model, mini_batch)

    changed_weight = model.feature_extractor.f.conv1.weight.detach().numpy().flatten()
    changed_moving_average = (
        model.feature_extractor.f.bn1.running_mean.detach().numpy().flatten()
    )
    changed_predictor_weight = (
        model.predictor.classifier.weight.detach().numpy().flatten()
    )

    assert all(original_weight != changed_weight)
    assert all(original_moving_average != changed_moving_average)
    assert all(original_predictor_weight != changed_predictor_weight)


@pytest.mark.parametrize("frozen_feature_extractor", [False, True])
def test_classifier_with_feature_extractor_eval(frozen_feature_extractor: bool):
    mini_batch = torch.rand(2, 3, 32, 32)

    model = ClassifierWithFeatureExtractor(
        feature_extractor=ContrastiveModel(),
        predictor=LinearClassifier(512),
        frozen_feature_extractor=frozen_feature_extractor,
    )
    model.eval()
    original_weight = numpy.array(
        model.feature_extractor.f.conv1.weight.detach().numpy()
    )
    original_moving_average = numpy.array(model.feature_extractor.f.bn1.running_mean)
    original_predictor_weight = numpy.array(
        model.predictor.classifier.weight.detach().numpy()
    )

    eval_param(model, mini_batch)

    unchanged_weight = model.feature_extractor.f.conv1.weight.detach().numpy()
    unchanged_moving_average = (
        model.feature_extractor.f.bn1.running_mean.detach().numpy()
    )
    unchanged_predictor_weight = model.predictor.classifier.weight.detach().numpy()

    numpy.testing.assert_array_equal(original_weight, unchanged_weight)
    numpy.testing.assert_array_equal(original_moving_average, unchanged_moving_average)
    numpy.testing.assert_array_equal(
        original_predictor_weight, unchanged_predictor_weight
    )
