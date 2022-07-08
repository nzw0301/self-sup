import pytest
import torch
import numpy as np
from self_sup.loss import NT_Xent


@pytest.mark.parametrize("reduction", ("mean", "sum", "none"))
def test_expected_shape(reduction: str) -> None:
    batch_size = 32
    feature_dim = 128
    feature_0 = torch.rand((batch_size, feature_dim))
    feature_1 = torch.rand((batch_size, feature_dim))
    loss = NT_Xent(reduction=reduction)(feature_0, feature_1)
    if reduction in {"sum", "mean"}:
        assert loss.size() == ()
    else:
        assert loss.size() == (2, batch_size)


@pytest.mark.parametrize("reduction", ("mean", "sum", "none"))
def test_positive_values(reduction: str) -> None:
    batch_size = 32
    feature_dim = 128
    feature_0 = torch.rand((batch_size, feature_dim)) - 0.5
    feature_1 = torch.rand((batch_size, feature_dim)) - 0.5
    loss = NT_Xent(reduction=reduction)(feature_0, feature_1)
    assert all(loss.numpy().flatten() > 0)


@pytest.mark.parametrize("reduction", ("mean", "sum"))
def test_exchangeable_argument_forward(reduction: str) -> None:
    batch_size = 32
    feature_dim = 128
    nt_xent_loss = NT_Xent(reduction=reduction)
    feature_0 = torch.rand((batch_size, feature_dim))
    feature_1 = torch.rand((batch_size, feature_dim))
    loss_01 = nt_xent_loss(feature_0, feature_1)
    loss_10 = nt_xent_loss(feature_1, feature_0)
    assert loss_01 == loss_10


def test_non_exchangeable_argument_forward() -> None:
    batch_size = 32
    feature_dim = 128
    nt_xent_loss = NT_Xent(reduction="none")
    feature_0 = torch.rand((batch_size, feature_dim))
    feature_1 = torch.rand((batch_size, feature_dim))
    loss_01 = nt_xent_loss(feature_0, feature_1)
    loss_10 = nt_xent_loss(feature_1, feature_0)
    assert not np.array_equal(loss_01, loss_10)
    np.testing.assert_array_equal(loss_01[0], loss_10[1])
    np.testing.assert_array_equal(loss_01[1], loss_10[0])


@pytest.mark.parametrize("t", (-1.0, 0.0))
def test_error_temperature(t: float) -> None:

    with pytest.raises(ValueError):
        NT_Xent(temperature=t)


def test_error_reduction() -> None:

    with pytest.raises(ValueError):
        NT_Xent(reduction="UNSUPPORTED REDUCTION")
