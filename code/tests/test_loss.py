# import numpy as np
import torch
import pytest
from self_sup.loss import NT_Xent


@pytest.mark.parametrize("reduction", ("mean", "sum", "none"))
def test_expected_shape(reduction: str) -> None:
    batch_size = 32
    feature_dim = 128
    nt_xent = NT_Xent(reduction=reduction)
    feature_0 = torch.rand((batch_size, feature_dim))
    feature_1 = torch.rand((batch_size, feature_dim))
    loss = nt_xent(feature_0, feature_1)
    if reduction in {"sum", "mean"}:
        assert loss.size() == ()
    else:
        assert loss.size() == (2, batch_size)

@pytest.mark.parametrize("reduction", ("mean", "sum", "none"))
def test_positive_values(reduction: str) -> None:
    batch_size = 32
    feature_dim = 128
    nt_xent = NT_Xent(reduction=reduction)
    feature_0 = torch.rand((batch_size, feature_dim)) - 0.5
    feature_1 = torch.rand((batch_size, feature_dim)) - 0.5
    loss = nt_xent(feature_0, feature_1)
    assert all(loss.numpy().flatten() > 0)

@pytest.mark.parametrize("t", (-1, 0.))
def test_error_temperature(t: float) -> None:

    with pytest.raises(ValueError):
        NT_Xent(temperature=t)


def test_error_reduction() -> None:

    with pytest.raises(ValueError):
        NT_Xent(reduction="UNSUPPORTED REDUCTION")
