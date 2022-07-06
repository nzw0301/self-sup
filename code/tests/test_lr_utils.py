import numpy as np
import pytest
from self_sup.lr_utils import calculate_lr_list, calculate_scaled_lr


@pytest.mark.parametrize("lr_schedule_name", ("linear", "square"))
def test_calculate_scaled_lr(lr_schedule_name: str) -> None:
    lr = 0.1
    batch_size = 32
    scaled_lr = calculate_scaled_lr(
        base_lr=lr, batch_size=batch_size, lr_schedule=lr_schedule_name,
    )
    assert scaled_lr > 0.0


def test_calculate_lr_list_simsiam_small_minibatch() -> None:
    # Case 1: simsiam way's small mini-batch: update lr by epoch + no-warmup
    lr = 0.1
    epochs = 10
    warmup_epochs = 0
    num_lr_updates_per_epoch = 1

    lrs = calculate_lr_list(lr, num_lr_updates_per_epoch, warmup_epochs, epochs)

    assert len(lrs) == epochs * num_lr_updates_per_epoch
    # lr monotonically decreases
    assert all(np.diff(lrs) < 0.0)


def test_calculate_lr_list_simsiam_large_minibatch() -> None:
    # Case 2: simsiam way's large mini-batch: update lr by epoch + warmup
    lr = 0.1
    epochs = 10
    warmup_epochs = 3
    num_lr_updates_per_epoch = 1

    lrs = calculate_lr_list(lr, num_lr_updates_per_epoch, warmup_epochs, epochs)

    assert len(lrs) == epochs * num_lr_updates_per_epoch
    # linear warmup period: lr monotonically increases
    expected_linear_warmup = [0, lr / 2.0, lr]
    np.testing.assert_array_equal(expected_linear_warmup, lrs[:warmup_epochs])
    # cosine period: lr monotonically decreases
    assert all(np.diff(lrs[warmup_epochs:]) < 0.0)


def test_calculate_lr_list_simclr() -> None:
    # case 3: SimCLR/SWaV style: update lr by iteration + warmup
    lr = 0.1
    epochs = 10
    warmup_epochs = 3
    num_lr_updates_per_epoch = 20

    lrs = calculate_lr_list(lr, num_lr_updates_per_epoch, warmup_epochs, epochs)

    final_warmpup_epoch = warmup_epochs * num_lr_updates_per_epoch
    assert len(lrs) == epochs * num_lr_updates_per_epoch
    # linear warmup period: lr monotonically increases
    assert all(np.diff(lrs[:final_warmpup_epoch]) > 0.0)
    # cosine period: lr monotonically decreases
    assert all(np.diff(lrs[final_warmpup_epoch:]) < 0.0)
