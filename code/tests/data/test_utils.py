import numpy as np
import pytest
from self_sup.data.utils import (
    _train_val_split,
    create_data_loaders_from_datasets,
    get_train_val_test_datasets,
)
from torchvision.datasets.cifar import CIFAR10

root = "~/pytorch_data"


def test_train_val_split():
    num_original_samples = 50_000
    seed = 7
    ratio = 0.1

    train_data = CIFAR10(root=root, download=True, train=True)
    train, val = _train_val_split(
        rnd=np.random.RandomState(seed),
        train_dataset=train_data,
        validation_ratio=ratio,
    )
    assert len(train) == int(num_original_samples * (1.0 - ratio))
    assert len(val) == int(num_original_samples * ratio)

    # Reproducibility
    diff_train, diff_val = _train_val_split(
        rnd=np.random.RandomState(seed),
        train_dataset=train_data,
        validation_ratio=ratio,
    )
    np.testing.assert_array_equal(diff_train.targets, train.targets)
    np.testing.assert_array_equal(diff_val.targets, val.targets)

    # Different seed generates different splits.
    diff_seed = seed + 1
    diff_train, diff_val = _train_val_split(
        rnd=np.random.RandomState(diff_seed),
        train_dataset=train_data,
        validation_ratio=ratio,
    )
    assert not np.array_equal(diff_train.targets, train.targets)
    assert not np.array_equal(diff_val.targets, val.targets)


def test_get_train_val_test_datasets():
    num_original_train_samples = 50_000
    num_original_test_samples = 10_000
    seed = 7
    ratio = 0.1
    rnd = np.random.RandomState(seed)

    with pytest.raises(ValueError):
        unsupported_dataset_name = "CIFAR130000"
        get_train_val_test_datasets(
            rnd=rnd, validation_ratio=ratio, dataset_name=unsupported_dataset_name
        )

    train, val, test = get_train_val_test_datasets(
        rnd=rnd, validation_ratio=ratio, dataset_name="cifar10"
    )
    assert len(train) == int(num_original_train_samples * (1.0 - ratio))
    assert len(val) == int(num_original_train_samples * ratio)
    assert len(test) == num_original_test_samples

    # For reproducibility.
    rnd = np.random.RandomState(seed)
    diff_train, diff_val, diff_test = get_train_val_test_datasets(
        rnd=rnd, validation_ratio=ratio, dataset_name="cifar10"
    )
    np.testing.assert_array_equal(diff_train.targets, train.targets)
    np.testing.assert_array_equal(diff_val.targets, val.targets)
    np.testing.assert_array_equal(diff_test.targets, test.targets)

    # Different seed generates different splits.
    rnd = np.random.RandomState(seed + 1)
    diff_train, diff_val, diff_test = get_train_val_test_datasets(
        rnd=rnd, validation_ratio=ratio, dataset_name="cifar10"
    )
    assert not np.array_equal(diff_train.targets, train.targets)
    assert not np.array_equal(diff_val.targets, val.targets)
    # test dataset should not change by train/val split.
    np.testing.assert_array_equal(diff_test.targets, test.targets)

    # if ratio = 0, validation set is None.
    train, val, test = get_train_val_test_datasets(
        rnd=rnd, validation_ratio=0.0, dataset_name="cifar10"
    )
    assert len(train) == num_original_train_samples
    assert val is None
    assert len(test) == num_original_test_samples


def test_create_data_loaders_from_datasets():
    num_original_train_samples = 50_000
    num_original_test_samples = 10_000
    seed = 7
    ratio = 0.1
    num_workers = 1
    rnd = np.random.RandomState(seed)
    train, val, test = get_train_val_test_datasets(
        rnd=rnd, validation_ratio=ratio, dataset_name="cifar10"
    )

    train_loader, val_loader, test_loader = create_data_loaders_from_datasets(
        num_workers=num_workers,
        train_batch_size=1,
        validation_batch_size=2,
        test_batch_size=3,
        train_dataset=train,
        validation_dataset=val,
        test_dataset=test,
        distributed=False,
    )

    assert len(train_loader.dataset) == int(num_original_train_samples * (1.0 - ratio))
    assert len(val_loader.dataset) == int(num_original_train_samples * ratio)
    assert len(test_loader.dataset) == num_original_test_samples
    assert train_loader.batch_size == 1
    assert val_loader.batch_size == 2
    assert test_loader.batch_size == 3

    # Only train dataloader.
    loaders = create_data_loaders_from_datasets(
        num_workers=num_workers,
        train_batch_size=1,
        train_dataset=train,
        distributed=False,
    )
    assert len(loaders) == 1
    assert len(loaders[0].dataset) == int(num_original_train_samples * (1.0 - ratio))
