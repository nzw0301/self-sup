import copy
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split as sk_train_val_split
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100


def create_data_loaders_from_datasets(
    num_workers: int,
    train_batch_size: Optional[int] = None,
    validation_batch_size: Optional[int] = None,
    test_batch_size: Optional[int] = None,
    ddp_sampler_seed: int = 0,
    train_dataset: Optional[torch.utils.data.Dataset] = None,
    validation_dataset: Optional[torch.utils.data.Dataset] = None,
    test_dataset: Optional[torch.utils.data.Dataset] = None,
    distributed: bool = True,
) -> List[torch.utils.data.DataLoader]:
    """
    :param num_workers: The number of workers for returned dataloaders.
    :param train_batch_size: The mini-batch size of train dataloader.
    :param validation_batch_size: The mini-batch size of validation dataloader.
    :param test_batch_size: The mini-batch size of test dataloader.
    :param ddp_sampler_seed: Random seed value of ddp sampler
    :param train_dataset: Dataset instance for training dataset.
    :param validation_dataset: Dataset instance for validation dataset.
    :param test_dataset: Dataset instance for test dataset.
    :param distributed: Whether or not to use DistributedSampler in dataloader.

    :return: list of DataLoaders.
    """
    data_loaders = []

    def _get_data_loader(
        dataset: torch.utils.data.Dataset,
        num_workers: int,
        batch_size: int,
        ddp_sampler_seed: int,
        distributed: bool,
        drop_last: bool,
    ) -> torch.utils.data.DataLoader:

        # https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler
        sampler = (
            torch.utils.data.distributed.DistributedSampler(
                dataset, shuffle=True, drop_last=drop_last, seed=ddp_sampler_seed
            )
            if distributed
            else None
        )

        return DataLoader(
            dataset=dataset,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=num_workers,
            batch_size=batch_size,
            pin_memory=True,
            drop_last=drop_last,
        )

    if train_dataset is not None:
        data_loaders.append(
            _get_data_loader(
                dataset=train_dataset,
                num_workers=num_workers,
                batch_size=train_batch_size,
                ddp_sampler_seed=ddp_sampler_seed,
                distributed=distributed,
                drop_last=True,
            )
        )

    for dataset, batch_size in zip(
        (validation_dataset, test_dataset), (validation_batch_size, test_batch_size)
    ):
        if dataset is not None:
            data_loaders.append(
                _get_data_loader(
                    dataset=dataset,
                    num_workers=num_workers,
                    batch_size=batch_size,
                    ddp_sampler_seed=ddp_sampler_seed,
                    distributed=distributed,
                    drop_last=False,
                )
            )

    return data_loaders


def _train_val_split(
    rnd: np.random.RandomState,
    train_dataset: torch.utils.data.Dataset,
    validation_ratio: float = 0.05,
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """
    Apply sklearn's `train_val_split` function to PyTorch's dataset instance.
    :param rnd: `np.random.RandomState` instance for reproducibility of train/val split.
    :param train_dataset: Training set that is an instance of PyTorch's dataset.
    :param validation_ratio: The ratio of validation data.
    :return: Tuple of training set and validation set.
    """

    assert 0.0 < validation_ratio < 1, "validation_ratio ratio should be (0, 1)"

    x_train, x_val, y_train, y_val = sk_train_val_split(
        train_dataset.data,
        train_dataset.targets,
        test_size=validation_ratio,
        random_state=rnd,
        stratify=train_dataset.targets,
    )

    sampled_train_dataset = copy.deepcopy(train_dataset)
    val_dataset = copy.deepcopy(train_dataset)

    sampled_train_dataset.data = x_train
    sampled_train_dataset.targets = y_train

    val_dataset.data = x_val
    val_dataset.targets = y_val

    return sampled_train_dataset, val_dataset


def get_train_val_test_datasets(
    rnd: np.random.RandomState,
    root: str = "~/pytorch_datasets",
    validation_ratio: float = 0.05,
    dataset_name: str = "cifar100",
    normalize: bool = False,
) -> Tuple[
    Optional[Union[CIFAR10, CIFAR100]],
    Optional[Union[CIFAR10, CIFAR100]],
    Optional[Union[CIFAR10, CIFAR100]],
]:
    """
    Create CIFAR-10/100 train/val/test data loaders

    :param rnd: `np.random.RandomState` instance.
    :param validation_ratio: The ratio of validation data. If this value is `0.`, returned `val_set` is `None`.
    :param root: Path to save data.
    :param dataset_name: The name if dataset. cifar10 or cifar100
    :param normalize: flag to perform channel-wise normalization as pre-processing.

    :return: Tuple of (train, val, test).
    """

    if dataset_name not in {"cifar10", "cifar100"}:
        raise ValueError

    transform = transforms.Compose([transforms.ToTensor(),])

    if dataset_name == "cifar10":
        DataSet = CIFAR10
    else:
        DataSet = CIFAR100

    train_dataset = DataSet(root=root, train=True, download=True, transform=transform)

    # create validation split
    if validation_ratio > 0.0:
        train_dataset, val_dataset = _train_val_split(
            rnd=rnd, train_dataset=train_dataset, validation_ratio=validation_ratio
        )
    else:
        val_dataset = None

    if normalize:
        # create a transform to do pre-processing
        train_loader = DataLoader(
            train_dataset, batch_size=len(train_dataset), shuffle=False
        )

        data = iter(train_loader).next()
        dim = [0, 2, 3]
        mean = data[0].mean(dim=dim).numpy()
        std = data[0].std(dim=dim).numpy()
        # end of creating a transform to do pre-processing

        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std),]
        )

        train_dataset.transform = transform

    if val_dataset is not None:
        val_dataset.transform = transform

    test_dataset = DataSet(root=root, train=False, download=True, transform=transform)

    return train_dataset, val_dataset, test_dataset
