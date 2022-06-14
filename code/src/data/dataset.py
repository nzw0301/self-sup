import numpy as np
import torch
import torchvision


class DownstreamDataset(torch.utils.data.Dataset):
    def __init__(self, data, targets):
        assert len(data) == len(targets)

        self.data = data
        self.targets = targets

    def __getitem__(self, index: int) -> tuple:
        return self.data[index], self.targets[index]

    def __len__(self) -> int:
        return len(self.data)
