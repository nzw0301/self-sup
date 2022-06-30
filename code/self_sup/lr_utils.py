import numpy as np


def calculate_scaled_lr(
    base_lr: float, batch_size: int, lr_schedule: str = "linear"
) -> float:
    """
    Proposed initial learning rates by SimCLR paper.
    Note: SimCLR paper says squared learning rate is better when the size of mini-batches is small.
    :return: Initial learning rate whose type is float.
    """

    assert base_lr > 0.0
    assert batch_size >= 1
    assert lr_schedule in {"linear", "square"}

    if lr_schedule == "linear":
        scaled_lr = base_lr * batch_size / 256.0
    else:
        scaled_lr = base_lr * np.sqrt(batch_size)

    return scaled_lr


def calculate_lr_list(
    lr: float, num_lr_updates_per_epoch: int, warmup_epochs: int, epochs: int
) -> np.ndarray:
    """
    scaling + linear warmup + cosine annealing without restart
    https://github.com/facebookresearch/swav/blob/master/main_swav.py#L178-L182
    Note that the first lr is 0.

    :param lr: base learning rate. This lr might be calculated by `calculate_scaled_lr` in self-supervised experiments.
    :param num_lr_updates_per_epoch: the number of iterations per epoch.
    :param warmup_epochs: the number of epochs for linear warmup.
    :param epochs: the number of total epochs including warmup.

    :return: np.ndarray of learning rates for all steps.

    NOTE:
        SimCLR and SWaV: num_lr_updates_per_epoch is the number of batches per epoch.
        SimSiam: num_lr_updates_per_epoch is 1.
    """

    assert lr > 0.0
    assert num_lr_updates_per_epoch > 0
    assert warmup_epochs >= 0
    assert epochs > 0

    warmup_lr_schedule = np.linspace(0, lr, num_lr_updates_per_epoch * warmup_epochs)
    num_non_warmup_epoch = epochs - warmup_epochs
    num_non_warmup_lr_updates = num_lr_updates_per_epoch * num_non_warmup_epoch
    iters = np.arange(num_lr_updates_per_epoch * (epochs - warmup_epochs))

    cosine_lr_schedule = np.array(
        [
            0.5 * lr * (1.0 + np.cos(np.pi * t / num_non_warmup_lr_updates))
            for t in iters
        ]
    )

    return np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
