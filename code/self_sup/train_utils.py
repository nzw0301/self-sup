import os
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import torch
import wandb
from omegaconf import OmegaConf
from torch.cuda.amp import GradScaler
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader

from self_sup.models.classifier import SupervisedModel
from self_sup.models.contrastive import CentroidClassifier, ContrastiveModel

logger = get_logger()


def centroid_eval(
    data_loader: DataLoader,
    device: torch.device,
    classifier: CentroidClassifier,
    top_k: int = 5,
) -> Tuple[float, float]:
    """
    :param data_loader: DataLoader of downstream task.
    :param device: PyTorch's device instance.
    :param classifier: Instance of MeanClassifier.
    :param top_k: The number of top-k to calculate accuracy.
    :return: Tuple of top-1 accuracy and top-k accuracy.
    """

    num_samples = len(data_loader.dataset)
    top_1_correct = 0
    top_k_correct = 0

    classifier.eval()
    with torch.no_grad():
        for x, y in data_loader:
            y = y.to(device)
            pred_top_k = torch.topk(classifier(x.to(device)), dim=1, k=top_k)[1]
            pred_top_1 = pred_top_k[:, 0]

            top_1_correct += pred_top_1.eq(y.view_as(pred_top_1)).sum().item()
            if top_k > 1:
                top_k_correct += (pred_top_k == y.view(len(y), 1)).sum().item()

    return top_1_correct / num_samples, top_k_correct / num_samples


def convert_vectors(
    data_loader: torch.utils.data.DataLoader,
    model: ContrastiveModel,
    device: torch.device,
    normalized: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert experiment to feature representations.
    :param data_loader: Tata loader for raw experiment.
    :param model: Pre-trained model.
    :param device: PyTorch's device instance.
    :param normalized: Whether normalize the feature representation or not.

    :return: Tuple of tensors: features and labels.
    """

    new_X = []
    new_y = []
    model.eval()

    with torch.no_grad():
        for x_batches, y_batches in data_loader:
            x_batches = x_batches.to(device)
            fs = model(x_batches)
            if normalized:
                fs = torch.nn.functional.normalize(fs, p=2, dim=1)
            new_X.append(fs)
            new_y.append(y_batches)

    X = torch.cat(new_X)
    y = torch.cat(new_y)

    return X, y


def validation(
    data_loader: DataLoader, model: SupervisedModel, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    :param data_loader: Data loader for a validation or test dataset.
    :param model: ResNet based classifier.
    :param device: A `torch.device` instance to store the data and metrics.

    :return: Categorical cross-entropy loss and the number of correctly predicted samples.
    """

    model.eval()

    sum_loss = torch.tensor([0.0], device=device)
    num_corrects = torch.tensor([0.0], device=device)

    with torch.inference_mode():
        for data, targets in data_loader:
            data, targets = data.to(device), targets.to(device)
            logits = model(data)  # (mini-batch-size, #classes)
            loss = cross_entropy(
                logits, targets, reduction="sum"
            )  # (mini-batch-size, )

            predicted = torch.max(logits.data, dim=1)[1]  # (mini-batch-size, )

            sum_loss += loss
            num_corrects += (predicted == targets).sum()

    return sum_loss, num_corrects


def supervised_training(
    model,
    train_data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    lr_list: Sequence[float],
    validation_data_loader: DataLoader,
    local_rank,
    cfg: OmegaConf,
) -> None:
    """
    TODO (nzw0301): this looks like "fit" method on some abstract DL package...
    """
    epochs = cfg["experiment"]["epochs"]
    train_batch_size = cfg["experiment"]["train_batch_size"]
    save_fname = Path(os.getcwd()) / cfg["experiment"]["output_model_name"]

    num_train_samples_per_epoch = (
        train_batch_size * len(train_data_loader) * cfg["distributed"]["world_size"]
    )
    num_val_samples = len(validation_data_loader.dataset)
    scaler = GradScaler()
    best_metric = np.finfo(np.float64).max

    for epoch in range(epochs):
        # one training loop
        model.train()
        train_data_loader.sampler.set_epoch(epoch)
        sum_train_loss = torch.tensor([0.0], device=local_rank)

        # update learning rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_list[epoch]

        for data, targets in train_data_loader:
            optimizer.zero_grad()
            data, targets = data.to(local_rank), targets.to(local_rank)
            with torch.autocast("cuda"):
                loss = cross_entropy(model(data), targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            sum_train_loss += loss.item() * train_batch_size

        torch.distributed.reduce(sum_train_loss, dst=0)

        # validation
        sum_val_loss, num_val_corrects = validation(
            validation_data_loader, model, local_rank
        )

        torch.distributed.reduce(sum_val_loss, dst=0)
        torch.distributed.reduce(num_val_corrects, dst=0)

        # logging, send values to wandb, and save checkpoint,
        if local_rank == 0:
            # logging
            progress = (epoch + 1) / (epochs + 1)
            lr_for_logging = optimizer.param_groups[0]["lr"]
            train_loss = sum_train_loss.item() / num_train_samples_per_epoch
            val_loss = sum_val_loss.item() / num_val_samples
            val_acc = num_val_corrects.item() / num_val_samples * 100.0

            log_message = (
                f"Epoch:{epoch}/{epochs} progress:{progress:.3f}, "
                f"train loss:{train_loss:.3f}, lr:{lr_for_logging:.7f}, "
                f"val loss:{val_loss:.3f}, val acc:{val_acc:.3f}"
            )
            logger.info(log_message)

            # send metrics to wandb
            wandb.log(
                data={
                    "supervised_train_loss": train_loss,
                    "supervised_val_loss": val_loss,
                    "supervised_val_acc": val_acc,
                },
                step=epoch,
            )

            # save checkpoint if metric improves.
            if cfg["experiment"]["metric"] == "loss":
                metric = val_loss
            else:
                # store metric as risk: 1 - accuracy
                metric = 1.0 - val_acc

            if metric <= best_metric:
                torch.save(model.state_dict(), save_fname)
                best_metric = metric

        torch.distributed.barrier()


def test_eval(model, test_data_loader: DataLoader, local_rank, save_fname: str) -> None:
    num_test_samples = len(test_data_loader.dataset)

    # evaluate the best performed checkpoint on the test dataset
    torch.distributed.barrier()
    map_location = {"cuda:%d" % 0: "cuda:%d" % local_rank}
    model.load_state_dict(torch.load(save_fname, map_location=map_location))
    test_loss, test_num_corrects = validation(test_data_loader, model, local_rank)

    torch.distributed.reduce(test_loss, dst=0)
    torch.distributed.reduce(test_num_corrects, dst=0)

    if local_rank == 0:
        test_loss = test_loss.item() / num_test_samples
        test_acc = test_num_corrects.item() / num_test_samples * 100.0

        wandb.run.summary["supervised_test_loss"] = test_loss
        wandb.run.summary["supervised_test_acc"] = test_acc

        log_message = f"test loss:{test_loss:.3f}, test acc:{test_acc:.3f}"
        logger.info(log_message)

    torch.distributed.barrier()
