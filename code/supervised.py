import os
from typing import Tuple

import hydra
import numpy as np
import torch
import torchvision
import wandb
from apex.parallel.LARC import LARC
from omegaconf import OmegaConf
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader

from self_sup.check_hydra_conf import check_hydra_conf
from self_sup.data.transforms import create_simclr_data_augmentation
from self_sup.data.utils import (
    create_data_loaders_from_datasets,
    get_train_val_test_datasets,
)
from self_sup.distributed_utils import init_ddp
from self_sup.logger import get_logger
from self_sup.lr_utils import calculate_initial_lr, calculate_warmup_lr
from self_sup.model import SupervisedModel


def validation(
    data_loader: DataLoader, model: SupervisedModel, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    :param data_loader: Data loader for a validation or test dataset.
    :param model: ResNet based classifier.
    :param device: `torch.device` instance to store the data and metrics.

    :return: validation loss and the number of corrected samples.
    """

    model.eval()

    sum_loss = torch.tensor([0.0], device=device)
    num_corrects = torch.tensor([0.0], device=device)

    with torch.no_grad():
        for data, targets in data_loader:
            data, targets = data.to(device), targets.to(device)
            unnormalized_features = model(data)
            loss = cross_entropy(unnormalized_features, targets, reduction="sum")

            predicted = torch.max(unnormalized_features.data, 1)[1]

            sum_loss += loss.item()
            num_corrects += (predicted == targets).sum()

    return sum_loss, num_corrects


@hydra.main(config_path="conf", config_name="supervised")
def main(cfg: OmegaConf):
    logger = get_logger()

    check_hydra_conf(cfg)

    local_rank = int(os.environ["LOCAL_RANK"]) if torch.cuda.is_available() else "cpu"
    init_ddp(cfg, local_rank)

    # for reproducibility
    seed = cfg["experiment"]["seed"]
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    rnd = np.random.RandomState(seed)

    logger.info("Using cuda:{}".format(local_rank))

    # initialise data loaders
    train_dataset, validation_dataset, test_dataset = get_train_val_test_datasets(
        rnd=rnd,
        root=cfg["dataset"]["root"],
        validation_ratio=cfg["dataset"]["validation_ratio"],
        dataset_name=cfg["dataset"]["name"],
        normalize=cfg["dataset"]["normalize"],
    )
    train_dataset.transform = create_simclr_data_augmentation(
        cfg["augmentation"]["strength"], size=cfg["augmentation"]["size"]
    )

    (
        train_data_loader,
        validation_data_loader,
        test_data_loader,
    ) = create_data_loaders_from_datasets(
        num_workers=cfg["experiment"]["num_workers"],
        train_batch_size=cfg["experiment"]["batch_size"],
        validation_batch_size=cfg["experiment"]["batch_size"],
        test_batch_size=cfg["experiment"]["batch_size"],
        ddp_sampler_seed=seed,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        test_dataset=test_dataset,
        distributed=True,
    )

    num_classes = len(np.unique(test_dataset.targets))
    num_train_samples_per_epoch = (
        cfg["experiment"]["batch_size"]
        * len(train_data_loader)
        * cfg["distributed"]["world_size"]
    )
    num_val_samples = len(validation_dataset)
    num_test_samples = len(test_dataset)

    if local_rank == 0:
        logger.info(
            f"#train: {len(train_dataset)}, #val: {num_val_samples}, #test: {num_test_samples}"
        )
        # TODO(nzw): init wandb
        wandb.init()

    model = SupervisedModel(
        base_cnn=cfg["architecture"]["base_cnn"], num_classes=num_classes,
    )
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    # TODO(nzw): add mixed precision

    # optimizer
    optimizer = torch.optim.SGD(
        params=model.parameters(),
        lr=calculate_initial_lr(cfg),
        momentum=cfg["optimizer"]["momentum"],
        nesterov=False,
        weight_decay=cfg["optimizer"]["decay"],
    )
    # https://github.com/google-research/simclr/blob/master/lars_optimizer.py#L26
    # optimizer = LARC(optimizer=optimizer, trust_coefficient=0.001, clip=False)

    # TODO(nzw): fix this part by following SWaV's way ir simsiam way.
    # num_gpus = cfg["distributed"]["world_size"]
    epochs = cfg["experiment"]["epochs"]
    best_metric = np.finfo(np.float64).max

    for epoch in range(epochs):
        # one training loop
        model.train()
        train_data_loader.sampler.set_epoch(epoch)
        sum_train_loss = torch.tensor([0.0], device=local_rank)

        for data, targets in train_data_loader:
            optimizer.zero_grad()
            data, targets = data.to(local_rank), targets.to(local_rank)
            loss = cross_entropy(model(data), targets)
            loss.backward()
            optimizer.step()
            sum_train_loss += loss.item()

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
            val_acc = num_val_corrects.item() / num_val_samples

            log_message = (
                f"Epoch:{epoch}/{epochs} progress:{progress:.3f} "
                f"train loss:{train_loss:.3f}, lr:{lr_for_logging:.7f}"
                f"val loss:{val_loss:.3f}, test loss:{val_acc:.3f}"
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
            if cfg["parameter"]["metric"] == "loss":
                metric = val_loss
            else:
                # store metric as risk: 1 - accuracy
                metric = 1.0 - val_acc

            if metric <= best_metric:
                save_fname = cfg["experiment"]["output_model_name"]
                torch.save(model.state_dict(), save_fname)
                best_metric = metric

            torch.distributed.barrier()

    # evaluate on test dataset
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
        wandb.save(str(save_fname))

    torch.distributed.barrier()


if __name__ == "__main__":
    """
    To run this code,
    `python launch.py --nproc_per_node={The number of GPUs on a single machine} supervised.py`
    """
    main()
