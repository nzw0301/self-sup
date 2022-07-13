import logging
import os
from pathlib import Path
from typing import Dict, Iterator, Sequence, Tuple, Union

import hydra
import numpy as np
import torch
import wandb
from apex.parallel.LARC import LARC
from omegaconf import OmegaConf
from self_sup.data.transforms import SimCLRTransforms
from self_sup.data.utils import (
    create_data_loaders_from_datasets,
    get_train_val_test_datasets,
)
from self_sup.distributed_utils import init_ddp
from self_sup.logger import get_logger
from self_sup.loss import NT_Xent
from self_sup.lr_utils import calculate_lr_list, calculate_scaled_lr
from self_sup.models.contrastive import ContrastiveModel
from self_sup.models.head import ProjectionHead
from self_sup.wandb_utils import flatten_omegaconf
from torch.cuda.amp import GradScaler


def exclude_from_wt_decay(
    named_params: Iterator,
    weight_decay: float,
    skip_list: Sequence[str] = ("bias", "bn", "projection_head"),
) -> Tuple[
    Dict[str, Union[float, torch.nn.parameter.Parameter]],
    Dict[str, Union[float, torch.nn.parameter.Parameter]],
]:
    """
    :param named_params: Model's named params. Usually, `model.named_parameters()`.
    :param weight_decay: weight_decay's parameter.
    :param skip_list: Sequence of names to exclude weight decay, the coefficient is zero.
    :return: Tuple of two dictionaries to specify the weight decay's co-efficient.
    """
    # Based on https://github.com/nzw0301/pytorch-lightning-bolts/blob/master/pl_bolts/models/self_supervised/simclr/simclr_module.py#L90-L105,
    # https://github.com/google-research/simclr/blob/3fb622131d1b6dee76d0d5f6aac67db84dab3800/model_util.py#L99

    params = []
    excluded_params = []

    for name, param in named_params:

        if not param.requires_grad:
            continue
        elif any(layer_name in name for layer_name in skip_list):
            excluded_params.append(param)
        else:
            params.append(param)

    return (
        {"params": params, "weight_decay": weight_decay},
        {"params": excluded_params, "weight_decay": 0.0},
    )


@hydra.main(config_path="conf", config_name="simclr")
def main(cfg: OmegaConf) -> None:
    logger = get_logger()

    local_rank = int(os.environ["LOCAL_RANK"]) if torch.cuda.is_available() else "cpu"
    init_ddp(cfg, local_rank)

    # for reproducibility
    seed = cfg["experiment"]["seed"]
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    rnd = np.random.RandomState(seed)

    logger.info("Using cuda:{}".format(local_rank))

    train_dataset, _, _ = get_train_val_test_datasets(
        rnd=rnd,
        root=cfg["dataset"]["root"],
        validation_ratio=cfg["dataset"]["validation_ratio"],
        dataset_name=cfg["dataset"]["name"],
        normalize=cfg["dataset"]["normalize"],
    )
    train_dataset.transform = SimCLRTransforms(
        strength=cfg["augmentation"]["strength"], size=cfg["augmentation"]["size"],
    )

    train_batch_size = cfg["experiment"]["train_batch_size"]
    train_data_loader = create_data_loaders_from_datasets(
        num_workers=cfg["experiment"]["num_workers"],
        train_batch_size=train_batch_size,
        ddp_sampler_seed=seed,
        train_dataset=train_dataset,
        distributed=True,
    )[0]

    num_gpus = cfg["distributed"]["world_size"]
    num_update_per_epoch = len(train_data_loader)
    num_train_samples_per_epoch = train_batch_size * num_update_per_epoch * num_gpus

    is_cifar = "cifar" in cfg["dataset"]["name"]

    if local_rank == 0:
        logger.info("#train: {}".format(len(train_data_loader.dataset)))
        wandb.init(
            dir=hydra.utils.get_original_cwd(),
            project="self-sup",
            entity="nzw0301",
            config=flatten_omegaconf(cfg),
            tags=(cfg["dataset"]["name"], "SimCLR"),
            group="seed-{}".format(seed),
        )

    model = ContrastiveModel(
        base_cnn=cfg["backbone"]["name"],
        head=ProjectionHead(
            input_dim=2048 if cfg["backbone"]["name"] == "resnet50" else 512,
            latent_dim=cfg["projection_head"]["d"],
            num_non_linear_blocks=cfg["projection_head"]["num_hidden_layer"],
        ),
        is_cifar=is_cifar,
    )
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    simclr_loss_function = NT_Xent(
        temperature=cfg["loss"]["temperature"], device=local_rank
    )

    epochs = cfg["experiment"]["epochs"]
    init_lr = calculate_scaled_lr(
        base_lr=cfg["optimizer"]["lr"],
        batch_size=train_batch_size,
        lr_schedule=cfg["lr_scheduler"]["name"],
    )
    optimizer = torch.optim.SGD(
        params=exclude_from_wt_decay(
            model.named_parameters(), weight_decay=cfg["optimizer"]["decay"]
        ),
        lr=init_lr,
        momentum=cfg["optimizer"]["momentum"],
    )
    lr_list = calculate_lr_list(
        init_lr,
        num_lr_updates_per_epoch=num_update_per_epoch,
        warmup_epochs=cfg["lr_scheduler"]["warmup_epochs"],
        epochs=epochs,
    )

    # https://github.com/google-research/simclr/blob/master/lars_optimizer.py#L26
    optimizer = LARC(optimizer=optimizer, trust_coefficient=0.001, clip=False)

    scaler = GradScaler()

    for epoch in range(epochs):
        model.train()
        train_data_loader.sampler.set_epoch(epoch)
        sum_train_loss = torch.tensor([0.0], device=local_rank)

        for i, (views, _) in enumerate(train_data_loader):

            # Adjust learning rate by applying linear warming
            # update learning rate.
            lr_index = epoch * num_update_per_epoch + i
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_list[lr_index]

            optimizer.zero_grad()

            views = [view.to(local_rank) for view in views]

            with torch.autocast("cuda"):
                z_0 = model(views[0])
                z_1 = model(views[1])
                loss = simclr_loss_function(z_0, z_1)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            sum_train_loss += loss.item() * train_batch_size

        torch.distributed.reduce(sum_train_loss, dst=0)

        if local_rank == 0:
            progress = (epoch + 1) / (epochs)
            avg_train_loss = sum_train_loss.item() / num_train_samples_per_epoch
            lr_for_logging = optimizer.param_groups[0]["lr"]
            logging.info(
                f"Epoch:{epoch+1}/{epochs} progress:{progress:.3f} loss:{avg_train_loss:.3f}, "
                f"lr:{lr_for_logging:.7f}"
            )

            # send metrics to wandb
            wandb.log(
                data={"train_contrastive_loss": avg_train_loss}, step=epoch,
            )

    if local_rank == 0:
        save_fname = Path(os.getcwd()) / cfg["experiment"]["output_model_name"]
        torch.save(model.state_dict(), save_fname)

    torch.distributed.barrier()


if __name__ == "__main__":
    main()
