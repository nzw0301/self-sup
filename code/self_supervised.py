import logging
import os

import hydra
import numpy as np
import torch
from apex.parallel.LARC import LARC
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from self_sup.logger import get_logger
from self_sup.check_hydra_conf import check_hydra_conf
from self_sup.data.transforms import SimCLRTransforms
from self_sup.data.utils import create_data_loaders_from_datasets, fetch_dataset
from self_sup.distributed_utils import init_ddp
from self_sup.loss import NT_Xent
from self_sup.lr_utils import calculate_initial_lr, calculate_warmup_lr
from self_sup.model import ContrastiveModel


def exclude_from_wt_decay(
    named_params, weight_decay, skip_list=("bias", "bn")
) -> tuple:
    """
    :param named_params: Model's named_params.
    :param weight_decay: weight_decay's parameter.
    :param skip_list: list of names to exclude weight decay.
    :return: dictionaries
    """
    # https://github.com/nzw0301/pytorch-lightning-bolts/blob/master/pl_bolts/models/self_supervised/simclr/simclr_module.py#L90-L105
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
def main(cfg: OmegaConf):
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

    logger.info("Using {}".format(local_rank))

    train_dataset, _, _ = get_train_val_test_datasets(
        rnd=rnd,
        root=cfg["dataset"]["root"],
        validation_ratio=cfg["dataset"]["validation_ratio"],
        dataset_name=cfg["dataset"]["name"],
        normalize=cfg["dataset"]["normalize"],
    )
    train_dataset.transform = SimCLRTransforms(
        strength=cfg["dataset"]["strength"], size=cfg["dataset"]["size"]
    )

    train_data_loader = create_data_loaders_from_datasets(
        num_workers=cfg["experiment"]["num_workers"],
        train_batch_size=train_batch_size,
        ddp_sampler_seed=seed,
        train_dataset=train_dataset,
        distributed=True,
    )[0]

    train_batch_size = cfg["experiment"]["train_batch_size"]
    num_gpus = cfg["distributed"]["world_size"]
    num_train_samples_per_epoch = train_batch_size * len(train_data_loader) * num_gpus

    if local_rank == 0:
        logger.info("#train: {}".format(len(train_data_loader.dataset)))
        wandb.init(
            dir=hydra.utils.get_original_cwd(),
            project="self-sup",
            entity="nzw0301",
            config=flatten_omegaconf(cfg),
            tags=(cfg["dataset"]["name"], "simclr"),
            group="seed-{}".format(seed),
        )

    model = ContrastiveModel(
        base_cnn=cfg["architecture"]["base_cnn"],
        d=cfg["parameter"]["d"],
        is_cifar=is_cifar,
    )
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    model.train()
    simclr_loss_function = NT_Xent(
        temperature=cfg["loss"]["temperature"], device=local_rank
    )

    optimizer = torch.optim.SGD(
        params=exclude_from_wt_decay(
            model.named_parameters(), weight_decay=cfg["optimizer"]["decay"]
        ),
        lr=calculate_initial_lr(cfg),
        momentum=cfg["optimizer"]["momentum"],
        nesterov=False,
        weight_decay=0.0,
    )

    # https://github.com/google-research/simclr/blob/master/lars_optimizer.py#L26
    optimizer = LARC(optimizer=optimizer, trust_coefficient=0.001, clip=False)

    cos_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer.optim, T_max=total_steps - warmup_steps,
    )

    for epoch in range(1, epochs + 1):
        train_data_loader.sampler.set_epoch(epoch)

        for views, _ in train_data_loader:
            # adjust learning rate by applying linear warming
            if current_step <= warmup_steps:
                lr = calculate_warmup_lr(cfg, warmup_steps, current_step)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

            optimizer.zero_grad()
            zs = [model(view.to(local_rank)) for view in views]
            loss = simclr_loss_function(zs)

            loss.backward()
            optimizer.step()

            # adjust learning rate by applying cosine annealing
            if current_step > warmup_steps:
                cos_lr_scheduler.step()

            current_step += 1

        if local_rank == 0:
            logging.info(
                "Epoch:{}/{} progress:{:.3f} loss:{:.3f}, lr:{:.7f}".format(
                    epoch,
                    epochs,
                    epoch / epochs,
                    loss.item(),
                    optimizer.param_groups[0]["lr"],
                )
            )

            if epoch % cfg["experiment"]["save_model_epoch"] == 0 or epoch == epochs:
                save_fname = "epoch_{}-{}".format(
                    epoch, cfg["experiment"]["output_model_name"]
                )
                torch.save(model.state_dict(), save_fname)


if __name__ == "__main__":
    main()
