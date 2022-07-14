import os
from pathlib import Path

import hydra
import numpy as np
import torch
import wandb
from omegaconf import OmegaConf

from self_sup.data.transforms import get_data_augmentation
from self_sup.data.utils import (
    create_data_loaders_from_datasets,
    get_train_val_test_datasets,
)
from self_sup.distributed_utils import init_ddp
from self_sup.logger import get_logger
from self_sup.lr_utils import calculate_lr_list, calculate_scaled_lr
from self_sup.models.classifier import SupervisedModel
from self_sup.models.utils import modify_resnet_by_simclr_for_cifar
from self_sup.train_utils import supervised_training, test_eval
from self_sup.wandb_utils import flatten_omegaconf

logger = get_logger()


# TODO (nzw0301):Version_base=None changes working dir somehow, so remove it as a hotfix...
@hydra.main(config_path="conf", config_name="supervised")
def main(cfg: OmegaConf):
    # check_hydra_conf(cfg)

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

    # initialise data loaders.
    train_dataset, validation_dataset, test_dataset = get_train_val_test_datasets(
        rnd=rnd,
        root=cfg["dataset"]["root"],
        validation_ratio=cfg["dataset"]["validation_ratio"],
        dataset_name=cfg["dataset"]["name"],
        normalize=cfg["dataset"]["normalize"],
    )
    train_dataset.transform = get_data_augmentation(cfg["augmentation"])

    train_batch_size = cfg["experiment"]["train_batch_size"]
    (
        train_data_loader,
        validation_data_loader,
        test_data_loader,
    ) = create_data_loaders_from_datasets(
        num_workers=cfg["experiment"]["num_workers"],
        train_batch_size=train_batch_size,
        validation_batch_size=cfg["experiment"]["eval_batch_size"],
        test_batch_size=cfg["experiment"]["eval_batch_size"],
        ddp_sampler_seed=seed,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        test_dataset=test_dataset,
        distributed=True,
    )

    num_classes = len(np.unique(test_dataset.targets))

    num_test_samples = len(test_dataset)

    if validation_dataset is None:
        validation_data_loader = test_data_loader
        logger.info(f"NOTE: Use test dataset as validation dataset too.")

    num_val_samples = len(validation_data_loader.dataset)

    if local_rank == 0:
        logger.info(
            f"#train: {len(train_dataset)}, #val: {num_val_samples}, #test: {num_test_samples}"
        )
        wandb.init(
            dir=hydra.utils.get_original_cwd(),
            project="self-sup",
            entity="nzw0301",
            config=flatten_omegaconf(cfg),
            tags=(cfg["dataset"]["name"], "supervised"),
            group="seed-{}".format(seed),
        )

    model = SupervisedModel(base_cnn=cfg["backbone"]["name"], num_classes=num_classes)
    # Without this modification for resnet-18 on CIFAR-10, the final accuracy drops about 5%.
    # TODO: check larger networks too.
    if "cifar" in cfg["dataset"]["name"]:
        model = modify_resnet_by_simclr_for_cifar(model)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    epochs = cfg["experiment"]["epochs"]
    init_lr = calculate_scaled_lr(
        base_lr=cfg["optimizer"]["lr"],
        batch_size=train_batch_size,
        lr_schedule=cfg["lr_scheduler"]["name"],
    )
    # simsiam version.
    lr_list = calculate_lr_list(
        init_lr,
        num_lr_updates_per_epoch=1,
        warmup_epochs=cfg["lr_scheduler"]["warmup_epochs"],
        epochs=epochs,
    )

    # optimizer
    optimizer = torch.optim.SGD(
        params=model.parameters(),
        lr=init_lr,
        momentum=cfg["optimizer"]["momentum"],
        nesterov=False,
        weight_decay=cfg["optimizer"]["decay"],
    )
    # optimizer = LARC(optimizer=optimizer, trust_coefficient=0.001, clip=False)

    save_fname = Path(os.getcwd()) / cfg["experiment"]["output_model_name"]

    supervised_training(
        model,
        train_data_loader,
        optimizer,
        lr_list,
        validation_data_loader,
        local_rank,
        cfg,
    )

    test_eval(model, test_data_loader, local_rank, save_fname)


if __name__ == "__main__":
    """
    To run this code,
    `torchrun --nproc_per_node={The number of GPUs on a single machine} supervised.py`
    """
    main()
