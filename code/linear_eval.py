import os
from pathlib import Path

import hydra
import numpy as np
import torch
import torchvision
import wandb
import yaml
from omegaconf import OmegaConf

from self_sup.data.transforms import get_data_augmentation
from self_sup.data.utils import (
    create_data_loaders_from_datasets,
    get_train_val_test_datasets,
)
from self_sup.distributed_utils import init_ddp
from self_sup.logger import get_logger
from self_sup.lr_utils import calculate_lr_list, calculate_scaled_lr
from self_sup.models.classifier import ClassifierWithFeatureExtractor, LinearClassifier
from self_sup.models.contrastive import get_contrastive_model
from self_sup.train_utils import supervised_training, test_eval
from self_sup.wandb_utils import flatten_omegaconf


@hydra.main(config_path="conf", config_name="linear_eval")
def main(cfg: OmegaConf):
    logger = get_logger()

    local_rank = int(os.environ["LOCAL_RANK"]) if torch.cuda.is_available() else "cpu"
    init_ddp(cfg)

    # to reproduce results
    seed = cfg["experiment"]["seed"]
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    rnd = np.random.RandomState(seed)

    logger.info("Using cuda:{}".format(local_rank))

    # load pre-trained model
    weights_path = Path(cfg["experiment"]["target_weight_file"])
    weight_name = weights_path.name
    self_sup_config_path = weights_path.parent / ".hydra" / "config.yaml"
    with open(self_sup_config_path) as f:
        pre_train_conf = yaml.load(f, Loader=yaml.FullLoader)

    # initialise data loaders.
    train_dataset, validation_dataset, test_dataset = get_train_val_test_datasets(
        rnd=rnd,
        root=cfg["dataset"]["root"],
        validation_ratio=cfg["dataset"]["validation_ratio"],
        dataset_name=cfg["dataset"]["name"],
        normalize=cfg["dataset"]["normalize"],
    )
    train_dataset.transform = get_data_augmentation(cfg["augmentation"])
    validation_dataset.transform = torchvision.transforms.Compose[
        torchvision.transforms.ToTensor()
    ]
    test_dataset.transform = torchvision.transforms.Compose[
        torchvision.transforms.ToTensor()
    ]

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

    feature_extractor = get_contrastive_model(pre_train_conf, local_rank, weights_path)

    # get the dimensionality of the representation
    if cfg["experiment"]["use_projection_head"]:
        num_last_units = feature_extractor.g.projection_head.linear2.out_features
    else:
        num_last_units = feature_extractor.g.projection_head.linear1.in_features
        feature_extractor.g = torch.nn.Identity()

    if local_rank == 0:
        logger.info(
            f"#train: {len(train_dataset)}, #val: {len(validation_dataset)}, #test:{len(test_dataset)}"
        )
        logger.info("Evaluation by using {}".format(weight_name))
        wandb.init(
            dir=hydra.utils.get_original_cwd(),
            project="self-sup",
            entity="nzw0301",
            config=flatten_omegaconf(cfg),
            tags=(cfg["dataset"]["name"], "linear-eval"),
            group="seed-{}".format(seed),
        )

    assert "cifar" in cfg["dataset"]["name"]
    if "100" in cfg["dataset"]["name"]:
        num_classes = 100
    else:
        num_classes = 10

    # initialise linear classifier
    # NOTE: the weights are not normalize
    classifier = LinearClassifier(num_last_units, num_classes, normalize=False).to(
        local_rank
    )
    model = ClassifierWithFeatureExtractor(
        feature_extractor=feature_extractor,
        predictor=classifier,
        frozen_feature_extractor=True,
    )

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

    supervised_training(
        model,
        train_data_loader,
        optimizer,
        lr_list,
        validation_data_loader,
        local_rank,
        cfg,
    )

    save_fname = Path(os.getcwd()) / cfg["experiment"]["output_model_name"]
    test_eval(model, test_data_loader, local_rank, save_fname)


if __name__ == "__main__":
    main()
