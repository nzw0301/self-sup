import json
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
from self_sup.eval_utils import learnable_eval
from self_sup.logger import get_logger
from self_sup.models.classifier import LinearClassifier
from self_sup.models.contrastive import get_contrastive_model
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
        num_last_units = model.g.projection_head.linear2.out_features
    else:
        num_last_units = model.g.projection_head.linear1.in_features
        model.g = torch.nn.Identity()

    if local_rank == 0:
        logger.info(
            "#train: {}, #val: {}".format(
                len(training_dataset), len(validation_dataset)
            )
        )
        logger.info("Evaluation by using {}".format(weight_name))

    # initialise linear classifier
    # NOTE: the weights are not normalize
    classifier = LinearClassifier(num_last_units, num_classes, normalize=False).to(
        local_rank
    )
    classifier = torch.nn.parallel.DistributedDataParallel(
        classifier, device_ids=[local_rank]
    )

    # execute linear evaluation protocol
    (
        train_accuracies,
        train_top_k_accuracies,
        train_losses,
        val_accuracies,
        val_top_k_accuracies,
        val_losses,
    ) = learnable_eval(
        cfg, classifier, model, training_data_loader, validation_data_loader, top_k
    )

    if rank == 0:
        classification_results = {}
        classification_results[weight_name] = {
            "train_accuracies": train_accuracies,
            "val_accuracies": val_accuracies,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_top_{}_accuracies".format(top_k): train_top_k_accuracies,
            "val_top_{}_accuracies".format(top_k): val_top_k_accuracies,
            "lowest_val_loss": min(val_losses),
            "highest_val_acc": max(val_accuracies),
            "highest_val_top_k_acc": max(val_top_k_accuracies),
        }

        logger.info(
            "train acc: {}, val acc: {}".format(
                max(train_accuracies), max(val_accuracies)
            )
        )

        fname = cfg["experiment"]["classification_results_json_fname"]

        with open(fname, "w") as f:
            json.dump(classification_results, f)


if __name__ == "__main__":
    main()
