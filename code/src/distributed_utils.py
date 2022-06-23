import torch
import torch.distributed as dist
from omegaconf import OmegaConf


def init_ddp(conf: OmegaConf, local_rank: int) -> None:

    world_size = conf["distributed"]["world_size"]
    dist_url = conf["distributed"]["dist_url"]

    # prepare DDP group.
    dist.init_process_group(
        backend="nccl", init_method=dist_url, world_size=world_size, rank=local_rank
    )
    torch.cuda.set_device(local_rank)


def cleanup():
    dist.destroy_process_group()
