import argparse
import os
from pathlib import Path

import yaml

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from models.model import Model

from utils.torch_utils import select_device
from utils.general import init_seed
from utils.datasets import create_dataloader


def train(hyp, opt: argparse.Namespace, device: torch.device, tb_writer: SummaryWriter):
    save_dir, epochs, batch_size, total_batch_size, weights, rank = (
        Path(opt.save_dir),
        opt.epochs,
        opt.batch_size,
        opt.total_batch_size,
        opt.weights,
        opt.global_rank,
    )

    wdir = save_dir / "weights"
    wdir.mkdir(parents=True, exist_ok=True)

    with open(save_dir / "hyp.yaml", "w") as f:
        yaml.dump(hyp, f, sort_keys=False)
    with open(save_dir / "opt.yaml", "w") as f:
        yaml.dump(vars(opt), f, sort_keys=False)

    cuda = device.type != "cpu"
    init_seed(777 + rank)
    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)

    # Model
    pretrained = weights.endswith(".pth")
    if pretrained:
        pass
    else:
        model = Model(opt.cfg).to(device)

    train_path = data_dict["train"]
    val_path = data_dict["val"]

    optimizer = init_optimzer(opt, model)

    # DP mode
    if cuda and rank == -1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    dataloader, dataset = create_dataloader()
    if rank in [-1, 0]:
        testloader, testset = create_dataloader()

    if cuda and rank != -1:
        model = DDP(model, device_ids=[opt.local_rank], output_device=opt.local_rank)


def init_optimzer(opt, model):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argment("--weights", type=str, defulat="", help="initial weight path")
    parser.add_argument("--cfg", type=str, default="", help="model .yaml path")
    parser.add_argument("--data", type=str, default="", help="data .yaml path")
    parser.add_argument("--hyp", type=str, default="", help="hyperparameters path")
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--resume", type=str, default="", help="resume most recent training"
    )
    parser.add_argument(
        "--device", default="", help="cuda devices, i.e. 0 or 1,2,3,4 or cpu"
    )

    parser.add_argument("--local_rank", default=-1, help="never change this value")
    opt = parser.parse_args()

    # Set DDP variable
    opt.world_size = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    opt.global_rank = int(os.environ["RANK"]) if "RANK" in os.environ else -1

    if opt.resume:
        pass
    else:
        opt.save_dir = ""

    # DDP mode
    opt.total_batch_size = opt.batch_size
    device = select_device(opt.device, batch_size=opt.batch_size)
    if opt.local_rank != -1:
        torch.cuda.set_device(opt.local_rank)
        device = torch.device("cuda", opt.local_rank)
        dist.init_process_groupi(backend="nccl", init_method="env://")
        assert (
            opt.batch_size % opt.world_size == 0
        ), "--batch-size must be multiple of CUDA device count"
        opt.batch_size = opt.total_batch_size // opt.world_size

    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)

    if opt.global_rank in [-1, 0]:
        tb_writer = SummaryWriter(opt.save_dir)
    train(hyp, opt, device, tb_writer)

