import argparse
import os
from pathlib import Path
import time

import yaml

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

from utils.torch_utils import ModelEMA, select_device, intersect_dicts, freeze_layer
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
    model = init_model(opt, weights, device)
    freeze_layer(model, opt.hyp.freeze)

    train_path = data_dict["train"]
    val_path = data_dict["val"]

    optimizer = init_optimzer(opt, model)

    ema = ModelEMA(model) if rank in [-1, 0] else None

    start_epoch = resume(weights, optimizer, ema, device)

    # DP mode
    if cuda and rank == -1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    dataloader, dataset = create_dataloader(
        train_path,
        batch_size,
        opt,
        hyp=hyp,
        rank=rank,
        world_size=opt.world_size,
        workers=opt.workers,
    )
    nb = len(dataloader)
    # Process 0
    if rank in [-1, 0]:
        testloader, testset = create_dataloader(
            val_path,
            batch_size * 2,
            opt,
            hyp=hyp,
            rank=-1,
            world_size=opt.world_size,
            workers=opt.workers,
        )

    if cuda and rank != -1:
        model = DDP(model, device_ids=[opt.local_rank], output_device=opt.local_rank)

    scaler = amp.GradScaler(enabled=opt.hyp.amp)
    # TODO: LossComputer

    # Start training
    t0 = time.time()
    for epoch in range(start_epoch, epochs):
        model.train()
        if rank != -1:
            dataloader.sampler.set_epoch(epoch)
        pbar = enumerate(dataloader)
        if rank in [-1, 0]:
            pbar = tqdm(pbar, total=nb)

        optimizer.zero_grad()
        for i, (img, labels) in pbar:
            # TODO: Training process
            pass


def init_model(opt, weights, device):
    module = __import__(f"models.{opt.cfg.module}")
    # TODO: Model configure opt
    model = getattr(module, f"{opt.cfg.architecture}").to(device)

    pretrained = weights.endswith(".pth")
    if pretrained:
        checkpoint = torch.load(weights, map_location=device)
        state_dict = checkpoint["model"]
        state_dict = intersect_dicts(state_dict, model.state_dict())
        model.load_state_dict(state_dict, strict=False)

    return model


def init_optimzer(opt, model: nn.Module) -> optim.Optimizer:
    eps = 1e-4 if opt.hyp.amp else 1e-8
    args = opt.hyp.optim_args
    kwargs = {"lr": opt.hyp.lr, "params": model.parameters(), "eps": eps}

    return getattr(optim, opt.hyp.optimizer)(*args, **kwargs)


def resume(weights, optimizer: optim.Optimizer, ema: ModelEMA, device, results):
    start_epoch = 0
    pretrained = weights.endswith(".pth")
    if pretrained:
        checkpoint = torch.load(weights, map_location=device)
        if (ckpt_optim := checkpoint.get("optimizer", None)) is not None:
            optimizer.load_state_dict(ckpt_optim)

        if ema and (ckpt_ema := checkpoint.get("ema", False)):
            ema.ema.load_state_dict(ckpt_ema.state_dict())
            ema.updates = ckpt_ema["updates"]

        start_epoch = checkpoint["epoch"] + 1

    return start_epoch


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
    parser.add_argument("--workers", type=int, default=4)
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

