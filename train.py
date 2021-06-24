import argparse
import sys
from pathlib import Path

from sconf import Config, dump_args

import torch
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

from models.model import Generator
from models.modules.modules import weights_init
from trainer import Trainer, Evaluator

from utils.torch_utils import ModelEMA, is_main_worker, load_checkpoint
from utils.general import init_seed
from utils.logger import Logger
from utils.datasets import create_dataloader


def parse_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg", type=str, default="./config/defaults.yaml", help="model .yaml path"
    )
    parser.add_argment("--weights", type=str, defulat="", help="initial weight path")
    parser.add_argument(
        "--resume", type=str, default="", help="resume most recent training"
    )
    parser.add_argument(
        "--device", default="", help="cuda devices, i.e. 0 or 1,2,3,4 or cpu"
    )

    args, left_argv = parser.parse_known_args()
    cfg = Config(args.cfg, default="./config/defaults.yaml")
    cfg.argv_update(left_argv)

    if cfg.use_ddp:
        cfg.n_workers = 0

    cfg.work_dir = Path(cfg.work_dir)
    (cfg.work_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    return args, cfg


def train_ddp(local_rank, args, cfg, world_size):
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:" + str(cfg.port),
        world_size=world_size,
        rank=local_rank,
    )
    cfg.batch_size = cfg.batch_size // world_size
    train(args, cfg, local_rank=local_rank)
    dist.destroy_process_group()


def train(args, cfg, local_rank=-1):
    cfg.gpu = local_rank
    torch.cuda.set_device(local_rank)

    # logger setting
    logger_path = cfg.work_dir / "log.log"
    logger = Logger.get(file_path=logger_path, level="info", colorize=True)

    tb_writer = SummaryWriter(logger_path)

    args_str = dump_args(args)
    if is_main_worker(cfg.gpu):
        logger.info("Run Argv:\n> {}".format(" ".join(sys.argv)))
        logger.info("Args:\n{}".format(args_str))
        logger.info("Configs:\n{}".format(cfg.dumps()))

    logger.info("Get dataset...")

    train_loader, train_dataset = create_dataloader(
        cfg.dataset.train,
        cfg.batch_size,
        use_ddp=cfg.use_ddp,
        n_workers=cfg.n_workers,
        shuffle=True,
    )

    if is_main_worker(cfg.gpu):
        test_loader, test_dataset = create_dataloader(
            cfg.datset.val,
            cfg.batch_size,
            use_ddp=False,
            workers=cfg.n_workers,
            shuffle=False,
        )

    # logger.inf("Build model ...")
    generator = Generator(cfg)
    optimizer = optim.Adam(generator.parameters(), lr=cfg.lr, betas=cfg.adam_betas)

    if cfg.resume:
        start_epoch, loss = load_checkpoint(cfg.resume, generator, optimizer)
        logger.info(f"Resumed checkpoint from {cfg.resume} (Epoch {start_epoch})")

    evaluator = Evaluator()

    trainer = Trainer(
        generator, optimizer, tb_writer, logger, evaluator, test_loader, cfg
    )
    trainer.train(train_loader, start_epoch, cfg.epoch)

    generator.apply(weights_init(cfg.init))

    return generator


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
    args, cfg = parse_args_and_config()

    init_seed(7777)

    if cfg.use_ddp:
        ngpus_per_node = torch.cuda_device_count()
        world_size = ngpus_per_node
        mp.spawn(train_ddp, nprocs=ngpus_per_node, args=(args, cfg, world_size))
    else:
        train(args, cfg)
