import argparse
from pathlib import Path

import torch

from sconf import Config

from utils.datasets import create_dataloader


def eval():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        type=str,
        default="./config/defaults.yaml",
        help="path to config file(.yaml)",
    )
    parser.add_argument("--weight", type=str, help="path to weight to evaluate(.pth)")
    parser.add_argument("--result-dir", type=str, help="path to save the result file")
    args, left_argv = parser.parse_known_args()

    cfg = Config(*args.cfg, default="./cfgs/defaults.yaml")
    cfg.argv_update(left_argv)
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    # Init model
    ckpt = torch.load(args.weight)
    # load weight

    test_dataloader, test_dataset = create_dataloader()

    for batch in test_dataloader:
        pass


if __name__ == "__main__":
    eval()
