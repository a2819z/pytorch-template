"""
These codes have been modified based on the code of yolo v5
Original code is https://github.com/ultralytics/yolov5
"""
import os
import math
from copy import deepcopy
from contextlib import contextmanager

import torch
import torch.nn as nn


@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """
    Decorator to make all processes in distributed training wait for each local_master to do something.
    """
    if local_rank not in [-1, 0]:
        torch.distributed.barrier()
    yield
    if local_rank == 0:
        torch.distributed.barrier()


def select_device(device: str, batch_size: int) -> torch.device:
    cpu = device.lower() == "cpu"
    if cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    elif device:
        os.environ["CUDA_VISIBLE_DEVICES"] = device
        assert (
            torch.cuda.is_available()
        ), f"CUDA unabailable, invalid device {device} requested"

    cuda = not cpu and torch.cuda.is_available()
    if cuda:
        n = torch.cuda.device_count()
        if n > 1 and batch_size:
            assert (
                batch_size % n == 0
            ), f"batch-size {batch_size} not multiple of GPU count {n}"
        s = ""
        for idx, d in enumerate(device.split(",")) if device else range(n):
            p = torch.cuda.get_device_properties(idx)
            s += f"CUDA:{d} ({p.name}, {p.total_memory / 1024 ** 2}MB\n)"

    print(s)  # TODO: print -> logger
    return torch.device("cuda:0" if cuda else "cpu")


def intersect_dicts(da: dict, db: dict, exclude=()) -> dict:
    # Dictionary intersection of matching keys and shapes, omitting 'exclude' keys, using da values
    return {
        k: v
        for k, v in da.items()
        if k in db and not any(x in k for x in exclude) and v.shape == db[k].shape
    }


def freeze_layer(model: nn.Module, freeze: list):
    for k, v in model.named_parameters():
        v.required_grad = True
        if any(x in k for x in freeze):
            print(f"freezing {k}")
            v.requires_grad = False


def is_parallel(model: nn.Module) -> bool:
    return type(model) in (
        nn.parallel.DataParallel,
        nn.parallel.DistributedDataParallel,
    )


def copy_attr(a, b, include=(), exclude=()):
    # Copy attributes from b to a, options to only include [...] and to exclude [...]
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith("_") or k in exclude:
            continue
        else:
            setattr(a, k, v)


class ModelEMA:
    def __init__(self, model: nn.Module, decay=0.9999, updates=0):
        self.ema = deepcopy(model.module if is_parallel(model) else model).eval()
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model: nn.Module):
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = (
                model.module.state_dict() if is_parallel(model) else model.state_dict()
            )
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1.0 - d) * msd[k].detach()

    def update_attr(self, model, include=(), exclude=("process_group', 'reducer")):
        copy_attr(self.ema, model, include, exclude)
