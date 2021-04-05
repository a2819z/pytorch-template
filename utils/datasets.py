import os

import torch

from utils.torch_utils import torch_distributed_zero_first


def create_dataloader(
    path, batch_size, opt, hyp=None, augment=None, rank=-1, world_size=1, workers=4
):
    with torch_distributed_zero_first(rank):
        data_module = __import__(f"dataset.{opt.data.type}")
        dataset = data_module(path, opt.data.args)  # TODO: yaml args, kwargs type check

    batch_size = min(batch_size, len(dataset))
    nw = min(
        [os.cpu_count() // world_size, batch_size if batch_size > 1 else 0, workers]
    )  # number of workers
    sampler = (
        torch.utils.data.distributed.DistributedSampler(dataset) if rank != -1 else None
    )
    loader = (
        torch.utils.data.DataLoader if True else InfiniteDataLoader
    )  # TODO: InfiniteDataLoader flags

    dataloader = loader(
        dataset, batch_size=batch_size, num_workers=nw, sampler=sampler, pin_memory=True
    )

    return dataloader, dataset


class InfiniteDataLoader(torch.utils.data.dataloader.DataLoader):
    """ Dataloader that reuses workers
    Uses same syntax as vanilla DataLoader
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "batch_sampler", _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)
