import torch


def create_dataloader(
    cfg, batch_size, augment=None, use_ddp=False, n_workers=4, shuffle=True
):
    import dataset

    data_module = getattr(dataset, cfg.type)
    dataset = data_module(cfg.path, **cfg["args"])

    if use_ddp:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        shuffle = False
    else:
        sampler = None

    loader = (
        torch.utils.data.DataLoader if True else InfiniteDataLoader
    )  # TODO: InfiniteDataLoader flags

    dataloader = loader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=n_workers,
        sampler=sampler,
        pin_memory=True,
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
