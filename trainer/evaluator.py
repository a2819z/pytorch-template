import torch


def torch_eval(val_fn):
    @torch.no_grad()
    def wrapped(self, model, *args, **kwargs):
        model.eval()
        ret = val_fn(self, model, *args, **kwargs)
        model.train()

        return ret

    return wrapped


class Evaluator:
    def __init__(self):
        pass
