import torch

from tqdm import tqdm

from utils.torch_utils import is_main_worker
from utils.general import dummy_context_mgr
from .base_trainer import BaseTrainer


class Trianer(BaseTrainer):
    def __init__(self, model, optimizer, writer, logger, evaluator, test_loader, cfg):
        super().__init__(model, optimizer, writer, logger, evaluator, test_loader, cfg)

    def train(self, loader, start_epoch, epochs):
        iter_step = 0

        pbar = enumerate(loader)
        if is_main_worker(self.cfg.gpu):
            pbar = tqdm(pbar, total=len(loader))

        for epoch in range(start_epoch, epochs):
            self.model.train()
            if self.cfg.use_ddp:
                loader.set_epoch(epoch)

            self.optimizer.zero_grad()
            for i, (img, labels) in pbar:
                with torch.cuda.amp.autocast() if self.cfg.amp else dummy_context_mgr() as mpc:
                    """
                    1. Model Forwarding
                    2. Loss Calculation
                    3. Optimize
                    4. Validation
                    """
                    pass

                if is_main_worker(self.cfg.gpu):
                    if iter_step % self.cfg.print_freq == 0:
                        self.log()
                        pass

                    if iter_step % self.cfg.val_freq == 0:
                        pass
                        # self.evaluator.evaluate(self.model, self.test_loader, iter_step)

            iter_step += 1

    def log(self):
        pass
