from abc import abstractmethod
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP

import torch.nn.functional as F

from utils.torch_utils import ModelEMA, is_main_worker


class BaseTrainer:
    def __init__(self, model, optimizer, writer, logger, evaluator, test_loader, cfg):
        self.model = model
        self.ema = ModelEMA(self.model) if is_main_worker(cfg.gpu) else None
        self.optimizer = optimizer

        self.scaler = amp.GradScaler(enabled=cfg.amp)

        self.cfg = cfg

        self.model = self.set_ddp(self.model)

        self.tb_writer = writer
        self.logger = logger
        self.evaluator = evaluator
        self.test_loader = test_loader

        self.losses = {}

    def set_ddp(self, model):
        if self.cfg.use_ddp:
            return DDP(model, device_ids=[self.cfg.gpu], output_device=self.cfg.gpu)

        return model

    def clear_losses(self):
        self.losses = {}

    @abstractmethod
    def train(self):
        raise NotImplementedError

    def sync_ema(self):
        if self.ema is not None:
            self.ema.update(self.model)

    def save_checkpoint(self):
        pass
