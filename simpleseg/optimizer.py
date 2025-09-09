"""
Pytorch Optimizer and Scheduler Related Task
"""
import math
import logging
import torch
from torch import optim
from typing import List


def build_optimizer(cfg, model):

    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        # for swin model
        if "relative_position_bias_table" in key:
            weight_decay = 0.
        if "pos_embed" in key:
            weight_decay = 0.
        # if "norm" in key:
        #     # print(key)
        #     weight_decay = 0.
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    if cfg.SOLVER.OPTIMIZER == "sgd":
        optimizer = optim.SGD(
            params,
            lr=cfg.SOLVER.BASE_LR,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            momentum=cfg.SOLVER.MOMENTUM,
            nesterov=False
        )
    elif cfg.SOLVER.OPTIMIZER == "adamw":
        optimizer = optim.AdamW(
            params,
            lr=cfg.SOLVER.BASE_LR,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            amsgrad=cfg.SOLVER.AMSGRAD
        )
    else:
        raise NotImplementedError()

    return optimizer


class PolyLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        max_iters,
        power,
        last_epoch=-1,
    ):
        self.power = power
        self.max_iters = max_iters
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        # print("1")
        factor = math.pow(
            (1.0 - self.last_epoch / self.max_iters), self.power)
        return [lr * factor for lr in self.base_lrs]


def _get_warmup_factor_at_iter(method, iteration, warmup_iters, warmup_factor):
    if iteration >= warmup_iters:
        return 1.0
    if method == "constant":
        return warmup_factor
    elif method == "linear":
        alpha = iteration / warmup_iters
        return warmup_factor * (1 - alpha) + alpha
    else:
        raise ValueError("Unknown warmup method: {}".format(method))


class WarmupPolyLR(torch.optim.lr_scheduler._LRScheduler):

    def __init__(
        self,
        optimizer,
        max_iters,
        warmup_factor=0.001,
        warmup_iters=1000,
        warmup_method="linear",
        power=0.9,
        last_epoch=-1,
        constant_ending=0.0,
    ):
        self.max_iters = max_iters
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        self.power = power
        self.constant_ending = constant_ending
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = _get_warmup_factor_at_iter(
            self.warmup_method, self.last_epoch, self.warmup_iters, self.warmup_factor
        )
        if self.constant_ending > 0 and warmup_factor == 1.0:
            if (
                math.pow((1.0 - self.last_epoch / self.max_iters), self.power)
                < self.constant_ending
            ):
                return [base_lr * self.constant_ending for base_lr in self.base_lrs]
        return [
            base_lr * warmup_factor *
            math.pow((1.0 - self.last_epoch / self.max_iters), self.power)
            for base_lr in self.base_lrs
        ]


def build_poly_lr_scheduler(cfg, opt, scheduler_iteraions):
    if cfg.SOLVER.SCHEDULER == "poly":
        return PolyLR(opt, scheduler_iteraions, cfg.SOLVER.POWER)
    elif cfg.SOLVER.SCHEDULER == "warmup_poly":
        return WarmupPolyLR(
            opt,
            scheduler_iteraions,
            cfg.SOLVER.WARMUP_FACTOR,
            cfg.SOLVER.WARMUP_ITERS,
            cfg.SOLVER.WARMUP_METHOD,
            power=cfg.SOLVER.POWER
        )
    else:
        raise NotImplementedError("not implemented")


if __name__ == '__main__':
    from torch.nn import Parameter
    from torch.optim import SGD
    model = [Parameter(torch.randn(2, 2, requires_grad=True))]
    optimizer = SGD(model, lr=1.)

    scheduler = PolyLR(optimizer, 0.9, 100)

    for epoch in range(10):
        print(epoch, scheduler.get_lr()[0])
        optimizer.step()
        scheduler.step()
