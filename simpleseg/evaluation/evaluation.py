import torch
import numpy as np
import torch.distributed as dist


class ConfusionMatrix:

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = None
        self._reset()

    def _reset(self):
        self.hist = torch.zeros(
            (self.num_classes, self.num_classes),
            dtype=torch.int64)

    def synchronize_from_all(self):
        hist_tensor = self.hist.cuda()
        dist.all_reduce(
            hist_tensor, op=torch.distributed.ReduceOp.SUM)
        self.hist = hist_tensor.cpu()

    def update(self, pred, target):
        mask = (target >= 0) & (target < self.num_classes)
        inds = self.num_classes * target[mask].to(torch.int64) + pred[mask]
        self.hist += torch.bincount(
            inds, minlength=self.num_classes**2).reshape(self.num_classes, self.num_classes)

    def summarize(self):
        hist = self.hist.float()
        acc = torch.diag(hist).sum() / hist.sum()
        acc_cls = torch.diag(hist) / (hist.sum(dim=1) + 1e-6)
        iu = torch.diag(hist) / (hist.sum(dim=1) +
                                 hist.sum(axis=0) - torch.diag(hist) + 1e-6)
        # print(iu, iu.shape)
        return {
            "pixel_accuracy": acc.item(),
            "class_accuracy": acc_cls.cpu().tolist(),
            "iu": iu.cpu().tolist(),
            "mIoU": iu.mean().item(),
            "mAcc": acc_cls.mean().item()}
