import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import loss
# from torch.nn.modules.loss import CrossEntropyLoss
# from config import cfg


class OHEMCrossEntropy2dOld(nn.Module):
    """
        Ohem Cross Entropy Tensor Version
    """

    def __init__(self, ignore_index=255, thresh=0.7, min_kept=10000, use_weight=False):
        super(OHEMCrossEntropy2dOld, self).__init__()
        self.ignore_index = ignore_index
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        if use_weight:
            weight = torch.FloatTensor(
                [0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489,
                 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955,
                 1.0865, 1.1529, 1.0507])
            self.criterion = nn.CrossEntropyLoss(
                reduction="mean", weight=weight, ignore_index=ignore_index)
        else:
            self.criterion = nn.CrossEntropyLoss(
                reduction="mean", ignore_index=ignore_index)

    def forward(self, pred, target):
        b, c, h, w = pred.size()
        target = target.view(-1)
        # print(target[target > 10])
        valid_mask = target.ne(self.ignore_index)
        target = target * valid_mask.long()
        num_valid = valid_mask.sum()
        # print(self.ignore_index)
        prob = F.softmax(pred, dim=1)
        prob = (prob.transpose(0, 1)).reshape(c, -1)
        # print(prob)
        if self.min_kept > num_valid:
            print('Labels: {}'.format(num_valid))
        elif num_valid > 0:
            prob = prob.masked_fill_(~valid_mask, 1)
            #  print(target)
            mask_prob = prob[
                target, torch.arange(len(target), dtype=torch.long)]
            threshold = self.thresh
            if self.min_kept > 0:
                _, index = mask_prob.sort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if mask_prob[threshold_index] > self.thresh:
                    threshold = mask_prob[threshold_index]
                kept_mask = mask_prob.le(threshold)
                target = target * kept_mask.long()
                valid_mask = valid_mask * kept_mask

        target = target.masked_fill_(~valid_mask, self.ignore_index)
        target = target.view(b, h, w)

        return self.criterion(pred, target)


class OHEMCrossEntropyLoss(nn.Module):
    def __init__(self, thresh=0.7, min_kpt=100000, ignore_index=255):
        super(OHEMCrossEntropyLoss, self).__init__()
        self.thresh = - \
            torch.log(torch.tensor(thresh, dtype=torch.float)).cuda()
        self.min_kpt = min_kpt
        self.criteria = nn.CrossEntropyLoss(
            ignore_index=ignore_index, reduction='none')

    def forward(self, logits, labels):
        # N, C, H, W = logits.size()
        min_kpt = self.min_kpt * logits.size(0)
        # print(min_kpt)
        # print(logits, labels)
        loss = self.criteria(logits, labels).view(-1)
        # print(loss)
        loss, _ = torch.sort(loss, descending=True)
        if loss[min_kpt] > self.thresh:
            loss = loss[loss > self.thresh]
        else:
            loss = loss[:min_kpt]
        return torch.mean(loss)


class OHEMAuxSegLoss(nn.Module):

    def __init__(self, aux_weight, ignore_index=255):
        super().__init__()
        self.aux_weight = aux_weight
        self.criteria = OHEMCrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, preds, targets):
        h, w = targets.size(1), targets.size(2)
        main_pred, aux_preds = preds
        loss_dict = {}
        # false?
        # main_pred = F.upsample(
        #     input=main_pred, size=(h, w), mode='bilinear', align_corners=False)
        loss_dict["main_loss"] = self.criteria(main_pred, targets)
        # aux_loss = 0
        for idx, aux_p in enumerate(aux_preds):
            aux_p = F.interpolate(
                input=aux_p, size=(h, w), mode='bilinear', align_corners=False)
            aux_loss = self.aux_weight * self.criteria(aux_p, targets)
            loss_dict["loss_aux_{}".format(idx)] = aux_loss

        return loss_dict


def build_segmentation_loss(cfg, ignore_index):
    # name = cfg.LOSS.NAME
    criterion = OHEMAuxSegLoss(cfg.LOSS.AUX_WEIGHT, ignore_index=ignore_index)
    return criterion
