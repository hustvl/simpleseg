import logging
from fvcore.common.registry import Registry

SEGMENTOR = Registry("Segmentor")
SEGMENTOR.__doc__ = "segmentor registry"


def build_segmentor(cfg, num_classes, ignore_index):
    name = cfg.MODEL.NAME
    model = SEGMENTOR.get(name)(cfg, num_classes, ignore_index)
    model.cuda()
    return model
