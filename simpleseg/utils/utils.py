import logging
import os
import errno
from collections import OrderedDict
import torch
import torch.distributed as dist
from prettytable import PrettyTable


def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def reduce_dict(input_dict, average=True):

    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.reduce(values, dst=0)
        if dist.get_rank() == 0 and average:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


def is_main_process():
    return get_rank() == 0


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def strip_prefix_if_present(state_dict, prefix):
    keys = sorted(state_dict.keys())
    if not all(key.startswith(prefix) for key in keys):
        return state_dict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        stripped_state_dict[key.replace(prefix, "")] = value
    return stripped_state_dict


def save_checkpoint(name, model, iteration, optimizer=None, scheduler=None, extra=None):

    data = dict()
    data['model'] = model.state_dict()
    data['iteration'] = iteration
    if optimizer is not None:
        data['optimizer'] = optimizer.state_dict()
    if scheduler is not None:
        data['scheduler'] = scheduler.state_dict()
    if extra is not None:
        data['extra'] = extra

    torch.save(data, name)


def load_checkpoint(name):

    data = torch.load(name, map_location='cpu')
    data['model'] = strip_prefix_if_present(data['model'], prefix="module.")
    return data


def output_evaluation_results(results, dataset_name, logger):

    if dataset_name == "cityscapes":
        from simpleseg.datasets.cityscapes import Cityscapes
        classes = Cityscapes.CLASSES
    elif dataset_name == "camvid":
        from simpleseg.datasets.camvid import CamVid
        classes = CamVid.CLASSES
    elif dataset_name == "ade20k":
        from simpleseg.datasets.ade20k import ADE20K
        classes = ADE20K.CLASSES
    else:
        raise NotImplementedError()

    # class table
    # from prettytable import PrettyTable
    # logger.info("Evaluation Results:")
    t = PrettyTable(['class', 'mIoU', 'Acc'])
    for cls, iu, acc in zip(classes, results['iu'], results['class_accuracy']):
        t.add_row([cls, '{:.2f}'.format(iu * 100), '{:.2f}'.format(acc * 100)])
    logger.info("Evaluation Results:\n{}".format(t))
    # logger.info("\n{}".format(t))
    t = PrettyTable(['mIoU', 'mAcc', 'aAcc'])
    t.add_row(
        ['{:.2f}'.format(results['mIoU']*100),
         '{:.2f}'.format(results['mAcc']*100),
         '{:.2f}'.format(results['pixel_accuracy']*100)])
    logger.info("Overall Results:\n{}".format(t))
