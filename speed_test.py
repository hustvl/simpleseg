import argparse
import logging
import torch
import time

from simpleseg.config import get_cfg_defaults
from simpleseg.utils.logger import setup_logger
from simpleseg.utils.utils import mkdir
from simpleseg.models import build_segmentor


def flops_counter(model, input_size):
    pass


def measure_time(model, input_size, num_runs=500, warmup=20):
    logger = logging.getLogger("simpleseg.measure_time")
    model.eval()
    model_input = torch.randn([1, 3] + input_size, dtype=torch.float32).cuda()

    latency = 0
    latency_counter = 0
    logger.info("start measuring inference speed...")
    for i in range(num_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            model(model_input)
        torch.cuda.synchronize()
        end = time.perf_counter() - start
        if i >= warmup:
            latency += end
            latency_counter += 1

    time_per_image = 1.0 * latency / latency_counter
    fps = 1 / time_per_image

    logger.info("inference speed for image ({}x{}):".format(
        input_size[0], input_size[1]))
    logger.info("inference speed: {:.3f} ms per image {:.2f} FPS".format(
        time_per_image * 1000, fps))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Semantic Segmentation')
    parser.add_argument(
        '--cfg',
        type=str,
        required=True
    )
    # begin hard coding
    parser.add_argument(
        '--dataset',
        type=str,
        default='cityscapes'
    )
    input_sizes = {
        'cityscapes': [1024, 2048],
        'camvid': [960, 720]
    }
    # end hard coding

    args = parser.parse_args()
    cfg = get_cfg_defaults()

    cfg.merge_from_file(args.cfg)
    cfg.MODEL.NORM = "BN"
    cfg.freeze()
    mkdir(cfg.OUTPUT_DIR)
    logger = setup_logger('simpleseg', cfg.OUTPUT_DIR,
                          0, filename='benchmark.txt')

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enable = True

    model = build_segmentor(cfg, cfg.DATASET.NUM_CLASSES, 255)
    logger.info(model)

    measure_time(model, input_sizes[args.dataset])
