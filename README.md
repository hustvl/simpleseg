# SimpleSeg

SimpleSeg is a lightweight, research-friendly repository for semantic segmentation from [HUSTVL](https://github.com/hustvl). It aims to provide clean baselines and utilities for training, evaluating, and deploying segmentation models in Python.

## Features

- Minimal, readable code structure for rapid experimentation
- Reproducible training and evaluation routines
- Configurable model, dataset, and augmentation pipelines
- Single- and multi-GPU training with PyTorch (if configured)
- Checkpointing, logging, and metric tracking (mIoU, pixel accuracy, etc.)


## Usage

1. Download pre-trained weights: [Google Drive]()

2. Training

```bash
python -m torch.distributed.launch --nproc_per_node ${GPUS} train.py --cfg ${CONFIG}
```

3. Evaluation

```bash
python test.py --cfg ${CONFIG} --ckpt {CKPT} TEST.BATCH_SIZE 1
```



## Citation
If you find this repository useful in your research, please cite it:
```bibtex
@article{gain2025,
  author       = {Tianheng Cheng and
                  Xinggang Wang and
                  Junchao Liao and
                  Wenyu Liu},
  title        = {Cross-layer attentive feature upsampling for low-latency semantic
                  segmentation},
  journal      = {Mach. Vis. Appl.},
  volume       = {36},
  number       = {1},
  pages        = {18},
  year         = {2025}
}
```


## Acknowledgements

- This project is maintained by [HUSTVL](https://github.com/hustvl).
- Thanks to the open-source community and prior work in semantic segmentation.
