import os
import sys
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Cityscapes Evaluation')
    parser.add_argument('pred', type=str)
    parser.add_argument('-d', '--dataset', type=str,
                        default='./data/cityscapes')

    args = parser.parse_args()

    os.environ['CITYSCAPES_RESULTS'] = args.pred
    os.environ['CITYSCAPES_DATASET'] = args.dataset

    os.system('csEvalPixelLevelSemanticLabeling')
