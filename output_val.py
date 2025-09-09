import os
from simpleseg.datasets.cityscapes import make_cityscapes_dataset

if __name__ == "__main__":

    images, targets = make_cityscapes_dataset('./data/cityscapes/', 'val')
    with open('images.txt', 'w+') as f:
        for idx, image in enumerate(images):
            f.write('{}:{}\n'.format(idx, os.path.basename(image)))
