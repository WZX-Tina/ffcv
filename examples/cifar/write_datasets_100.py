from argparse import ArgumentParser
from typing import List
import time
import numpy as np
from tqdm import tqdm

import torch as ch
import torchvision

from fastargs import get_current_config
from fastargs.decorators import param
from fastargs import Param, Section
from fastargs.validation import And, OneOf

from ffcv.writer import DatasetWriter
from ffcv.fields import IntField, RGBImageField

Section('data', 'arguments to give the writer').params(
    train_dataset=Param(str, 'Where to write the new dataset', required=True),
    val_dataset=Param(str, 'Where to write the new dataset', required=True),
)

import torchvision.transforms as tt
@param('data.train_dataset')
@param('data.val_dataset')
def main(train_dataset, val_dataset):
    mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
    std = [0.26733428587941854, 0.25643846292120615, 0.2761504713263903]

    transform_train = tt.Compose([tt.RandomCrop(32, padding=4,padding_mode='reflect')])
    datasets = {
        'train': torchvision.datasets.CIFAR100('/tmp', train=True, download=True,transform=transform_train),
        'test': torchvision.datasets.CIFAR100('/tmp', train=False, download=True)
        }

    for (name, ds) in datasets.items():
        path = train_dataset if name == 'train' else val_dataset
        writer = DatasetWriter(path, {
            'image': RGBImageField(),
            'label': IntField()
        })
        writer.from_indexed_dataset(ds)


if __name__ == "__main__":
    config = get_current_config()
    parser = ArgumentParser(description='Fast CIFAR-100 training')
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    config.summary()

    main()
    print('write dataset completed')
