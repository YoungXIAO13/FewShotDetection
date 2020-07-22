# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__sets = {}

from datasets.coco import coco
from datasets.pascal_voc import pascal_voc
from datasets.objectnet3d import objectnet3d


# Set up objectnet3d_<split>
for split in ['train', 'val', 'test', 'shots']:
    name = 'objectnet3d_{}'.format(split)
    data_path = '/home/xiao/Datasets/ObjectNet3D'
    csv_file = 'ObjectNet3D_new.txt'
    if split == 'shots':
        split = 'train'
        csv_file = 'Shots.txt'
    __sets[name] = (lambda split=split, data_path=data_path, csv_file=csv_file: objectnet3d(split, data_path, csv_file))


# # Set up voc_<year>_<split>
for year in ['2007', '2012']:
    for split in ['train', 'val', 'trainval', 'test', 'shots',
                  'train_first_split', 'train_second_split', 'train_third_split']:
        name = 'voc_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: pascal_voc(split, year))

for year in ['2014']:
    for split in ['train', 'val', 'minival', 'valminusminival', 'trainval', 'shots']:
        name = 'coco_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: coco(split, year))

for year in ['2017']:
    for split in ['train', 'val']:
        name = 'coco_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: coco(split, year))


def get_imdb(name):
    """Get an imdb (image database) by name."""
    if name not in __sets:
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()


def list_imdbs():
    """List all registered imdbs."""
    return list(__sets.keys())
