# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
from model.utils.config import cfg
import os.path as osp
import sys
import os
import numpy as np
import scipy.sparse
import scipy.io as sio
import pickle
import json
import uuid
# COCO API
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as COCOmask

import torch.utils.data as data
import cv2
import torch
import random
import collections
import time


class MetaDatasetCOCO(data.Dataset):
    def __init__(self, root, image_set, year, img_size, shots=1, phase=1, shuffle=False, inter=False):
        self.shuffle = shuffle
        self.img_size = img_size
        self.phase = phase
        subset = 'inter' if inter else 'shots'
        self.shot_path = os.path.join(root, 'annotations', 'instances_{}2014.json'.format(subset))
        self.shots = shots
        if phase == 2:
            self.shots = shots * 3

        # name, paths
        self._year = year
        self._image_set = image_set
        self._data_path = root

        # load COCO API, classes, class <-> id mappings
        self._COCO = COCO(self._get_ann_file())
        self.json_data = self._COCO.dataset.copy()
        cats = self._COCO.loadCats(self._COCO.getCatIds())

        self._classes = tuple(['__background__'] +
                              [c['name'] for c in cats if c['name'] not in cfg.VOC_CLASSES] +
                              [c['name'] for c in cats if c['name'] in cfg.VOC_CLASSES])

        self._class_to_coco_cat_id = dict(list(zip([c['name'] for c in cats], self._COCO.getCatIds())))
        self._image_index = self._load_image_set_index()

        # Some image sets are "views" (i.e. subsets) into others.
        # For example, minival2014 is a random 5000 image subset of val2014.
        # This mapping tells us where the view's images and proposals come from.
        self._view_map = {
            'minival2014': 'val2014',  # 5k val2014 subset
            'valminusminival2014': 'val2014',  # val2014 \setminus minival2014
            'valminuscapval2014': 'val2014',
            'capval2014': 'val2014',
            'captest2014': 'val2014',
            'shots2014': 'train2014',
            'inter2014': 'train2014'
        }
        coco_name = image_set + year  # e.g., "val2014"
        self._data_name = (self._view_map[coco_name] if coco_name in self._view_map else coco_name)

        if phase == 1:
            self.metaclass = tuple([c['name'] for c in cats if c['name'] not in cfg.VOC_CLASSES])
        else:
            self.metaclass = tuple([c['name'] for c in cats if c['name'] not in cfg.VOC_CLASSES] +
                                   [c['name'] for c in cats if c['name'] in cfg.VOC_CLASSES])
        class_to_idx = dict(zip(self.metaclass, range(len(self.metaclass))))  # class to index mapping

        self.prndata = []
        self.prncls = []

        prn_image_pth = os.path.join(root, 'annotations', 'prn_image_{}shots.pt'.format(shots))
        prn_mask_pth = os.path.join(root, 'annotations', 'prn_mask_{}shots.pt'.format(shots))

        if os.path.exists(prn_image_pth) and os.path.exists(prn_mask_pth) and self.phase == 1:
            prn_image = torch.load(prn_image_pth)
            prn_mask = torch.load(prn_mask_pth)
        else:
            prn_image, prn_mask = self.get_prndata()

            torch.save(prn_image, prn_image_pth)
            torch.save(prn_mask, prn_mask_pth)

        for i in range(shots):
            cls = []
            data = []
            for n, key in enumerate(list(prn_image.keys())):
                img = torch.from_numpy(np.array(prn_image[key][i % len(prn_image[key])]))
                img = img.unsqueeze(0)
                mask = torch.from_numpy(np.array(prn_mask[key][i % len(prn_mask[key])]))
                mask = mask.unsqueeze(0)
                mask = mask.unsqueeze(3)
                imgmask = torch.cat([img, mask], dim=3)
                cls.append(class_to_idx[key])
                data.append(imgmask.permute(0, 3, 1, 2).contiguous())
            self.prncls.append(cls)
            self.prndata.append(torch.cat(data, dim=0))

    def __getitem__(self, index):
        return self.prndata[index], self.prncls[index]

    def __len__(self):
        return len(self.prndata)

    def _get_ann_file(self):
        prefix = 'instances' if self._image_set.find('test') == -1 else 'image_info'
        return osp.join(self._data_path, 'annotations', prefix + '_' + self._image_set + self._year + '.json')

    def _load_image_set_index(self):
        """
        Load image ids.
        """
        image_ids = self._COCO.getImgIds()
        return image_ids

    def _get_widths(self):
        anns = self._COCO.loadImgs(self._image_index)
        widths = [ann['width'] for ann in anns]
        return widths

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_id_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self._image_index[i]

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """

        # Example image path for index=119993:
        #   images/train2014/COCO_train2014_000000119993.jpg
        if self._year == '2017':
            file_name = str(index).zfill(12) + '.jpg'
        elif self._year == '2014':
            file_name = ('COCO_' + self._data_name + '_' + str(index).zfill(12) + '.jpg')
        image_path = osp.join(self._data_path, 'images', self._data_name, file_name)
        assert osp.exists(image_path), 'Path does not exist: {}'.format(image_path)
        return image_path

    def get_prndata(self):
        print('\ngenerating {} shot samples...'.format(self.shots))
        start = time.time()

        if self.shuffle:
            random.shuffle(self._image_index)
        prn_image = collections.defaultdict(list)
        prn_mask = collections.defaultdict(list)
        classes = collections.defaultdict(int)
        valid_img_ids = []

        for cls in self.metaclass:
            classes[cls] = 0

        for img_id in self._image_index:
            im_ann = self._COCO.loadImgs(img_id)[0]
            width = im_ann['width']
            height = im_ann['height']

            annIds = self._COCO.getAnnIds(imgIds=img_id, iscrowd=None)
            objs = self._COCO.loadAnns(annIds)

            # Sanitize bboxes -- some are invalid
            valid_objs = []
            for obj in objs:
                x1 = np.max((0, obj['bbox'][0]))
                y1 = np.max((0, obj['bbox'][1]))
                x2 = np.min((width - 1, x1 + np.max((0, obj['bbox'][2] - 1))))
                y2 = np.min((height - 1, y1 + np.max((0, obj['bbox'][3] - 1))))
                if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
                    obj['clean_bbox'] = [x1, y1, x2, y2]
                    valid_objs.append(obj)
            objs = valid_objs

            # Lookup table to map from COCO category ids to our internal class indices
            coco_cat_id_to_class = dict([(self._class_to_coco_cat_id[cls], cls) for cls in self._classes[1:]])

            img = cv2.imread(self.image_path_from_index(img_id), cv2.IMREAD_COLOR)
            img = img.astype(np.float32, copy=False)
            img -= cfg.PIXEL_MEANS

            mask = np.zeros((self.img_size, self.img_size), dtype=np.float32)
            h, w, _ = img.shape
            y_ration = float(h) / self.img_size
            x_ration = float(w) / self.img_size
            img_resize = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            for obj in objs:
                if obj['iscrowd']:
                    continue

                cls = coco_cat_id_to_class[obj['category_id']]
                if cls not in self.metaclass:
                    continue

                if classes[cls] >= self.shots:
                    break

                classes[cls] += 1
                x1 = int(obj['clean_bbox'][0] / x_ration)
                y1 = int(obj['clean_bbox'][1] / y_ration)
                x2 = int(obj['clean_bbox'][2] / x_ration)
                y2 = int(obj['clean_bbox'][3] / y_ration)

                mask[y1:y2, x1:x2] = 1

                prn_image[cls].append(img_resize)
                prn_mask[cls].append(mask)

                if img_id not in valid_img_ids:
                    valid_img_ids.append(img_id)
                break

            if len(classes) > 0 and min(classes.values()) == self.shots:
                break

        end = time.time()
        print('few-shot samples generated in {} s\n'.format(end - start))

        # filter the original json file
        new_images = []
        new_annotations = []

        for image in self.json_data['images']:
            if image['id'] in valid_img_ids:
                new_images.append(image)
        for annotation in self.json_data['annotations']:
            if annotation['image_id'] in valid_img_ids:
                new_annotations.append(annotation)

        self.json_data['images'] = new_images
        self.json_data['annotations'] = new_annotations

        def convert(o):
            if isinstance(o, np.int64): return int(o)
            raise TypeError

        with open(self.shot_path, 'w') as f:
            json.dump(self.json_data, f, default=convert)

        return prn_image, prn_mask

