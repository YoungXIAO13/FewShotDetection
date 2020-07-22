from __future__ import print_function
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import datasets
import datasets.objectnet3d
import os, sys
from datasets.imdb import imdb
import xml.dom.minidom as minidom
import numpy as np
import scipy.sparse
import scipy.io as sio
import subprocess
import pickle
from model.utils.config import cfg
import pandas as pd

import torch.utils.data as data
import cv2
import torch
import random
import collections
import time


class MetaDataset3D(data.Dataset):
    def __init__(self, data_path, csv_file, img_size, image_set='train', shots=1, phase=1, shuffle=False):

        self.shuffle = shuffle
        self.img_size = img_size
        self.phase = phase
        self.shot_path = os.path.join(data_path, 'Shots.txt')
        self.shots = shots
        if phase == 2:
            self.shots = shots * 3

        self._data_path = data_path
        assert os.path.exists(self._data_path), 'Path does not exist: {}'.format(self._data_path)

        df = pd.read_csv(os.path.join(data_path, csv_file))

        if image_set != 'train':
            df = df[df.occluded == 0]
            df = df[df.truncated == 0]
        self.df = df[df.set == image_set]

        self._classes = tuple(['__background__'] +
                              [c for c in np.unique(df.cat).tolist() if c not in cfg.NOVEL_3D_CLASSES] +
                              [c for c in np.unique(df.cat).tolist() if c in cfg.NOVEL_3D_CLASSES])
        self.num_classes = len(self._classes)

        if phase == 1:
            self.metaclass = [c for c in np.unique(df.cat).tolist() if c not in cfg.NOVEL_3D_CLASSES]
        else:
            self.metaclass = self._classes[1:]
        class_to_idx = dict(zip(self.metaclass, range(len(self.metaclass))))  # class to index mapping

        self._image_index = np.unique(self.df.im_path).tolist()

        self.prndata = []
        self.prncls = []

        prn_image_pth = os.path.join(data_path, 'prn_image_{}shots.pt'.format(shots))
        prn_mask_pth = os.path.join(data_path, 'prn_mask_{}shots.pt'.format(shots))

        if os.path.exists(prn_image_pth) and os.path.exists(prn_mask_pth):
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

    def image_id_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return i

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path, index)
        assert os.path.exists(image_path), 'path does not exist: {}'.format(image_path)
        return image_path

    def get_prndata(self):
        print('\ngenerating {} shot samples...'.format(self.shots))
        start = time.time()

        if self.shuffle:
            random.shuffle(self._image_index)
        prn_image = collections.defaultdict(list)
        prn_mask = collections.defaultdict(list)
        classes = collections.defaultdict(int)
        valid_annot = []

        for cls in self.metaclass:
            classes[cls] = 0

        for index in self._image_index:

            objs = self.df[self.df.im_path == index]
            num_objs = len(objs)

            img = cv2.imread(self.image_path_from_index(index), cv2.IMREAD_COLOR)
            # tile channels for 1-channel images
            if len(img.shape) == 2:
                img = img[:, :, np.newaxis]
                img = np.concatenate((img, img, img), axis=2)

            # drop the last channel for 4-channel images
            if img.shape[-1] == 4:
                img = img[:, :, :-1]

            img = img.astype(np.float32, copy=False)
            img -= cfg.PIXEL_MEANS

            mask = np.zeros((self.img_size, self.img_size), dtype=np.float32)
            h, w = img.shape[0], img.shape[1]
            y_ration = float(h) / self.img_size
            x_ration = float(w) / self.img_size
            img_resize = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

            for ix in range(num_objs):
                x1 = max(float(objs.iloc[ix]['left']), 0)
                y1 = max(float(objs.iloc[ix]['upper']), 0)
                x2 = min(float(objs.iloc[ix]['right']), objs.iloc[ix]['width'] - 1)
                y2 = min(float(objs.iloc[ix]['lower']), objs.iloc[ix]['height'] - 1)
                cls = objs.iloc[ix]['cat']

                if objs.iloc[ix]['difficult'] or objs.iloc[ix]['occluded'] or objs.iloc[ix]['truncated']:
                    continue

                if cls not in self.metaclass:
                    continue

                if classes[cls] >= self.shots:
                    break

                classes[cls] += 1

                x1 = int(x1 / x_ration)
                y1 = int(y1 / y_ration)
                x2 = int(x2 / x_ration)
                y2 = int(y2 / y_ration)
                mask[y1:y2, x1:x2] = 1

                prn_image[cls].append(img_resize)
                prn_mask[cls].append(mask)

                valid_annot.append(objs.iloc[[ix]])
                break

            if len(classes) > 0 and min(classes.values()) == self.shots:
                break

        end = time.time()
        print('few-shot samples generated in {} s\n'.format(end - start))

        # save filtered csv file
        valid_annot = pd.concat(valid_annot)
        valid_annot.to_csv(self.shot_path, index=False, header=True)

        return prn_image, prn_mask
