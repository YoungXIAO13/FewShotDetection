from __future__ import print_function
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import datasets
import datasets.pascal3d
import os, sys
from datasets.imdb import imdb
import xml.dom.minidom as minidom
import numpy as np
import scipy.sparse
import scipy.io as sio
import subprocess
import pickle
import pandas as pd
from .pascal3d_eval import pascal3d_eval

try:
    xrange  # Python 2
except NameError:
    xrange = range  # Python 3


class pascal3d(imdb):
    def __init__(self, image_set, data_path, csv_file='Pascal3D_new.txt'):
        imdb.__init__(self, 'pascal3d_{}'.format(image_set))
        self._image_set = image_set
        self._data_path = data_path
        assert os.path.exists(self._data_path), 'Path does not exist: {}'.format(self._data_path)

        df = pd.read_csv(os.path.join(data_path, csv_file))

        # filter out occluded and truncated samples
        # if image_set != 'train':
        #     df = df[df.occluded == 0]
        #     df = df[df.truncated == 0]

        self.df = df[df.set == self._image_set]

        self._classes = ('__background__',  # always index 0
                         'aeroplane', 'bicycle', 'boat', 'bottle', 'bus', 'car',
                         'chair', 'diningtable', 'motorbike', 'sofa', 'train', 'tvmonitor')
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))

        # the format of image index in pascal3d is 'Images/category_source/img_index.ext'
        self._image_index = np.unique(self.df.im_path).tolist()

        # Default to roidb handler
        self._roidb_handler = self.gt_roidb
        self._comp_id = 'comp4'

        # Specific config options
        self.config = {'cleanup': False,
                       'use_salt': True,
                       'use_diff': False,
                       'matlab_eval': False}

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

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.
        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = pickle.load(fid)
            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return roidb

        gt_roidb = [self._load_pascal3d_annotation(index) for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))

        return gt_roidb

    def _load_pascal3d_annotation(self, index):
        """
        Load image and bounding boxes info from txt files of pascal3d.
        """

        objs = self.df[self.df.im_path == index]
        num_objs = len(objs)

        # original annotation for object detection
        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros(num_objs, dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)

        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)
        ishards = np.zeros((num_objs), dtype=np.int32)

        # added for viewpoint estimation
        # gt_cads = np.zeros(num_objs, dtype=np.int32)
        # viewpoints = np.zeros((num_objs, 3), dtype=np.uint16)

        # Load object annotation into a data frame.
        for ix in range(num_objs):
            x1 = max(float(objs.iloc[ix]['left']), 0)
            y1 = max(float(objs.iloc[ix]['upper']), 0)
            x2 = min(float(objs.iloc[ix]['right']), objs.iloc[ix]['width'] - 1)
            y2 = min(float(objs.iloc[ix]['lower']), objs.iloc[ix]['height'] - 1)
            cls = self._class_to_ind[objs.iloc[ix]['cat']]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0

            ishards[ix] = objs.iloc[ix]['difficult']
            if cls not in self._classes:
                continue
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

            # gt_cads[ix] = objs.iloc[ix]['cad_index']
            # viewpoints[ix, 0] = (360 - objs.iloc[ix]['azimuth']) % 360
            # viewpoints[ix, 1] = min(objs.iloc[ix]['elevation'] + 90, 89.9)
            # viewpoints[ix, 2] = (objs.iloc[ix]['inplane_rotation'] + 180) % 360
            # assert ((viewpoints[:, 1] >= 0).all() & (viewpoints[:, 1] < 180).all()), \
            #     print('min {}, max{}'.format(viewpoints[:, 1].min(), viewpoints[:, 1].max()))

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes': boxes,
                'gt_classes': gt_classes,
                'gt_ishard': ishards,
                'gt_overlaps': overlaps,
                'flipped': False,
                'seg_areas': seg_areas}

        # return {'boxes': boxes,
        #         'gt_classes': gt_classes,
        #         'gt_overlaps': overlaps,
        #         'gt_cads': gt_cads,
        #         'viewpoints': viewpoints,
        #         'flipped': False}

    def _get_results_file_template(self):
        # data_path/results/<comp_id>_det_test_aeroplane.txt
        filename = self._comp_id + '_det_' + self._image_set + '_{:s}.txt'
        filedir = os.path.join(self._data_path, 'results')
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        path = os.path.join(filedir, filename)
        return path

    def _write_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print('Writing {} Pascal3D results file'.format(cls))
            filename = self._get_results_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in xrange(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))

    def _do_python_eval(self, output_dir='output'):
        cachedir = os.path.join(self._data_path, 'annotations_cache')
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)

        aps = []
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            filename = self._get_results_file_template().format(cls)
            ap = pascal3d_eval(filename, self.df, self._image_set, cls, cachedir, ovthresh=0.5, view=24)
            print('AP for {} = {:.3f}'.format(cls, ap))

            aps.append(ap)

        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        return np.mean(aps)

    def evaluate_detections(self, all_boxes, output_dir, **kwargs):
        self._write_results_file(all_boxes)
        AP = self._do_python_eval(output_dir)
        if self.config['cleanup']:
            for cls in self._classes:
                if cls == '__background__':
                    continue
                filename = self._get_results_file_template().format(cls)
                os.remove(filename)
        return AP