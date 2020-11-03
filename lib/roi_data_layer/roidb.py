"""Transform a roidb into a trainable roidb by adding a bunch of metadata."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datasets
import numpy as np
from model.utils.config import cfg
from datasets.factory import get_imdb
import PIL
import pdb
import collections


def prepare_roidb(imdb):
    """Enrich the imdb's roidb by adding some derived quantities that
      are useful for training. This function precomputes the maximum
      overlap, taken over ground-truth boxes, between each ROI and
      each ground-truth box. The class with maximum overlap is also
      recorded.
      """

    roidb = imdb.roidb
    if not (imdb.name.startswith('coco') or imdb.name.startswith('vg')):
        sizes = [PIL.Image.open(imdb.image_path_at(i)).size for i in range(imdb.num_images)]

    for i in range(len(imdb.image_index)):
        roidb[i]['img_id'] = imdb.image_id_at(i)
        roidb[i]['image'] = imdb.image_path_at(i)
        if not (imdb.name.startswith('coco') or imdb.name.startswith('vg')):
            roidb[i]['width'] = sizes[i][0]
            roidb[i]['height'] = sizes[i][1]
        # need gt_overlaps as a dense array for argmax
        gt_overlaps = roidb[i]['gt_overlaps'].toarray()
        # max overlap with gt over classes (columns)
        max_overlaps = gt_overlaps.max(axis=1)
        # gt class that had the max overlap
        max_classes = gt_overlaps.argmax(axis=1)
        roidb[i]['max_classes'] = max_classes
        roidb[i]['max_overlaps'] = max_overlaps
        # sanity checks
        # max overlap of 0 => class should be zero (background)
        zero_inds = np.where(max_overlaps == 0)[0]
        assert all(max_classes[zero_inds] == 0)
        # max overlap > 0 => class should not be zero (must be a fg class)
        nonzero_inds = np.where(max_overlaps > 0)[0]
        assert all(max_classes[nonzero_inds] != 0)


def update_keyvalue(rdb, idx):
    ## update the roidb keyvaule
    r = rdb.copy()
    keys = ['gt_classes', 'boxes']
    for k in keys:
        if isinstance(r[k], list):
            r[k] = [rdb[k][idx]]
        elif isinstance(r[k], np.ndarray):
            r[k] = np.array(rdb[k[idx]], dtype=r[k].dtype)
    return r


def clean_roidb(roidb):
    new_roidb = []
    for idx, rdb in enumerate(roidb):
        boxes = []
        gt_classes = []
        gt_overlaps = []
        max_classes = []
        max_overlaps = []

        if len(rdb['gt_classes']) > 0 and np.sum(rdb['gt_classes']) != 0:
            for i in range(len(rdb['gt_classes'])):
                boxes.append(rdb['boxes'][i])
                gt_classes.append(rdb['gt_classes'][i])
                gt_overlaps.append(rdb['gt_overlaps'][i])
                max_classes.append(rdb['max_classes'][i])
                max_overlaps.append(rdb['max_overlaps'][i])

        if len(boxes) > 0:
            new_roidb.append(
                {'boxes': np.array(boxes, dtype=np.uint16),
                 'gt_classes': np.array(gt_classes, dtype=np.int32),
                 'gt_overlaps': gt_overlaps,
                 'flipped': rdb['flipped'],
                 'img_id': rdb['img_id'], 'image': rdb['image'],
                 'width': rdb['width'], 'height': rdb['height'],
                 'max_classes': np.array(max_classes),
                 'need_crop': rdb['need_crop'],
                 'max_overlaps': np.array(max_overlaps, dtype=np.float32)})

    return new_roidb


def filter_class_roidb_flip(roidb, shot, imdb, base_num=15):

    class_count = collections.defaultdict(int)
    for cls in range(1, len(imdb.classes)):
        class_count[cls] = 0

    # filtering samples by class
    new_roidb = []
    length = len(roidb) // 2
    for idx, rdb in enumerate(roidb[:length]):
        boxes = []
        gt_classes = []
        gt_overlaps = []
        max_classes = []
        max_overlaps = []

        for i in range(len(rdb['gt_classes'])):
            cls_id = rdb['gt_classes'][i]

            # include novel object classes as background in the base training stage
            if shot == 0 and cls_id > base_num:
                continue

            # novel classes
            if class_count[cls_id] < shot and cls_id > base_num:
                boxes.append(rdb['boxes'][i])
                gt_classes.append(rdb['gt_classes'][i])
                gt_overlaps.append(rdb['gt_overlaps'][i])
                max_classes.append(rdb['max_classes'][i])
                max_overlaps.append(rdb['max_overlaps'][i])

                class_count[cls_id] += 1

            # base classes
            if cls_id <= base_num:
                boxes.append(rdb['boxes'][i])
                gt_classes.append(rdb['gt_classes'][i])
                gt_overlaps.append(rdb['gt_overlaps'][i])
                max_classes.append(rdb['max_classes'][i])
                max_overlaps.append(rdb['max_overlaps'][i])

                class_count[cls_id] += 1

        if len(boxes) > 0:
            new_roidb.append(
                {'boxes': np.array(boxes, dtype=np.uint16),
                 'gt_classes': np.array(gt_classes, dtype=np.int32),
                 'gt_overlaps': gt_overlaps,
                 'flipped': rdb['flipped'],
                 'img_id': rdb['img_id'], 'image': rdb['image'],
                 'width': rdb['width'], 'height': rdb['height'],
                 'max_classes': np.array(max_classes),
                 'need_crop': rdb['need_crop'],
                 'max_overlaps': np.array(max_overlaps, dtype=np.float32)})

    # appending the flipped samples
    for i in range(len(new_roidb)):
        width = new_roidb[i]['width']
        boxes = new_roidb[i]['boxes'].copy()
        oldx1 = boxes[:, 0].copy()
        oldx2 = boxes[:, 2].copy()
        boxes[:, 0] = width - oldx2 - 1
        boxes[:, 2] = width - oldx1 - 1
        assert (boxes[:, 2] >= boxes[:, 0]).all()
        entry = {'boxes': boxes,
                 'gt_classes': new_roidb[i]['gt_classes'],
                 'gt_overlaps': new_roidb[i]['gt_overlaps'],
                 'flipped': True,
                 'img_id': new_roidb[i]['img_id'], 'image': new_roidb[i]['image'],
                 'width': width, 'height': new_roidb[i]['height'],
                 'max_classes': new_roidb[i]['max_classes'],
                 'need_crop': new_roidb[i]['need_crop'],
                 'max_overlaps': new_roidb[i]['max_overlaps']}

        new_roidb.append(entry)

    return new_roidb




def rank_roidb_ratio(roidb):
    # rank roidb based on the ratio between width and height.
    ratio_large = 2  # largest ratio to preserve.
    ratio_small = 0.5  # smallest ratio to preserve.
    
    ratio_list = []
    for i in range(len(roidb)):
        width = roidb[i]['width']
        height = roidb[i]['height']
        ratio = width / float(height)

        if ratio > ratio_large:
            roidb[i]['need_crop'] = 1
            ratio = ratio_large
        elif ratio < ratio_small:
            roidb[i]['need_crop'] = 1
            ratio = ratio_small
        else:
            roidb[i]['need_crop'] = 0

        ratio_list.append(ratio)

    ratio_list = np.array(ratio_list)
    ratio_index = np.argsort(ratio_list)
    return ratio_list[ratio_index], ratio_index


def filter_roidb(roidb):
    # filter the image without bounding box.
    print('before filtering, there are %d images...' % (len(roidb)))
    i = 0
    while i < len(roidb):
        if len(roidb[i]['boxes']) == 0:
            del roidb[i]
            i -= 1
        i += 1

    print('after filtering, there are %d images...' % (len(roidb)))
    return roidb


def combined_roidb(imdb_names, training=True):
    """
    Combine multiple roidbs
    """

    def get_training_roidb(imdb):
        """Returns a roidb (Region of Interest database) for use in training."""
        if cfg.TRAIN.USE_FLIPPED:
            print('Appending horizontally-flipped training examples...')
            imdb.append_flipped_images()
            print('done')

        print('Preparing training data...')

        prepare_roidb(imdb)
        print('done')
        return imdb.roidb

    def get_roidb(imdb_name):
        imdb = get_imdb(imdb_name)
        print('Loaded dataset `{:s}` for training'.format(imdb.name))
        imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)  # gt
        print('Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD))
        roidb = get_training_roidb(imdb)
        return roidb

    roidbs = [get_roidb(s) for s in imdb_names.split('+')]
    roidb = roidbs[0]

    if len(roidbs) > 1:
        for r in roidbs[1:]:
            roidb.extend(r)
        tmp = get_imdb(imdb_names.split('+')[1])
        imdb = datasets.imdb.imdb(imdb_names, tmp.classes)
    else:
        imdb = get_imdb(imdb_names)

    if training:
        roidb = filter_roidb(roidb)

    ratio_list, ratio_index = rank_roidb_ratio(roidb)

    return imdb, roidb, ratio_list, ratio_index
