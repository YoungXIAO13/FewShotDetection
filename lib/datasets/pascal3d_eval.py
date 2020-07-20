# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import numpy as np
from math import radians


def parse_rec(df, filename):
    """ Parse PASCAL 3D annotation file """
    objects = []
    objs = df[df.im_path == filename]
    for ix in range(len(objs)):
        obj_struct = {}
        obj_struct['class'] = objs.iloc[ix]['cat']

        x1 = max(int(objs.iloc[ix]['left']), 0)
        y1 = max(int(objs.iloc[ix]['upper']), 0)
        x2 = min(int(objs.iloc[ix]['right']), int(objs.iloc[ix]['height'] - 1))
        y2 = min(int(objs.iloc[ix]['lower']), int(objs.iloc[ix]['width'] - 1))

        obj_struct['bbox'] = [x1, y1, x2, y2]

        obj_struct['difficult'] = objs.iloc[ix]['difficult']
        obj_struct['truncated'] = objs.iloc[ix]['truncated']
        obj_struct['occluded'] = objs.iloc[ix]['occluded']
        objects.append(obj_struct)

    return objects


def voc_ap(rec, prec):
    """
    Compute VOC AP given precision and recall.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def angles_to_matrix(angles):
    """Compute the rotation matrix from euler angles in degrees"""
    azi = radians(angles[0])
    ele = radians(angles[1])
    inp = radians(angles[2])
    element1 = np.cos(inp) * np.cos(azi) - np.sin(inp) * np.cos(ele) * np.sin(azi)
    element2 = np.sin(inp) * np.cos(azi) + np.cos(inp) * np.cos(ele) * np.sin(azi)
    element3 = np.sin(ele) * np.sin(azi)
    element4 = -np.cos(inp) * np.sin(azi) - np.sin(inp) * np.cos(ele) * np.cos(azi)
    element5 = -np.sin(inp) * np.sin(azi) + np.cos(inp) * np.cos(ele) * np.cos(azi)
    element6 = np.sin(ele) * np.cos(azi)
    element7 = np.sin(inp) * np.sin(ele)
    element8 = -np.cos(inp) * np.sin(ele)
    element9 = np.cos(ele)
    R_mat = np.array((element1, element2, element3,
                      element4, element5, element6,
                      element7, element8, element9)).reshape(3, 3)
    return R_mat


def azimuth_match_interval(azi_pred, azi_gt, view=24):
    offset = (360 / view) / 2
    step = 360 / view
    interval_pred = int((azi_pred + offset) % 360 // step)
    interval_gt = int((azi_gt + offset) % 360 // step)
    return interval_pred == interval_gt


def viewpoint_err(vp_pred, vp_gt):
    """ Compute the Rotation Matrix error between viewpoints"""
    R_pred = angles_to_matrix(vp_pred)
    R_gt = angles_to_matrix(vp_gt)
    R_err = np.arccos(((np.sum(R_pred * R_gt)).clip(-1., 3.) - 1.) / 2)
    return R_err


def pascal3d_eval(detpath, df, subset, classname, cachedir, ovthresh=0.5):
    """
    Top level function that does the PASCAL VOC evaluation.
    detpath: Path to detections
            detpath.format(classname) should produce the detection results file.
    df: Data frame get from annotation csv file for the subset [train / val / test]
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    """
    # assumes detections are in detpath.format(classname)
    # cachedir caches the annotations in a pickle file

    # first load gt
    df = df[df.set == subset]
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, '{}_annots.pkl'.format(subset))

    # read list of images
    imagenames = np.unique(df.im_path).tolist()

    if not os.path.isfile(cachefile):
        # load annotations
        recs = {}
        for i, imagename in enumerate(imagenames):
            recs[imagename] = parse_rec(df, imagename)
            if i % 100 == 0:
                print('Reading annotation for {:d}/{:d}'.format(i + 1, len(imagenames)))
        # save
        print('Saving cached annotations to {:s}'.format(cachefile))
        with open(cachefile, 'wb') as f:
            pickle.dump(recs, f)
    else:
        # load
        with open(cachefile, 'rb') as f:
            try:
                recs = pickle.load(f)
            except:
                recs = pickle.load(f, encoding='bytes')

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['class'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)

        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)

    if BB.shape[0] > 0:
        # sort by descending confidence
        sorted_ind = np.argsort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # iterate for each detection
        for d in range(nd):
            R = class_recs[image_ids[d]]
            bb = BB[d, :].astype(float)

            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)

            if BBGT.size > 0:
                # intersection
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin + 1., 0.)
                ih = np.maximum(iymax - iymin + 1., 0.)
                inters = iw * ih

                # union
                uni = (bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) + \
                      (BBGT[:, 2] - BBGT[:, 0] + 1.) * (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters

                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            # if the detection is correct
            if ovmax > ovthresh:
                if R['difficult'][jmax]:
                    continue
                if not R['det'][jmax]:
                    R['det'][jmax] = 1
                    tp[d] = 1.
                else:
                    fp[d] = 1.

            else:
                fp[d] = 1.

    # compute metrics AP
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec)

    return ap