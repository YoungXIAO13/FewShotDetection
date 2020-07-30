from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import torch
import cv2
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import pickle
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
from model.nms.nms_wrapper import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections, vis_detections_label_only

from matplotlib import pyplot as plt
import torch.utils.data as Data
from model.utils.net_utils import weights_normal_init, save_net, load_net, \
    adjust_learning_rate, save_checkpoint, clip_gradient
#from tsne import plot_embedding
import collections

import pickle
try:
    xrange  # Python 2
except NameError:
    xrange = range  # Python 3


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a Meta R-CNN network')
    # Define Model and data
    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset:coco2017,coco,pascal_07_12',
                        default='pascal_07_12', type=str)
    parser.add_argument('--net', dest='net',
                        help='metarcnn',
                        default='metarcnn', type=str)
    # Define testing parameters
    parser.add_argument('--cuda', dest='cuda',
                        default=True, type=bool,
                        help='whether use CUDA')
    parser.add_argument('--cag', dest='class_agnostic',
                        default=False, type=bool,
                        help='whether perform class_agnostic bbox regression')
    # Define meta parameters
    parser.add_argument('--meta_test', dest='meta_test', default=False, type=bool,
                        help='whether perform meta testing')
    parser.add_argument('--meta_loss', dest='meta_loss', default=False, type=bool,
                        help='whether perform adding meta loss')

    parser.add_argument('--shots', dest='shots',
                        help='the number of meta input',
                        default=1, type=int)
    parser.add_argument('--meta_type', dest='meta_type', default=1, type=int,
                        help='choose which sets of metaclass')
    parser.add_argument('--phase', dest='phase',
                        help='the phase of training process',
                        default=1, type=int)
    # resume trained model
    parser.add_argument('--load_dir', dest='load_dir',
                        help='directory to load models', default="exps",
                        type=str)
    parser.add_argument('--checksession', dest='checksession',
                        help='checksession to load model',
                        default=3256, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load network',
                        default=12, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load network',
                        default=21985, type=int)
    # Others
    parser.add_argument('--bs', dest='batch_size',
                        help='batch_size',
                        default=1, type=int)
    parser.add_argument('--vis', dest='vis',
                        help='visualization mode',
                        action='store_true')
    parser.add_argument('--save', dest='save_dir',
                        help='directory to save logs', default='models',
                        type=str)
    args = parser.parse_args()
    return args


lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY

if __name__ == '__main__':
    args = parse_args()

    if args.net == 'metarcnn':
        from model.faster_rcnn.resnet import resnet
    print('Called with args:')
    print(args)
    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    np.random.seed(cfg.RNG_SEED)
    if args.dataset == "coco":
        args.imdb_name = "coco_2014_train+coco_2014_valminusminival"
        args.imdbval_name = "coco_2014_minival"
        args.set_cfgs = ['ANCHOR_SCALES', '[2, 4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']

    elif args.dataset == "pascal_voc_0712":
        args.imdbval_name = "voc_2007_test"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']

    elif args.dataset == "object3d":
        args.imdbval_name = "objectnet3d_val"
        args.set_cfgs = ['ANCHOR_SCALES', '[2, 4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.25, 0.5, 1, 2, 4]', 'MAX_NUM_GT_BOXES', '50']

    # the number of sets of metaclass
    cfg.TRAIN.META_TYPE = args.meta_type
    args.cfg_file = "cfgs/res101_ms.yml"
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    # print('Using config:')
    # pprint.pprint(cfg)

    cfg.TRAIN.USE_FLIPPED = False
    imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdbval_name, False)
    imdb.competition_mode(on=True)

    input_dir = args.load_dir
    if not os.path.exists(input_dir):
        raise Exception('There is no input directory for loading network from ' + input_dir)
    load_name = os.path.join(input_dir,
                             '{}_{}_{}_{}.pth'.format(args.dataset, str(args.net), args.checksession,
                                                         args.checkepoch))
    # initilize the network here.
    if args.net == 'metarcnn':
        num_layers = 101 if args.dataset == 'pascal_voc_0712' else 50

        if args.dataset == 'inter':
            num_cls = 21
        else:
            num_cls = imdb.num_classes

        fasterRCNN = resnet(num_cls, num_layers, pretrained=True, class_agnostic=args.class_agnostic,
                            meta_train=False, meta_test=args.meta_test, meta_loss=args.meta_loss)
    else:
        print('No module define')

    load_name = os.path.join(input_dir,
                             '{}_{}_{}_{}.pth'.format(args.dataset, str(args.net), args.checksession, args.checkepoch))
    fasterRCNN.create_architecture()
    print("load checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name)
    fasterRCNN.load_state_dict(checkpoint['model'])
    if 'pooling_mode' in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint['pooling_mode']
    print('load model successfully!')
    if args.cuda:
        cfg.CUDA = True

    if args.cuda:
        fasterRCNN.cuda()

    start = time.time()
    max_per_image = 100

    vis = args.vis
    if vis:
        thresh = 0.5
    else:
        thresh = 0.0001

    fasterRCNN.eval()
    empty_array = np.transpose(np.array([[], [], [], [], []]), (1, 0))

    # if meta test
    mean_class_attentions = None
    if args.meta_test:
        print('loading mean class attentions!')
        mean_class_attentions = pickle.load(open(os.path.join(
            input_dir, 'meta_type_{}'.format(args.meta_type),
            str(args.phase) + '_shots_' + str(args.shots) + '_mean_class_attentions.pkl'), 'rb'))

    num_images = len(imdb.image_index)
    all_boxes = [[[] for _ in range(num_images)] for _ in range(num_cls)]

    output_dir = os.path.join(input_dir, args.dataset)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dataset = roibatchLoader(roidb, ratio_list, ratio_index, args.batch_size,
                             num_cls, training=False, normalize=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=0, pin_memory=True)

    data_iter = iter(dataloader)

    _t = {'im_detect': time.time(), 'misc': time.time()}
    det_file = os.path.join(output_dir, 'detections.pkl')

    with torch.no_grad():
        for i in range(num_images):
            data = next(data_iter)
            im_data_list = []
            im_info_list = []
            gt_boxes_list = []
            num_boxes_list = []
            # initilize the tensor holder here.
            im_data = torch.FloatTensor(1)
            im_info = torch.FloatTensor(1)
            num_boxes = torch.LongTensor(1)
            gt_boxes = torch.FloatTensor(1)
            # ship to cuda
            if args.cuda:
                im_data = im_data.cuda()
                im_info = im_info.cuda()
                num_boxes = num_boxes.cuda()
                gt_boxes = gt_boxes.cuda()
            # make variable
            im_data = Variable(im_data)
            im_info = Variable(im_info)
            num_boxes = Variable(num_boxes)
            gt_boxes = Variable(gt_boxes)
            im_data.data.resize_(data[0].size()).copy_(data[0])
            im_info.data.resize_(data[1].size()).copy_(data[1])
            gt_boxes.data.resize_(data[2].size()).copy_(data[2])
            num_boxes.data.resize_(data[3].size()).copy_(data[3])

            im_data_list.append(im_data)
            im_info_list.append(im_info)
            gt_boxes_list.append(gt_boxes)
            num_boxes_list.append(num_boxes)
            det_tic = time.time()
            rois, \
            rpn_loss_cls, rpn_loss_box, \
            RCNN_loss_cls, RCNN_loss_bbox, \
            rois_label, cls_prob_list, bbox_pred_list, _ = fasterRCNN(im_data_list, im_info_list,
                                                                      gt_boxes_list,
                                                                      num_boxes_list,
                                                                      mean_class_attentions=mean_class_attentions)
            if args.meta_test:
                for clsidx in range(len(cls_prob_list)):
                    cls_prob = cls_prob_list[clsidx]
                    bbox_pred = bbox_pred_list[clsidx]
                    scores = cls_prob.data
                    boxes = rois.data[:, :, 1:5]
                    if cfg.TEST.BBOX_REG:
                        # Apply bounding-box regression deltas
                        box_deltas = bbox_pred.data
                        if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                            # Optionally normalize targets by a precomputed mean and stdev
                            if args.class_agnostic:
                                box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(
                                    cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                             + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                                box_deltas = box_deltas.view(1, -1, 4)
                            else:
                                box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(
                                    cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                             + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                                box_deltas = box_deltas.view(1, -1, 4 * num_cls)

                        pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
                        pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)

                    else:
                        # Simply repeat the boxes, once for each class
                        pred_boxes = np.tile(boxes, (1, scores.shape[1]))

                    pred_boxes /= im_info[0][2]
                    scores = scores.squeeze()
                    pred_boxes = pred_boxes.squeeze()
                    if clsidx == 0:
                        allscores = scores[:, clsidx].unsqueeze(1)
                        allpredboxes = pred_boxes if args.class_agnostic else pred_boxes[:, (clsidx) * 4:(clsidx + 1) * 4]

                        allscores = torch.cat([allscores, scores[:, (clsidx + 1)].unsqueeze(1)], dim=1)
                        allpredboxes = torch.cat([allpredboxes, pred_boxes], dim=1) if args.class_agnostic else \
                            torch.cat([allpredboxes, pred_boxes[:, (clsidx + 1) * 4:(clsidx + 2) * 4]], dim=1)
                    else:
                        allscores = torch.cat([allscores, scores[:, (clsidx + 1)].unsqueeze(1)], dim=1)
                        allpredboxes = torch.cat([allpredboxes, pred_boxes], dim=1) if args.class_agnostic else \
                            torch.cat([allpredboxes, pred_boxes[:, (clsidx + 1) * 4:(clsidx + 2) * 4]], dim=1)

                scores = allscores
                pred_boxes = allpredboxes
            else:
                scores = cls_prob_list.data
                boxes = rois.data[:, :, 1:5]
                if cfg.TEST.BBOX_REG:
                    # Apply bounding-box regression deltas
                    box_deltas = bbox_pred_list.data
                    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                        # Optionally normalize targets by a precomputed mean and stdev
                        if args.class_agnostic:
                            box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(
                                cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                         + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                            box_deltas = box_deltas.view(1, -1, 4)
                        else:
                            box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(
                                cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                         + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                            box_deltas = box_deltas.view(1, -1, 4 * num_cls)

                    pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
                    pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
                else:
                    # Simply repeat the boxes, once for each class
                    pred_boxes = np.tile(boxes, (1, scores.shape[1]))
                pred_boxes /= data[1][0][2]
                scores = scores.squeeze()

            pred_boxes = pred_boxes.squeeze()
            det_toc = time.time()
            detect_time = det_toc - det_tic
            misc_tic = time.time()
            if vis:
                im = cv2.imread(imdb.image_path_from_index(int(data[4])))
                im2show = np.copy(im)
            for j in range(1, num_cls):
                inds = torch.nonzero(scores[:, j] > thresh).view(-1)
                # if there is det
                if inds.numel() > 0:
                    cls_scores = scores[:, j][inds]
                    _, order = torch.sort(cls_scores, 0, True)
                    if args.class_agnostic:
                        cls_boxes = pred_boxes[inds, :]
                    else:
                        cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]
                    cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                    cls_dets = cls_dets[order]
                    keep = nms(cls_dets, cfg.TEST.NMS)
                    cls_dets = cls_dets[keep.view(-1).long()]
                    if vis:
                        im2show = vis_detections_label_only(im2show, imdb.classes[j], cls_dets.cpu().numpy(), 0.3)
                    all_boxes[j][i] = cls_dets.cpu().numpy()
                else:
                    all_boxes[j][i] = empty_array

            # Limit to max_per_image detections *over all classes*
            if max_per_image > 0:
                image_scores = np.hstack([all_boxes[j][i][:, -1] for j in range(1, num_cls)])
                if len(image_scores) > max_per_image:
                    image_thresh = np.sort(image_scores)[-max_per_image]
                    for j in range(1, num_cls):
                        keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                        all_boxes[j][i] = all_boxes[j][i][keep, :]

            misc_toc = time.time()
            nms_time = misc_toc - misc_tic

            sys.stdout.write(
                'im_detect: {:d}/{:d} {:.3f}s {:.3f}s  \r'.format(i + 1, num_images, detect_time, nms_time))
            sys.stdout.flush()

            if vis:
                im_dir = 'vis/' + str(data[4].numpy()[0]) + '_det.png'
                cv2.imwrite(im_dir, im2show)
                plt.imshow(im2show[:, :, ::-1])
                plt.show()

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')
    imdb.evaluate_detections(all_boxes, output_dir, **vars(args))
    end = time.time()
    print("test time: %0.4fs" % (end - start))
