# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler

from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import weights_normal_init, save_net, load_net, \
      adjust_learning_rate, save_checkpoint, clip_gradient

from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet_GNN import resnet

import pickle

def cout_w(prob):
    prob_weight = prob
    sum_value = np.sum(prob_weight)
    prob_weight = prob_weight / sum_value
    meam_value = np.mean(prob_weight)

    prob_weight = prob_weight - meam_value + 1
    prob_weight = np.log(prob_weight)
    return prob_weight

def cout_a(prob, NUM_ATTR_REL):
    prob_weight = prob[:, : NUM_ATTR_REL]
    sum_value = np.sum(prob_weight, keepdims=True, axis=1)
    zero_inds = np.nonzero(sum_value == 0)[0]
    sum_value[zero_inds,:] = 1.

    prob_weight = prob_weight / np.repeat(sum_value, prob_weight.shape[1], axis=1)

    meam_value = np.mean(prob_weight, keepdims=True, axis=1)
    meam_value = np.repeat(meam_value, prob_weight.shape[1], axis=1)
    prob_weight = prob_weight - meam_value + 1
    prob_weight = np.log(prob_weight)

    cls_cls_a = np.zeros((1001, 1001))
    for row in range(prob_weight.shape[0]):
        temp = np.zeros(1001)
        for col in range(prob_weight.shape[1]):
            xx = prob_weight[row, col] * prob_weight[:, col]
            temp += xx
        cls_cls_a[row] = temp
    pickle.dump(cls_cls_a, open('data/vg/clscls_attr.pkl', 'wb'))

    return cls_cls_a

def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--dataset', dest='dataset',
                      help='training dataset',
                      default='vg', type=str)
  parser.add_argument('--net', dest='net',
                    help='vgg16, res101',
                    default='baseline', type=str)
  parser.add_argument('--start_epoch', dest='start_epoch',
                      help='starting epoch',
                      default=1, type=int)
  parser.add_argument('--epochs', dest='max_epochs',
                      help='number of epochs to train',
                      default=20, type=int)
  parser.add_argument('--disp_interval', dest='disp_interval',
                      help='number of iterations to display',
                      default=100, type=int)
  parser.add_argument('--checkpoint_interval', dest='checkpoint_interval',
                      help='number of iterations to display',
                      default=10000, type=int)

  parser.add_argument('--save_dir', dest='save_dir',
                      help='directory to save models', default="exps/baseline/models",
                      nargs=argparse.REMAINDER)
  parser.add_argument('--nw', dest='num_workers',
                      help='number of worker to load data',
                      default=2, type=int)
  parser.add_argument('--cuda', dest='cuda',default=True, type=bool,
                      help='whether use CUDA')
  parser.add_argument('--ls', dest='large_scale',
                      help='whether use large imag scale',
                      action='store_true')
  parser.add_argument('--ms', dest='multi_scale',
                      help='whether to use multi scale training',
                      action='store_true')
  parser.add_argument('--mGPUs', dest='mGPUs',
                      help='whether use multiple GPUs',
                      action='store_true')
  parser.add_argument('--bs', dest='batch_size',
                      help='batch_size',
                      default=2, type=int)
  parser.add_argument('--cag', dest='class_agnostic',default=False, type=bool,
                      help='whether perform class_agnostic bbox regression')

# config optimization
  parser.add_argument('--o', dest='optimizer',
                      help='training optimizer',
                      default="sgd", type=str)
  parser.add_argument('--lr', dest='lr',
                      help='starting learning rate',
                      default=0.001, type=float)
  parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                      help='step to do learning rate decay, unit is epoch',
                      default=4, type=int)
  parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                      help='learning rate decay ratio',
                      default=0.1, type=float)

# set training session
  parser.add_argument('--s', dest='session',
                      help='training session',
                      default=1, type=int)

# resume trained model
  parser.add_argument('--r', dest='resume',
                      help='resume checkpoint or not',
                      default=False, type=bool)
  parser.add_argument('--checksession', dest='checksession',
                      help='checksession to load model',
                      default=1, type=int)
  parser.add_argument('--checkepoch', dest='checkepoch',
                      help='checkepoch to load model',
                      default=1, type=int)
  parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load model',
                      default=0, type=int)
# log and diaplay
  parser.add_argument('--use_tfboard', dest='use_tfboard',
                      help='whether use tensorflow tensorboard',
                      default=False, type=bool)
  parser.add_argument('--log_dir', dest='log_dir',
                      help='directory to save logs', default='logs',
                      type=str)

  args = parser.parse_args()
  return args


class sampler(Sampler):
  def __init__(self, train_size, batch_size):
    self.num_data = train_size
    self.num_per_batch = int(train_size / batch_size)
    self.batch_size = batch_size
    self.range = torch.arange(0,batch_size).view(1, batch_size).long()
    self.leftover_flag = False
    if train_size % batch_size:
      self.leftover = torch.arange(self.num_per_batch*batch_size, train_size).long()
      self.leftover_flag = True

  def __iter__(self):
    rand_num = torch.randperm(self.num_per_batch).view(-1,1) * self.batch_size
    self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range
    self.rand_num_view = self.rand_num.view(-1)

    if self.leftover_flag:
      self.rand_num_view = torch.cat((self.rand_num_view, self.leftover),0)

    return iter(self.rand_num_view)

  def __len__(self):
    return self.num_data


  args = parse_args()

  print('Called with args:')
  print(args)

  if args.use_tfboard:
    from model.utils.logger import Logger
    # Set the logger
    logger = Logger(args.log_dir)

  if args.dataset == "pascal_voc":
      args.imdb_name = "voc_2007_trainval"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
  elif args.dataset == "pascal_voc_0712":
      args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
  elif args.dataset == "coco":
      args.imdb_name = "coco_2014_train+coco_2014_valminusminival"
      args.imdbval_name = "coco_2014_minival"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
  elif args.dataset == "imagenet":
      args.imdb_name = "imagenet_train"
      args.imdbval_name = "imagenet_val"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']
  elif args.dataset == "vg":
      # train sizes: train, smalltrain, minitrain
      # train scale: ['150-50-20', '150-50-50', '500-150-80', '750-250-150', '1750-700-450', '1600-400-20']
      args.imdb_name = "vg_train"
      args.imdbval_name = "vg_val"
      args.set_cfgs = ['ANCHOR_SCALES', '[2, 4, 8, 16, 32]', 'MAX_NUM_GT_BOXES', '50']
      # args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']

  # args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)
  args.cfg_file = "cfgs/res101_ms.yml"#.format(args.net + "_ms" if args.multi_scale else "")

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  print('Using config:')
  pprint.pprint(cfg)
  np.random.seed(cfg.RNG_SEED)

  #torch.backends.cudnn.benchmark = True
  if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

  # train set
  # -- Note: Use validation set and disable the flipped to enable faster loading.
  cfg.TRAIN.USE_FLIPPED = True
  cfg.USE_GPU_NMS = args.cuda
  imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdb_name)
  train_size = len(roidb)

  print('{:d} roidb entries'.format(len(roidb)))
  sys.stdout.flush()

  output_dir = args.save_dir[0] + "/" + args.net + "/" + args.dataset
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  # log_f = open(os.path.join(args.log_dir, time.strftime("%Y-%m-%d-%H:%M.txt", time.localtime())), "w")
  # old_stdout = sys.stdout
  # sys.stdout = log_f

  sampler_batch = sampler(train_size, args.batch_size)

  dataset = roibatchLoader(roidb, ratio_list, ratio_index, args.batch_size, imdb.num_classes, training=True)

  dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                            sampler=sampler_batch, num_workers=args.num_workers, pin_memory=False)

  # initilize the tensor holder here.
  im_data = torch.FloatTensor(1)
  im_info = torch.FloatTensor(1)
  num_boxes = torch.LongTensor(1)
  gt_boxes = torch.FloatTensor(1)
#  attr_prob = torch.FloatTensor(imdb._clscls_attr)
  # attr_prob = torch.FloatTensor(cout_a(imdb._class_to_attr, 200))
#  pair_prob = torch.FloatTensor(cout_w(imdb.pair))
#  pair_prob = pair_prob.unsqueeze(0).repeat(args.batch_size, 1, 1)


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
#  pair_prob = Variable(pair_prob)
#  attr_prob = Variable(attr_prob)

  if args.cuda:
    cfg.CUDA = True

  # initilize the network here.
  if args.net == 'baseline':
    fasterRCNN = resnet(imdb.classes, 101, pretrained=True, class_agnostic=args.class_agnostic)
  elif args.net == 'relationloss':
    fasterRCNN = resnet(imdb.classes, 101, pretrained=False, class_agnostic=args.class_agnostic, pair_prob=pair_prob)
  elif args.net == 'attributeloss':
    fasterRCNN = resnet(imdb.classes, 101, pretrained=True, class_agnostic=args.class_agnostic, attr_prob=attr_prob)


  fasterRCNN.create_architecture()

  lr = cfg.TRAIN.LEARNING_RATE
  lr = args.lr
  #tr_momentum = cfg.TRAIN.MOMENTUM
  #tr_momentum = args.momentum

  params = []
  for key, value in dict(fasterRCNN.named_parameters()).items():
    if value.requires_grad:
      if 'bias' in key:
        params += [{'params':[value],'lr':lr*(cfg.TRAIN.DOUBLE_BIAS + 1), \
                'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
      else:
        params += [{'params':[value],'lr':lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

  if args.optimizer == "adam":
    lr = lr * 0.1
    optimizer = torch.optim.Adam(params)

  elif args.optimizer == "sgd":
    optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

  if args.resume:
    load_name = os.path.join(output_dir,
      'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))
    print("loading checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name)
    args.session = checkpoint['session']
    args.start_epoch = checkpoint['epoch']
    fasterRCNN.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    lr = optimizer.param_groups[0]['lr']
    if 'pooling_mode' in checkpoint.keys():
      cfg.POOLING_MODE = checkpoint['pooling_mode']
    print("loaded checkpoint %s" % (load_name))

  if args.mGPUs:
    fasterRCNN = nn.DataParallel(fasterRCNN)

  if args.cuda:
    fasterRCNN.cuda()

  iters_per_epoch = int(train_size / args.batch_size)


fasterRCNN = resnet(imdb.classes, 101, pretrained=True, class_agnostic=args.class_agnostic)
fasterRCNN.create_architecture()
fasterRCNN.cuda()
fasterRCNN.train()
data_iter = iter(dataloader)

data = next(data_iter)
im_data.data.resize_(data[0].size()).copy_(data[0])
im_info.data.resize_(data[1].size()).copy_(data[1])
gt_boxes.data.resize_(data[2].size()).copy_(data[2])
num_boxes.data.resize_(data[3].size()).copy_(data[3])

fasterRCNN.zero_grad()



batch_size = im_data.size(0)

im_info = im_info.data
gt_boxes = gt_boxes.data
num_boxes = num_boxes.data

# feed image data to base model to obtain base feature map
base_feat = fasterRCNN.RCNN_base(im_data)

# feed base feature map tp RPN to obtain rois
rois, rpn_loss_cls, rpn_loss_bbox = fasterRCNN.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes)

# if it is training phrase, then use ground trubut bboxes for refining
if fasterRCNN.training:
    roi_data = fasterRCNN.RCNN_proposal_target(rois, gt_boxes, num_boxes)
    rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data

    rois_label = Variable(rois_label.view(-1).long())
    rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
    rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
    rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
else:
    rois_label = None
    rois_target = None
    rois_inside_ws = None
    rois_outside_ws = None
    rpn_loss_cls = 0
    rpn_loss_bbox = 0

rois = Variable(rois)
# do roi pooling based on predicted rois

if cfg.POOLING_MODE == 'crop':
    # pdb.set_trace()
    # pooled_feat_anchor = _crop_pool_layer(base_feat, rois.view(-1, 5))
    grid_xy = _affine_grid_gen(rois.view(-1, 5), base_feat.size()[2:], fasterRCNN.grid_size)
    grid_yx = torch.stack([grid_xy.data[:,:,:,1], grid_xy.data[:,:,:,0]], 3).contiguous()
    pooled_feat = fasterRCNN.RCNN_roi_crop(base_feat, Variable(grid_yx).detach())
    if cfg.CROP_RESIZE_WITH_MAX_POOL:
        pooled_feat = F.max_pool2d(pooled_feat, 2, 2)
elif cfg.POOLING_MODE == 'align':
    pooled_feat = fasterRCNN.RCNN_roi_align(base_feat, rois.view(-1, 5))
elif cfg.POOLING_MODE == 'pool':
    pooled_feat = fasterRCNN.RCNN_roi_pool(base_feat, rois.view(-1,5))