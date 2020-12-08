import torch.utils.data as data
import cv2
import torch
import collections
import time
import os
import numpy as np
import json
import os.path as osp

from model.utils.config import cfg
from pycocotools.coco import COCO


class MetaDatasetTFA(data.Dataset):
    def __init__(self, root, image_set, year, img_size, shots=10):
        self.img_size = img_size
        self.TFA_split = os.path.join(root, 'annotations', 'TFA', 'cocosplit')
        self.shot_path = os.path.join(root, 'annotations', 'instances_TFA{}shot2014.json'.format(shots))
        self.shots = shots

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

        self.metaclass = tuple([c['name'] for c in cats if c['name'] not in cfg.VOC_CLASSES] +
                               [c['name'] for c in cats if c['name'] in cfg.VOC_CLASSES])
        class_to_idx = dict(zip(self.metaclass, range(len(self.metaclass))))  # class to index mapping

        self.prndata = []
        self.prncls = []

        prn_image_pth = os.path.join(root, 'annotations', 'TFA', 'prn_image_{}shots.pt'.format(shots))
        prn_mask_pth = os.path.join(root, 'annotations', 'TFA', 'prn_mask_{}shots.pt'.format(shots))

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

    def _get_ann_file(self):
        prefix = 'instances' if self._image_set.find('test') == -1 else 'image_info'
        return osp.join(self._data_path, 'annotations', prefix + '_' + self._image_set + self._year + '.json')

    def get_prndata(self):
        print('\ngenerating {} shot samples...'.format(self.shots))
        start = time.time()

        prn_image = collections.defaultdict(list)
        prn_mask = collections.defaultdict(list)
        classes = collections.defaultdict(int)

        sample_images = []
        sample_annots = []

        for cls in self.metaclass:
            cls_json_file = 'full_box_{}shot_{}_trainval.json'.format(self.shots, cls)
            samples = json.load(open(osp.join(self.TFA_split, cls_json_file), 'r'))

            sample_annots.extend(samples['annotations'])
            sample_images.extend(samples['images'])

            for d_annot in samples['annotations']:
                img_id = d_annot['image_id']

                for d_img in samples['images']:
                    if d_img['id'] == img_id:
                        img_info = d_img
                        break

                width = img_info['width']
                height = img_info['height']

                x1 = np.max((0, d_annot['bbox'][0]))
                y1 = np.max((0, d_annot['bbox'][1]))
                x2 = np.min((width - 1, x1 + np.max((0, d_annot['bbox'][2] - 1))))
                y2 = np.min((height - 1, y1 + np.max((0, d_annot['bbox'][3] - 1))))

                if d_annot['area'] > 0 and x2 >= x1 and y2 >= y1:
                    set_name = img_info['file_name'].split('_')[1]
                    img_path = osp.join(self._data_path, 'images', set_name, img_info['file_name'])
                    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                    img = img.astype(np.float32, copy=False)
                    img -= cfg.PIXEL_MEANS

                    mask = np.zeros((self.img_size, self.img_size), dtype=np.float32)
                    h, w, _ = img.shape
                    y_ration = float(h) / self.img_size
                    x_ration = float(w) / self.img_size
                    img_resize = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

                    classes[cls] += 1
                    x1_r = int(x1 / x_ration)
                    y1_r = int(y1 / y_ration)
                    x2_r = int(x2 / x_ration)
                    y2_r = int(y2 / y_ration)

                    mask[y1_r:y2_r, x1_r:x2_r] = 1

                    prn_image[cls].append(img_resize)
                    prn_mask[cls].append(mask)

        end = time.time()
        print('few-shot samples generated in {} s\n'.format(end - start))
        print('minimum sample number among all classes is {}'.format(min(classes.values())))

        self.json_data['images'] = sample_images
        self.json_data['annotations'] = sample_annots

        def convert(o):
            if isinstance(o, np.int64): return int(o)
            raise TypeError

        with open(self.shot_path, 'w') as f:
            json.dump(self.json_data, f, default=convert)

        return prn_image, prn_mask
