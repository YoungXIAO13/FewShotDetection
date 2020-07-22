# Few-Shot Object Detection

(ECCV 2020) PyTorch implementation of paper "Few-Shot Object Detection and Viewpoint Estimation for Objects in the Wild"\
[\[PDF\]]() [\[Project webpage\]](http://imagine.enpc.fr/~xiaoy/FSDetView/)

<p align="center">
<img src="https://github.com/YoungXIAO13/FewShotDetection/blob/master/img/PipelineDet.png" width="800px" alt="teaser">
</p>

If our project is helpful for your research, please consider citing:
```Bash
@INPROCEEDINGS{Xiao2020FSDetView,
    author    = {Yang Xiao and Renaud Marlet},
    title     = {Few-Shot Object Detetcion and Viewpoint Estimation for Objects in the Wild},
    booktitle = {European Conference on Computer Vision (ECCV)},
    year      = {2020}}
```

## Tabel of Contents
* [Installation](#installation)
* [Data Preparation](#data-preparation)
* [Getting Start](#getting-start)


## Installation

Code built on top of [MetaR-CNN](https://github.com/yanxp/MetaR-CNN).
 
**Requirements**

* CUDA 8.0
* Python=3.6
* PyTorch=0.4.0
* torchvision=0.2.1
* gcc >= 4.9 

**Build**

Create conda env:
```sh
conda create --name FSdetection --file spec-file.txt
conda activate FSdetection
```

Compile the CUDA dependencies:
```sh
cd {repo_root}/lib
sh make.sh
```

## Data Preparation

We evaluate our method on two commonly-used benchmarks:

####[PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/)
 
We use the train/val sets of PASCAL VOC 2007+2012 for training and the test set of PASCAL VOC 2007 for evaluation. 
We split the 20 object classes into 15 base classes and 5 novel classes, and we consider 3 splits proposed in [FSRW](https://github.com/ucbdrive/few-shot-object-detection/blob/master/fsdet/data/datasets/builtin_meta.py). 

* Download PASCAL VOC 2007+2012 datasets, create softlink named ``VOCdevkit`` in the folder ``data/``.

* Data structure should look like:
```
data/VOCdevkit
    VOC{2007,2012}/
        Annotations/
        ImageSets/
        JPEGImages/
```

* Following **Meta R-CNN**, please download the three base classes splits [Baidu](https://pan.baidu.com/s/11IxGujTTegLEXFsaiohV_Q) | 
[GoogleDrive](https://drive.google.com/drive/folders/14gtxnxWokk3eO6Oe5SrEG6_R9Dt6efT8?usp=sharing) 
and put them into VOC2007 and VOC2012 ImageSets/Main dirs.

####[COCO](https://cocodataset.org/#home) 

We use COCO 2014 and keep the 5k images from minival set for evaluation and use the rest for training. 
We use the 20 object classes that are the same with PASCAL VOC as novel classes and use the rest as base classes.

* Download COCO 2014 dataset, create softlink named ``coco`` in the folder ``data/``.

* Data structure should look like:
```
data/coco
    annotations/
    images/
        train2014/
        val2014/
```

## Getting Start

**Pre-trained ResNet**:
We used [ResNet101](https://www.dropbox.com/s/iev3tkbz5wyyuz9/resnet101_caffe.pth?dl=0) pretrained model on ImageNet in our experiments. 
Download it and put it into the ``data/pretrained_model/``.

#### Command lines

Training and testing on **PASCAL_VOC**:
```sh
# the first split on VOC
bash run/run1.sh

# the second split on VOC
bash run/run2.sh

# the third split on VOC
bash run/run3.sh
```

Training and testing on **COCO**:
```sh
bash run/run_coco.sh
```

**Testing results** will be writen in ``./save_models/{exp}/{dataset_name}/Kshots_out.txt``.
