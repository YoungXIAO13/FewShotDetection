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


## Installation

Code built on top of [MetaR-CNN](https://github.com/yanxp/MetaR-CNN) with requirement of 
PyTorch 0.4.0 compiled with CUDA 8.0 and gcc >= 4.9. 

Create conda env:
```sh
conda create --name FSDet --file spec-file.txt
source activate FSDet
```

Compile the CUDA dependencies:
```sh
cd {repo_root}/lib
sh make.sh
```

#### Data Preparation

**Pre-trained ResNet**:
We used [ResNet101](https://www.dropbox.com/s/iev3tkbz5wyyuz9/resnet101_caffe.pth?dl=0) pretrained model on ImageNet in our experiments. 
Download it and put it into the ``data/pretrained_model/``.

**PASCAL_VOC**: 
Download PASCAL VOC 2007 and 2012 datasets, create softlink named ``VOCdevkit`` in the folder ``data/``.

Following Meta R-CNN, please download the three base classes splits [Baidu](https://pan.baidu.com/s/11IxGujTTegLEXFsaiohV_Q) | 
[GoogleDrive](https://drive.google.com/drive/folders/14gtxnxWokk3eO6Oe5SrEG6_R9Dt6efT8?usp=sharing) 
and put them into VOC2007 and VOC2012 ImageSets/Main dirs.

**MS COCO**: 
Download COCO 2014 dataset, create softlink named ``coco`` in the folder ``data/``.


## Commands

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

Testing results will be writen in ``./save_models/{exp}/{dataset_name}/Kshots_out.txt``.
