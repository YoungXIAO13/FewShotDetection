# Few-Shot Object Detection

(ECCV 2020) PyTorch implementation of paper "Few-Shot Object Detection and Viewpoint Estimation for Objects in the Wild"\
[\[PDF\]]() [\[Project webpage\]](http://imagine.enpc.fr/~xiaoy/FSDetView/)

<p align="center">
<img src="https://github.com/YoungXIAO13/FewShotDetection/blob/master/img/PipelineDet.png" width="400px" alt="teaser">
</p>

If our project is helpful for your research, please consider citing:
```Bash
@INPROCEEDINGS{Xiao2020FSDetView,
    author    = {Yang Xiao and Renaud Marlet},
    title     = {Few-Shot Object Detetcion and Viewpoint Estimation for Objects in the Wild},
    booktitle = {European Conference on Computer Vision (ECCV)},
    year      = {2020}}
```

## Table of Content
* [Installation](#installation)
* [PASCAL](#training)
* [COCO](#testing)

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
It will compile all the Faster R-CNN modules you need. 

#### Data Preparation

Create a data folder under the repo,

```sh
cd {repo_root}
mkdir data
```
**PASCAL_VOC 07+12**: Please follow the instructions in [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn#beyond-the-demo-installation-for-training-and-testing-models) to prepare VOC datasets. 
After downloading the data, create softlinks in the folder ``data/``.

Following Meta R-CNN, please download the three base classes splits [Baidu](https://pan.baidu.com/s/11IxGujTTegLEXFsaiohV_Q) | 
[GoogleDrive](https://drive.google.com/drive/folders/14gtxnxWokk3eO6Oe5SrEG6_R9Dt6efT8?usp=sharing) 
and put them into VOC2007 and VOC2012 ImageSets/Main dirs.

**MS COCO**: 


### Training
We used [ResNet101](https://www.dropbox.com/s/iev3tkbz5wyyuz9/resnet101_caffe.pth?dl=0) pretrained model on ImageNet in our experiments. Download it and put it into the data/pretrained_model/.


#### the first phase
```sh
$>CUDA_VISIBLE_DEVICES=0 python train_metarcnn.py --dataset pascal_voc_0712 --epochs 21 --bs 4 --nw 8 --log_dir checkpoint --save_dir models/meta/first --meta_type 1 --meta_train True --meta_loss True 
```
#### the second phase
```sh
$>CUDA_VISIBLE_DEVICES=0 python train_metarcnn.py --dataset pascal_voc_0712 --epochs 30 --bs 4 --nw 8 --log_dir checkpoint --save_dir models/meta/first --r True --checksession 1 --checkepoch 20 --checkpoint 3081 --phase 2 --shots 10 --meta_train True --meta_loss True --meta_type 1
```
### Testing

if you want to evaluate the performance of meta trained model, simply run:
```sh
$>CUDA_VISIBLE_DEVICES=0 python test_metarcnn.py --dataset pascal_voc_0712 --net metarcnn --load_dir models/meta/first  --checksession 10 --checkepoch 30 --checkpoint 111 --shots 10  --meta_type 1 --meta_test True --meta_loss True --phase 2
```
