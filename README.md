# Few-Shot Object Detection

(ECCV 2020) PyTorch implementation of paper "Few-Shot Object Detection and Viewpoint Estimation for Objects in the Wild"\
[\[PDF\]](https://arxiv.org/abs/2007.12107) [\[Project webpage\]](http://imagine.enpc.fr/~xiaoy/FSDetView/) [\[Code (Viewpoint)\]](https://github.com/YoungXIAO13/FewShotViewpoint)

<p align="center">
<img src="https://github.com/YoungXIAO13/FewShotDetection/blob/master/img/PipelineDet.png" width="800px" alt="teaser">
</p>

If our project is helpful for your research, please consider citing:
```
@INPROCEEDINGS{Xiao2020FSDetView,
    author    = {Yang Xiao and Renaud Marlet},
    title     = {Few-Shot Object Detection and Viewpoint Estimation for Objects in the Wild},
    booktitle = {European Conference on Computer Vision (ECCV)},
    year      = {2020}}
```

## ChangeLog
* **[Dec-15 2020]** There could be a problem for the COCO base model weight downloaded in ``download_models.sh``, please download the correct model weight from [here](https://drive.google.com/file/d/1BlpUNxuSVcictDGp5mnwiZJm2NoEtPXP/view?usp=sharing) and put it in ``save_models/COCO/``.


## Tabel of Contents
* [Installation](#installation)
* [Data Preparation](#data-preparation)
* [Getting Started](#getting-started)
* [Quantitative Results](#quantitative-results)


## Installation

Code built on top of [MetaR-CNN](https://github.com/yanxp/MetaR-CNN).
 
**Requirements**

* CUDA 8.0
* Python=3.6
* PyTorch=0.4.0
* torchvision=0.2.1
* gcc >= 4.9 

**Build**

* Create conda env:
```sh
conda create --name FSdetection --file spec-file.txt
conda activate FSdetection
```

* Compile the CUDA dependencies:
```sh
cd {repo_root}/lib
sh make.sh
```

## Data Preparation

We evaluate our method on two commonly-used benchmarks. 
See [data/README.md](https://github.com/YoungXIAO13/FewShotDetection/tree/master/data/README.md) for more details.

### PASCAL VOC
 
We use the train/val sets of PASCAL VOC 2007+2012 for training and the test set of PASCAL VOC 2007 for evaluation. 
We split the 20 object classes into 15 base classes and 5 novel classes, and we consider 3 splits proposed in [FSRW](https://github.com/ucbdrive/few-shot-object-detection/blob/master/fsdet/data/datasets/builtin_meta.py). 

Download [PASCAL VOC 2007+2012](http://host.robots.ox.ac.uk/pascal/VOC/), create softlink named ``VOCdevkit`` in the folder ``data/``.


### COCO

We use COCO 2014 and keep the 5k images from minival set for evaluation and use the rest for training. 
We use the 20 object classes that are the same with PASCAL VOC as novel classes and use the rest as base classes.

Download [COCO 2014](https://cocodataset.org/#home), create softlink named ``coco`` in the folder ``data/``.
Please follow the instructins [here](https://github.com/rbgirshick/py-faster-rcnn/tree/master/data) to download the [instances_minival2014.json](https://dl.dropboxusercontent.com/s/o43o90bna78omob/instances_minival2014.json.zip?dl=0) and [instances_valminusminival2014.json](https://dl.dropboxusercontent.com/s/s3tw5zcg7395368/instances_valminusminival2014.json.zip?dl=0).


## Getting Started

### 1. Base-Class Training

**Pre-trained ResNet**:
folloing Meta R-CNN, we used [ResNet101](https://www.dropbox.com/s/iev3tkbz5wyyuz9/resnet101_caffe.pth?dl=0) for PASCAL VOC and ResNet50 for MS-COCO.
Download it and put it into the ``data/pretrained_model/``.

* We provide pre-trained models of **base-class training**:
```bash
bash download_models.sh
```
You will get a dir like:
```
save_models/
    COCO/
    VOC_first/
    VOC_second/
    VOC_third/
```

* You can also train it yourself:
```bash
# the first split on VOC
bash run/train_voc_first.sh

# the second split on VOC
bash run/train_voc_second.sh

# the third split on VOC
bash run/train_voc_third.sh

# NonVOC / VOC split on COCO
bash run/train_coco.sh
```

### 2. Few-Shot Fine-tuning

Fine-tune the base-training models on a balanced training data including both base and novel classes 
(3K instances per base class and K instances per novel class):
```bash
bash run/finetune_voc_first.sh

bash run/finetune_voc_second.sh

bash run/finetune_voc_third.sh

bash run/finetune_coco.sh
```


### 3. Testing

Evaluation is conducted on the test set of PASCAL VOC 2007 or minival set of COCO 2014:
```bash
bash run/test_voc_first.sh

bash run/test_voc_second.sh

bash run/test_voc_third.sh

bash run/test_coco.sh
```


## Quantitative Results

### Multiple Runs

By running multiple times (~10) the few-shot fine-tuning experiments and averaging the results, we got the performance below:

**Pascal-VOC (AP@50)**

|          | Split-1 (Base) | Split-1 (Novel) | Split-2 (Base) | Split-2 (Novel) | Split-3 (Base) | Split-3 (Novel) |
| :------: | :------:       | :------:        | :------:       | :------:        | :------:       | :------:        |
| K=1      |  64.2          |   24.2          |   66.9         |    21.6         |   66.7         |    21.1         |
| K=2      |  67.8          |   35.3          |   69.9         |    24.6         |   69.1         |    30.0         |
| K=3      |  69.4          |   42.2          |   70.8         |    31.9         |   69.9         |    37.2         |
| K=5      |  69.8          |   49.1          |   71.4         |    37.0         |   70.9         |    43.8         |
| K=10     |  71.1          |   57.4          |   72.2         |    45.7         |   72.2         |    49.6         |



**MS-COCO**

|          | AP (Base) | AP@50 (Base) | AP@75 (Base) | AP (Novel) | AP@50 (Novel) | AP@75 (Novel) |
| :------: | :------:  | :------:     | :------:     | :------:   | :------:      | :------:      |
| K=1      |  3.6      |   9.8        |   1.7        |    4.5     |   12.4        |    2.2        |
| K=2      |  5.0      |   13.0       |   2.7        |    6.6     |   17.1        |    3.5        |
| K=3      |  5.9      |   14.7       |   3.9        |    7.2     |   18.7        |    3.7        |
| K=5      |  8.6      |   20.3       |   6.0        |    10.7    |   24.5        |    6.7        |
| K=10     |  10.5     |   23.3       |   8.2        |    12.5    |   27.3        |    9.8        |
| K=30     |  12.7     |   26.1       |   9.7        |    14.7    |   30.6        |    12.2       |


### Specific Split

For a direct and quick comparison on COCO, we also run experiments using the specific few-shot sample split provided in [TFA](https://github.com/ucbdrive/few-shot-object-detection).

* Download their [json files](http://dl.yf.io/fs-det/datasets/) into the annotation folder of COCO:
```bash
cd ./data/coco/annotations
mkdir TFA && cd TFA
wget -r --no-parent  http://dl.yf.io/fs-det/datasets/cocosplit/
mv dl.yf.io/fs-det/datasets/cocosplit/ cocosplit && rm -r dl.yf.io 
```

* You will see a set of json files in format **"full_box_{K}shot_{cls}_trainval.json"** as well as 9 folders named "seed{i}"

* Then you can run the command line in ``run/finetune_coco_TFA.sh`` to finish few-shot fine-tuning and testing


**MS-COCO**

We get the following results:

|          | AP (Base) | AP@50 (Base) | AP@75 (Base) | AP (Novel) | AP@50 (Novel) | AP@75 (Novel) |
| :------: | :------:  | :------:     | :------:     | :------:   | :------:      | :------:      |
| K=1      |  2.4      |   7.0        |   1.0        |    3.2     |   8.9         |    1.4        |
| K=2      |  4.4      |   11.9       |   2.2        |    4.9     |   13.3        |    2.3        |
| K=3      |  4.9      |   13.6       |   2.2        |    6.7     |   18.6        |    2.9        |
| K=5      |  7.0      |   17.5       |   4.4        |    8.1     |   20.1        |    4.4        |
| K=10     |  9.0      |   21.2       |   6.1        |    10.7    |   25.6        |    6.5        |
| K=30     |  12.3     |   26.4       |   10.2       |    15.9    |   31.7        |    15.1       | 

**Note**: the difference between the performance of multiple runs and the performance of this specific split can be explained by the different sample configuration in the few-shot fine-tuning stage.
As we follow the strategy proposed in [MetaR-CNN](https://github.com/yanxp/MetaR-CNN) to sample 3K instances of base class and K instances of novel class, 
without any specific adjustment, the performance would naturally degrade when we simply use the split of [TFA](https://github.com/ucbdrive/few-shot-object-detection) where only K instances are considered for each base class.   
