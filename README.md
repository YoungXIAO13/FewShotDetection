## Meta R-CNN : Towards General Solver for Instance-level Low-shot Learning.

Code for reproducing the results in the following paper, and the code is built on top of [jwyang/faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch)

**<a href="https://yanxp.github.io/metarcnn.html">Meta R-CNN : Towards General Solver for Instance-level Low-shot Learning</a>**

<a href="https://yanxp.github.io/">Xiaopeng Yan*</a>,
<a href="http://cziliang.com">Ziliang Chen*</a>,
Anni Xu, Xiaoxi Wang, 
<a href="https://lemondan.github.io/">Xiaodan Liang</a>,
<a href="http://www.linliang.net/">Liang Lin</a>

Sun Yat-Sen University, Presented at  *IEEE International Conference on Computer Vision [(ICCV2019)](http://iccv2019.thecvf.com/)*	

<p align=center><img width="80%" src="demo/metarcnn.png"/></p>

### License

For Academic Research Use Only!

### Requirements

+ python packages

  + PyTorch = 0.3.1
    
    *This project can not support pytorch 0.4, higher version will not recur results.*

  + Torchvision >= 0.2.0

  + cython

  + pyyaml

  + easydict

  + opencv-python

  + matplotlib

  + numpy

  + scipy

  + tensorboardX

    You can install above package using ```pip```:

    ```sh
    pip install Cython easydict matplotlib opencv-python pyyaml scipy
    ```

+ CUDA 8.0

+ gcc >= 4.9

### Misc

Tested on Ubuntu 14.04 with a Titan X GPU (12G) and Intel(R) Xeon(R) CPU E5-2623 v3 @ 3.00GHz.

### Getting Started

Clone the repo:

```
https://github.com/yanxp/MetaR-CNN.git
```

### Compilation

Compile the CUDA dependencies:

```sh
cd {repo_root}/lib
sh make.sh
```
It will compile all the modules you need, including NMS, ROI_Pooing, ROI_Crop and ROI_Align. 

### Data Preparation

Create a data folder under the repo,

```sh
cd {repo_root}
mkdir data
```
**PASCAL_VOC 07+12**: Please follow the instructions in [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn#beyond-the-demo-installation-for-training-and-testing-models) to prepare VOC datasets. Actually, you can refer to any others. After downloading the data, create softlinks in the folder data/.

please download the three base classes [splits](https://pan.baidu.com/s/11IxGujTTegLEXFsaiohV_Q)[[GoogleDrive](https://drive.google.com/drive/folders/14gtxnxWokk3eO6Oe5SrEG6_R9Dt6efT8?usp=sharing)] and put them into VOC2007 and VOC2012 ImageSets/Main dirs.

### Training
We used [ResNet101](https://www.dropbox.com/s/iev3tkbz5wyyuz9/resnet101_caffe.pth?dl=0) pretrained model on ImageNet in our experiments. Download it and put it into the data/pretrained_model/.

for example, if you want to train the first split of base and novel class with meta learning, just run:

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

we provide the part models with meta training and without meta training in the following:
[Meta Models](https://pan.baidu.com/s/1N3PW9WTi82lbdURNAz7EFA)[[GoogleDrive](https://drive.google.com/file/d/19gapxklxKCwYIyGszOMhQKNDqYOLeubn/view?usp=sharing)] and [WoMeta Models](https://pan.baidu.com/s/1GkjUJmaOaEWzh3z2fs7ieA)[[GoogleDrive](https://drive.google.com/file/d/1G6xYH9M_bAAqUec1ARufv0ELi_pd7ERj/view?usp=sharing)]

### Citation

```
@inproceedings{yanICCV19metarcnn,
    Author = {Yan, Xiaopeng and Chen, Ziliang and Xu, Anni and Wang, Xiaoxi and Liang, Xiaodan and Lin, Liang},
    Title = {Meta R-CNN : Towards General Solver for Instance-level Low-shot Learning.},
    Booktitle = {Proc. of IEEE International Conference on Computer Vision ({ICCV})},
    Year = {2019}
}
```

### Contact

If you have any questions about this repo, please feel free to contact [yanxp3@mail3.sysu.edu.cn](mailto:yanxp3@mail3.sysu.edu.cn).
