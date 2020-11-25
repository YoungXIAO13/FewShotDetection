# Few-Shot Object Detection (Data Preparation)

First go to the data dir ``cd ./data``

## PASCAL VOC

Download data from official website:
```bash
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
tar xvf VOCtrainval_06-Nov-2007.tar && rm VOCtrainval_06-Nov-2007.tar 

wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
tar xvf VOCtest_06-Nov-2007.tar && rm VOCtest_06-Nov-2007.tar

wget host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar xvf VOCtrainval_11-May-2012.tar && rm VOCtrainval_11-May-2012.tar 
```

Data structure should look like:
```
data/VOCdevkit
    VOC{2007,2012}/
        Annotations/
        ImageSets/
        JPEGImages/
        ...
```

Move the dataset split files to the correct path:
```bash
mv  VOCsplits/VOC2007/*  VOCdevkit/VOC2007/ImageSets/Main/
mv  VOCsplits/VOC2012/*  VOCdevkit/VOC2012/ImageSets/Main/  
```


## COCO

Download data from official website:
```bash
mkdir coco

# download images
cd coco
mkdir images

wget images.cocodataset.org/zips/train2014.zip
unzip train2014.zip && rm train2014.zip
mv train2014 images/

wget images.cocodataset.org/zips/val2014.zip
unzip val2014.zip && rm val2014.zip
mv val2014 images/

# download annaotations
cd ..
wget images.cocodataset.org/annotations/annotations_trainval2014.zip
unzip annotations_trainval2014.zip && rm annotations_trainval2014.zip
```

Data structure should look like:
```
data/coco
    annotations/
    images/
        train2014/
        val2014/
```



## Custom Dataset

To experiment with your own dataset, first save the dataset or create a soft link in the folder ```ProjectRootDir/data```
Moreover, you should have a **csv-format** file containing all the annotations such as:
```
set,im_path,cls,difficult,left,upper,right,lower,height,width
train,ImagePathInDataset,cat,False,x1,y1,x2,y2,H,W
train,ImagePathInDataset,dog,False,x1,y1,x2,y2,H,W
val,ImagePathInDataset,cat,False,x1,y1,x2,y2,H,W
val,ImagePathInDataset,dog,False,x1,y1,x2,y2,H,W
...
```

Then, look into [factory.py](https://github.com/YoungXIAO13/FewShotDetection/blob/4e8d0c4a0352133113b8438a6a5fa8195661c6c0/lib/datasets/factory.py#L21) to set up correctly the dataset root path and split names (train, val, etc).\
Also, set up the novel classes for your custom dataset in the [config.py](https://github.com/YoungXIAO13/FewShotDetection/blob/1a77fbd81fb2f319e517c70faf5b9d7eca6b0546/lib/model/utils/config.py#L27)

Once this is done, look further into \
[custom.py](https://github.com/YoungXIAO13/FewShotDetection/blob/master/lib/datasets/custom.py) for dataset creation;\
[custom_metadata.py](https://github.com/YoungXIAO13/FewShotDetection/blob/master/lib/datasets/custom_metadata.py) for few-shot class data creation;\
[custom_eval.py](https://github.com/YoungXIAO13/FewShotDetection/blob/master/lib/datasets/custom_eval.py) for evaluation.

Finally, you should set the correct values for custom dataset in [train.py](https://github.com/YoungXIAO13/FewShotDetection/blob/1a77fbd81fb2f319e517c70faf5b9d7eca6b0546/train.py#L198) and [test.py](https://github.com/YoungXIAO13/FewShotDetection/blob/1a77fbd81fb2f319e517c70faf5b9d7eca6b0546/test.py#L129)


