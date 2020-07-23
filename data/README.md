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
