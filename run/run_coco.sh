#!/usr/bin/env bash

gpu=0
base_dir="save_models/COCO"

CUDA_VISIBLE_DEVICES=$gpu python train_metarcnn.py --dataset coco \
--epochs 21 --bs 4 --nw 8 \
--log_dir checkpoint --save_dir $base_dir \
--meta_type 0 --meta_train True --meta_loss True

for i in 1
do

mkdir ${base_dir}_${i}
cp ${base_dir}/*_20.pth  ${base_dir}_${i}/

for j in 10 30
do
CUDA_VISIBLE_DEVICES=$gpu python train_metarcnn.py --dataset coco \
--epochs 30 --bs 4 --nw 8 \
--log_dir checkpoint --save_dir ${base_dir}_${i} \
--r True --checksession 200 --checkepoch 20 \
--meta_type 0 --shots $j --phase 2 --meta_train True --meta_loss True

CUDA_VISIBLE_DEVICES=$gpu python test_metarcnn.py --dataset coco \
--load_dir ${base_dir}_${i}  --meta_type 0 \
--checksession $j --checkepoch 29 --shots $j \
--phase 2 --meta_test True --meta_loss True
done

for j in 20 21 22 23 24 25 26 27 28
do
rm ${base_dir}_${i}/*_${j}.pth
done

done
