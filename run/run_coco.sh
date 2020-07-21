#!/usr/bin/env bash

gpu=0
base_dir="save_models/COCO"

# base class training
CUDA_VISIBLE_DEVICES=$gpu python train_metarcnn.py --dataset coco \
--epochs 21 --bs 4 --nw 8 \
--log_dir checkpoint --save_dir $base_dir \
--meta_type 0 --meta_train True --meta_loss True

# number of shots
for j in 10 30
do
# few-shot fine-tuning
CUDA_VISIBLE_DEVICES=$gpu python train_metarcnn.py --dataset coco \
--epochs 30 --bs 4 --nw 8 \
--log_dir checkpoint --save_dir $base_dir \
--r True --checksession 200 --checkepoch 20 \
--meta_type 0 --shots $j --phase 2 --meta_train True --meta_loss True

# testing on base and novel class
CUDA_VISIBLE_DEVICES=$gpu python test_metarcnn.py --dataset coco \
--load_dir $base_dir  --meta_type 0 \
--checksession $j --checkepoch 29 --shots $j \
--phase 2 --meta_test True --meta_loss True
done
