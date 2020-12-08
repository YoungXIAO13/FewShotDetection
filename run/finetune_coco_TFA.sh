#!/usr/bin/env bash

base_dir="save_models/COCO"

# number of shots
for j in 1 2 3 5 10 30
do
# few-shot fine-tuning
CUDA_VISIBLE_DEVICES=3 python train.py --dataset coco \
--epochs 30 --bs 4 --nw 8 --log_dir checkpoint --save_dir $base_dir \
--r True --checksession 200 --checkepoch 20 --meta_type 0 --shots $j --phase 2 \
--meta_train True --meta_loss True --TFA True

CUDA_VISIBLE_DEVICES=3 python test.py --dataset coco \
--load_dir $base_dir  --meta_type 0 \
--checksession $j --checkepoch 29 --shots $j \
--phase 2 --meta_test True --meta_loss True
done