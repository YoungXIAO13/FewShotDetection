#!/usr/bin/env bash

base_dir="save_models/VOC_third"

# number of shots
for j in 1 2 3 5 10
do
# few-shot fine-tuning
python train.py --dataset pascal_voc_0712 \
--epochs 30 --bs 4 --nw 8 \
--log_dir checkpoint --save_dir $base_dir \
--r True --checksession 200 --checkepoch 20 \
--meta_type 3 --shots $j --phase 2 --meta_train True --meta_loss True
done
