#!/usr/bin/env bash

base_dir="save_models/VOC_first"

# number of shots
for j in 1 2 3 5 10
do
# testing on base and novel class
python test.py --dataset pascal_voc_0712 \
--load_dir $base_dir  --meta_type 1 \
--checksession $j --checkepoch 29 --shots $j \
--phase 2 --meta_test True --meta_loss True
done
