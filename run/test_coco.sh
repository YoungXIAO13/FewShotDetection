#!/usr/bin/env bash

base_dir="save_models/COCO"

# number of shots
for j in 10 30
do
# testing on base and novel class
python test.py --dataset coco \
--load_dir $base_dir  --meta_type 0 \
--checksession $j --checkepoch 29 --shots $j \
--phase 2 --meta_test True --meta_loss True
done