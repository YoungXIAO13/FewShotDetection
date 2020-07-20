#!/usr/bin/env bash

base_dir="models/TDID_cos/VOC_second"
gpu=3

#CUDA_VISIBLE_DEVICES=0 python train_metarcnn.py --dataset pascal_voc_0712 \
#--epochs 21 --bs 4 --nw 8 \
#--log_dir checkpoint --save_dir $base_dir \
#--meta_type 2 --meta_train True --meta_loss True \
#--TDID True --cat True --cos

for i in 1 2 3 4 5 6 7 8 9 10
do

mkdir ${base_dir}_${i}
cp ${base_dir}/*_200_20_*  ${base_dir}_${i}/

for j in 1 2 5
do
CUDA_VISIBLE_DEVICES=$gpu python train_metarcnn.py --dataset pascal_voc_0712 \
--epochs 30 --bs 4 --nw 8 \
--log_dir checkpoint --save_dir ${base_dir}_${i} \
--r True --checksession 200 --checkepoch 20 --checkpoint 3063 \
--meta_type 2 --shots $j --phase 2 --meta_train True --meta_loss True \
--TDID True --cat True --cos
done

for j in 21 22 23 24 25 26 27 28
do
rm ${base_dir}_${i}/*_${j}_*
done

rm ${base_dir}_${i}/*_200_20_*

done
