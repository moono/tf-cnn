#!/usr/bin/env bash

python train_classifier_mirrored.py \
    --model_dir='/model_dir' \
    --save_name='single-gpu' \
    --num_gpus=1

python train_classifier_mirrored.py \
    --model_dir='/model_dir' \
    --save_name='multi-gpu-o1' \
    --num_gpus=2 \
    --multi_gpu_option=1

python train_classifier_mirrored.py \
    --model_dir='/model_dir' \
    --save_name='multi-gpu-o2' \
    --num_gpus=2 \
    --multi_gpu_option=2

python train_classifier_mirrored.py \
    --model_dir='/model_dir' \
    --save_name='multi-gpu-o3' \
    --num_gpus=2 \
    --multi_gpu_option=3
