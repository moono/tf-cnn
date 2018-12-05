#!/usr/bin/env bash

python train_classifier_mirrored.py \
    --model_dir='/model_dir' \
    --save_name='multi-gpu-o1-b256-lrx1' \
    --epochs=20 \
    --batch_size_each=256 \
    --learning_rate=0.1 \
    --num_gpus=2 \
    --multi_gpu_option=1

python train_classifier_mirrored.py \
    --model_dir='/model_dir' \
    --save_name='multi-gpu-o2-b256-lrx1' \
    --epochs=20 \
    --batch_size_each=256 \
    --learning_rate=0.1 \
    --num_gpus=2 \
    --multi_gpu_option=2

python train_classifier_mirrored.py \
    --model_dir='/model_dir' \
    --save_name='multi-gpu-o3-b256-lrx1' \
    --epochs=20 \
    --batch_size_each=256 \
    --learning_rate=0.1 \
    --num_gpus=2 \
    --multi_gpu_option=3

python train_classifier_mirrored.py \
    --model_dir='/model_dir' \
    --save_name='multi-gpu-o2-b256-lrx2' \
    --epochs=20 \
    --batch_size_each=256 \
    --learning_rate=0.2 \
    --num_gpus=2 \
    --multi_gpu_option=2

python train_classifier_mirrored.py \
    --model_dir='/model_dir' \
    --save_name='multi-gpu-o2-b512-lrx2' \
    --epochs=20 \
    --batch_size_each=512 \
    --learning_rate=0.2 \
    --num_gpus=2 \
    --multi_gpu_option=2

python train_classifier_mirrored.py \
    --model_dir='/model_dir' \
    --save_name='single-gpu-b256-lrx1' \
    --epochs=20 \
    --batch_size_each=256 \
    --learning_rate=0.1 \
    --num_gpus=1

python train_classifier_mirrored.py \
    --model_dir='/model_dir' \
    --save_name='single-gpu-b512-lrx1' \
    --epochs=20 \
    --batch_size_each=512 \
    --learning_rate=0.1 \
    --num_gpus=1

python train_classifier_mirrored.py \
    --model_dir='/model_dir' \
    --save_name='single-gpu-b256-lrx2' \
    --epochs=20 \
    --batch_size_each=256 \
    --learning_rate=0.2 \
    --num_gpus=1

python train_classifier_mirrored.py \
    --model_dir='/model_dir' \
    --save_name='single-gpu-b512-lrx2' \
    --epochs=20 \
    --batch_size_each=512 \
    --learning_rate=0.2 \
    --num_gpus=1