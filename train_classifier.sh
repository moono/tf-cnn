#!/usr/bin/env bash


python train_classifier.py \
    --network_module resnet.network_resnet \
    --network_name resnet83 \
    --dataset_name cifar10 \
    --batch_size 256 \
    --learning_rate 0.1 \
    --weight_decay 1e-4


python train_classifier.py \
    --network_module resnet.network_resnet \
    --network_name resnet83 \
    --dataset_name cifar100 \
    --batch_size 256 \
    --learning_rate 0.1 \
    --weight_decay 1e-4


python train_classifier.py \
    --network_module resnet.network_resnet \
    --network_name resnet83 \
    --dataset_name mnist \
    --batch_size 256 \
    --learning_rate 0.1 \
    --weight_decay 1e-4


python train_classifier.py \
    --network_module resnet.network_resnet \
    --network_name resnet83 \
    --dataset_name fashion_mnist \
    --batch_size 256 \
    --learning_rate 0.1 \
    --weight_decay 1e-4