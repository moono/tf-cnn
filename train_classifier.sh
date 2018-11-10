#!/usr/bin/env bash


python train_classifier.py \
    --network_module resnet.network_resnet \
    --network_name resnet29 \
    --dataset_name mnist \
    --batch_size 128 \
    --learning_rate 0.1


python train_classifier.py \
    --network_module resnet.network_resnet \
    --network_name resnet29 \
    --dataset_name fashion_mnist \
    --batch_size 128 \
    --learning_rate 0.1


python train_classifier.py \
    --network_module resnet.network_resnet \
    --network_name resnet29 \
    --dataset_name cifar10 \
    --batch_size 128 \
    --learning_rate 0.1


python train_classifier.py \
    --network_module resnet.network_resnet \
    --network_name resnet29 \
    --dataset_name cifar100 \
    --batch_size 128 \
    --learning_rate 0.1