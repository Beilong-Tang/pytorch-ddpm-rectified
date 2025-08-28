#!/bin/bash

# # DDPM
# Sample
python sample.py \
    --flagfile ./pretrained_ckpt/DDPM_CIFAR10_EPS/flagfile.txt \
    --logdir ./pretrained_ckpt/DDPM_CIFAR10_EPS \
    --batch_size 128 \
    --sample_img_from_noise_pair \
    --sample_img_noise_pair_path ./sample_pairs/img_from_noise_pair/pretrained_ckpt \
    --num_images 50000 \
    --num_procs 8 \
    --gpus cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7

# Evaluate

