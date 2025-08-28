#!/bin/bash
## Train from a pretrained DDPM (Rectified)
CUDA_VISIBLE_DEVICES=1 python main.py --train \
    --flagfile ./config/CIFAR10.txt \
    --logdir ./logs/DDPM_CIFAR_10_EPS_RECTIFIED\
    --img_scp /home/btang5/work/2025/pytorch-ddpm/sample_pairs/img_from_noise_pair/pretrained_ckpt/image.scp\
    --noise_scp /home/btang5/work/2025/pytorch-ddpm/sample_pairs/img_from_noise_pair/pretrained_ckpt/noise.scp \
    --eval_step 2000 \
    --total_steps 400000 

## Train from a fixed noise but randomly select for each image
CUDA_VISIBLE_DEVICES=2 python main.py --train \
    --flagfile ./config/CIFAR10.txt \
    --logdir ./logs/DDPM_CIFAR_10_EPS_RANDOM\
    --img_scp /home/btang5/work/2025/pytorch-ddpm/sample_pairs/random_noise_gaussian/pretrained_ckpt/image.scp\
    --noise_scp /home/btang5/work/2025/pytorch-ddpm/sample_pairs/random_noise_gaussian/pretrained_ckpt/noise.scp \
    --eval_step 2000 \ 
    --total_steps 400000 
