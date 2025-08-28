#!/bin/bash

###########
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -o ... 'error in pipeline', -x 'print commands',
set -e
set -o pipefail


# Sample
echo "[sampling]"
python sample.py \
    --flagfile logs/DDPM_CIFAR10_EPS/flagfile.txt \
    --logdir logs/DDPM_CIFAR10_EPS \
    --batch_size 128 \
    --sample_img_from_noise_pair \
    --sample_img_noise_pair_path /home/btang5/work/2025/pytorch-ddpm/output/DDPM_CIFAR10_EPS_step_400000 \
    --num_images 50000 \
    --num_procs 8 \
    --gpus cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7

echo "[inferencing]"
# Inference
python eval.py \
    --eval_output_dir /home/btang5/work/2025/pytorch-ddpm/output/DDPM_CIFAR10_EPS_step_400000/images_npy
