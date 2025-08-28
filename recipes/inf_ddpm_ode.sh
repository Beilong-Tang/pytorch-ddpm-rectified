#!/bin/bash

###########
# HERE ODE Means no noise

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -o ... 'error in pipeline', -x 'print commands',
set -e
set -o pipefail


# Sample
echo "[sampling]"
output_dir=output/DDPM_CIFAR10_EPS_400000_ode
python sample.py \
    --flagfile logs/DDPM_CIFAR10_EPS/flagfile.txt \
    --logdir logs/DDPM_CIFAR10_EPS \
    --batch_size 128 \
    --sample_img_from_noise_pair \
    --sample_img_noise_pair_path "$output_dir" \
    --num_images 50000 \
    --num_procs 8 \
    --gpus cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7 \
    --no_noise

echo "[inferencing]"
# Inference
python eval.py \
    --eval_output_dir "$output_dir/images_npy"
