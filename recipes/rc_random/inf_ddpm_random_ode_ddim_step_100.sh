#!/bin/bash

###########
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -o ... 'error in pipeline', -x 'print commands',
set -e
set -o pipefail

# Sample
echo "[sampling]"
output_dir=output/DDPM_CIFAR10_EPS_RANDOM_400000_ode_ddim_step_100
python sample.py \
    --flagfile logs/DDPM_CIFAR_10_EPS_RANDOM/flagfile.txt \
    --logdir logs/DDPM_CIFAR_10_EPS_RANDOM \
    --batch_size 128 \
    --sample_img_from_noise_pair \
    --sample_img_noise_pair_path "$output_dir" \
    --num_images 50000 \
    --num_procs 5 \
    --gpus cuda:3,cuda:4,cuda:5,cuda:6,cuda:7

echo "[inferencing]s"
# Inference
CUDA_VISIBLE_DEVICES="7" python eval.py \
    --eval_output_dir "$output_dir/images_npy"
