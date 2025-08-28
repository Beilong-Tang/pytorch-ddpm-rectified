#!/bin/bash

###########
# Here ode means no noise

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -o ... 'error in pipeline', -x 'print commands',
set -e
set -o pipefail

# Sample
echo "[sampling]"
step=100
output_dir=output/DDPM_CIFAR10_EPS_RECTIFIED_400000_ode_ddim_step_$step
python sample_ddim.py \
    --flagfile logs/DDPM_CIFAR_10_EPS_RECTIFIED/flagfile.txt \
    --logdir logs/DDPM_CIFAR_10_EPS_RECTIFIED \
    --batch_size 48 \
    --sample_img_from_noise_pair \
    --sample_img_noise_pair_path "$output_dir" \
    --num_images 50000 \
    --num_procs 2 \
    --gpus cuda:3, cuda:4 \
    --steps "$step" ## DDIM sampling steps

echo "[inferencing]"
# Inferencsse
python eval.py \
    --eval_output_dir "$output_dir/images_npy"
