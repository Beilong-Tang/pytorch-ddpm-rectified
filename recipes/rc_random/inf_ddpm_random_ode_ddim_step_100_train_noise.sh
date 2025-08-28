#!/bin/bash

###########
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -o ... 'error in pipeline', -x 'print commands',
set -e
set -o pipefail

# Sample
echo "[sampling]"
step=100
output_dir=output/DDPM_CIFAR10_EPS_RANDOM_400000_ode_ddim_step_${step}_train_noise
train_noise_scp=/home/btang5/work/2025/pytorch-ddpm/sample_pairs/random_noise_gaussian/pretrained_ckpt/noise.scp
python sample_ddim.py \
    --flagfile logs/DDPM_CIFAR_10_EPS_RANDOM/flagfile.txt \
    --logdir logs/DDPM_CIFAR_10_EPS_RANDOM \
    --batch_size 80 \
    --sample_img_from_noise_pair \
    --sample_img_noise_pair_path "$output_dir" \
    --num_images 50000 \
    --num_procs 5 \
    --gpus cuda:3,cuda:4,cuda:5,cuda:6,cuda:7 \
    --steps "$step" \
    --noise_scp "$train_noise_scp"\
    --use_training_noise 

echo "[inferencing]"
# Inference
CUDA_VISIBLE_DEVICES="7" python eval.py \
    --eval_output_dir "$output_dir/images_npy"
