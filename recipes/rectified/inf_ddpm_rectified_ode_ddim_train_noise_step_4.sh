#!/bin/bash

###########
# Here ode means no noise

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -o ... 'error in pipeline', -x 'print commands',
set -e
set -o pipefail

# Sample
echo "[sampling]"
step=4
output_dir=output/rectified/DDPM_CIFAR10_EPS_RECTIFIED_400000_ode_ddim_step_${step}_train_noise
train_noise_scp=/home/btang5/work/2025/pytorch-ddpm/sample_pairs/img_from_noise_pair/pretrained_ckpt/noise.scp
python sample_ddim.py \
    --flagfile logs/DDPM_CIFAR_10_EPS_RECTIFIED/flagfile.txt \
    --logdir logs/DDPM_CIFAR_10_EPS_RECTIFIED \
    --batch_size 32 \
    --sample_img_from_noise_pair \
    --sample_img_noise_pair_path "$output_dir" \
    --num_images 50000 \
    --num_procs 2 \
    --gpus "cuda:3,cuda:4" \
    --steps "$step" \
    --noise_scp "$train_noise_scp"\
    --use_training_noise 

echo "[inferencing]"
# Inferencsse
CUDA_VISIBLE_DEVICES="7" python eval.py \
    --eval_output_dir "$output_dir/images_npy"
