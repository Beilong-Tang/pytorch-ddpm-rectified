import torch.multiprocessing as mp
import os
import warnings
import numpy as np
import tqdm
from PIL import Image
import math

import torch
from absl import app, flags
from tensorboardX import SummaryWriter
from torchvision.datasets import CIFAR10
from torchvision.utils import make_grid, save_image
from torchvision import transforms
from tqdm import trange

from diffusion import GaussianDiffusionTrainer, DDIMSampler
from model import UNet
from score.both import get_inception_and_fid_score


FLAGS = flags.FLAGS
flags.DEFINE_bool('train', False, help='train from scratch')
flags.DEFINE_bool('eval', False, help='load ckpt.pt and evaluate FID and IS')
flags.DEFINE_integer('steps', 100, help = "DDIM steps")
flags.DEFINE_bool("use_training_noise", False, help="whether to use training noise")
# Sample related 
flags.DEFINE_bool('sample_img_from_noise_pair', False, help='load ckpt.pt and generate sample noise pairs')
flags.DEFINE_list("gpus", ["cuda:1", "cuda:2", "cuda3", "cuda:4"], help = "gpus for inference")
flags.DEFINE_integer("num_procs", default=4, help="Number of processes to run")
# UNet
flags.DEFINE_integer('ch', 128, help='base channel of UNet')
flags.DEFINE_multi_integer('ch_mult', [1, 2, 2, 2], help='channel multiplier')
flags.DEFINE_multi_integer('attn', [1], help='add attention to these levels')
flags.DEFINE_integer('num_res_blocks', 2, help='# resblock in each level')
flags.DEFINE_float('dropout', 0.1, help='dropout rate of resblock')
# Gaussian Diffusion
flags.DEFINE_float('beta_1', 1e-4, help='start beta value')
flags.DEFINE_float('beta_T', 0.02, help='end beta value')
flags.DEFINE_integer('T', 1000, help='total diffusion steps')
flags.DEFINE_enum('mean_type', 'epsilon', ['xprev', 'xstart', 'epsilon'], help='predict variable')
flags.DEFINE_enum('var_type', 'fixedlarge', ['fixedlarge', 'fixedsmall'], help='variance type')
# Training
flags.DEFINE_float('lr', 2e-4, help='target learning rate')
flags.DEFINE_float('grad_clip', 1., help="gradient norm clipping")
flags.DEFINE_integer('total_steps', 800000, help='total training steps')
flags.DEFINE_integer('img_size', 32, help='image size')
flags.DEFINE_integer('warmup', 5000, help='learning rate warmup')
flags.DEFINE_integer('batch_size', 128, help='batch size')
flags.DEFINE_integer('num_workers', 4, help='workers of Dataloader')
flags.DEFINE_float('ema_decay', 0.9999, help="ema decay rate")
flags.DEFINE_bool('parallel', False, help='multi gpu training')
# Logging & Sampling
flags.DEFINE_string('logdir', './logs/DDPM_CIFAR10_EPS', help='log directory')
flags.DEFINE_integer('sample_size', 64, "sampling size of images")
flags.DEFINE_integer('sample_step', 1000, help='frequency of sampling')
# Evaluation
flags.DEFINE_integer('save_step', 5000, help='frequency of saving checkpoints, 0 to disable during training')
flags.DEFINE_integer('eval_step', 0, help='frequency of evaluating model, 0 to disable during training')
flags.DEFINE_integer('num_images', 50000, help='the number of generated images for evaluation')
flags.DEFINE_bool('fid_use_torch', False, help='calculate IS and FID on gpu')
flags.DEFINE_string('fid_cache', './stats/cifar10.train.npz', help='FID cache')
# Sample_img_noise_pair
flags.DEFINE_string("sample_img_noise_pair_path", default=None, help="output sample pair path")
# For compability
flags.DEFINE_string("noise_scp", default = None, help = "path to noise scp")
flags.DEFINE_string("img_scp", default = None, help = "path to noise scp")

from argparse import Namespace

class AttrDict(Namespace):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def __getattribute__(self, name: str):
        try:
            return super().__getattribute__(name)
        except AttributeError:
            return None
    def __getitem__(self, key):
        return self.__getattribute__(key)

def read_scp(scp_path):
    res = []
    with open(scp_path, "r") as f:
        for l in f.readlines():
            res.append(l.replace("\n","").split(" ")[-1])
    return res

def sample_img_from_noise_pair(rank, config):
    def evaluate_img_noise_pair(num_images, sampler, model, device, config,  rank = 0):
        model.eval()
        ct = 0
        batch_size = config.batch_size // config.num_procs

        noise_scp:str = config.noise_scp
        use_noise:bool = config.use_training_noise
        if use_noise:
            noise_paths = read_scp(noise_scp)
            noise_paths = noise_paths[rank::config.num_procs]
            print(f"rank {rank} got noises {len(noise_paths)} from noise scp")
            num_images = len(noise_paths)
        
        with torch.no_grad():
            desc = f"generating image and noise pairs of num {num_images} on rank {rank}"
            if rank == 0:
                pbar = trange(0, num_images, batch_size, desc=desc, dynamic_ncols=True)
            else:
                pbar = range(0, num_images, batch_size)
            for i in pbar:
                batch_size = min(batch_size, num_images - i)
                if use_noise:
                    noises = noise_paths[i:i+batch_size]
                    x_T = []
                    for _i in noises:
                        _n = torch.from_numpy(np.load(_i))# [3, H, W]
                        x_T.append(_n)
                    x_T = torch.stack(x_T, dim = 0) # [B, 3, H, W]
                else:
                    x_T = torch.randn((batch_size, 3, config.img_size, config.img_size))
                batch_images = sampler(x_T.to(device), steps = config.steps).cpu()
                batch_images = (batch_images + 1) / 2
                images = batch_images.numpy() # [B, 3, H, W]
                noises = x_T.numpy() # [B, 3, H, W]
                # save images and noises
                for _img, _noise in tqdm.tqdm(zip(images, noises), disable=True):
                    np.save(os.path.join(config.sample_img_noise_pair_path, "images_npy", f"r_{rank:02d}_{ct:06d}.npy"), _img)
                    np.save(os.path.join(config.sample_img_noise_pair_path, "noises", f"r_{rank:02d}_{ct:06d}.npy"), _noise)
                    _img = (_img * 255).clip(0, 255).astype(np.uint8)
                    _img = np.transpose(_img, (1, 2, 0))
                    Image.fromarray(_img).save(os.path.join(config.sample_img_noise_pair_path, "images", f"r_{rank:02d}_{ct:06d}.png"))
                    ct+=1

    config = AttrDict(**config)
    assert config.sample_img_noise_pair_path is not None
    os.makedirs(os.path.join(config.sample_img_noise_pair_path, "images_npy"), exist_ok=True)
    os.makedirs(os.path.join(config.sample_img_noise_pair_path, "images"), exist_ok=True)
    os.makedirs(os.path.join(config.sample_img_noise_pair_path, "noises"), exist_ok=True)

    device = config.gpus[rank % len(config.gpus)]
    num_imgs = math.ceil(config.num_images / config.num_procs)

    # model setup
    model = UNet(
        T=config.T, ch=config.ch, ch_mult=config.ch_mult, attn=config.attn,
        num_res_blocks=config.num_res_blocks, dropout=config.dropout)
    sampler = DDIMSampler(
        model, [config.beta_1, config.beta_T], config.T).to(device)
    # load model and evaluate
    ckpt = torch.load(os.path.join(config.logdir, 'ckpt.pt'), map_location='cpu')
    model.load_state_dict(ckpt['ema_model'])



    evaluate_img_noise_pair(num_imgs, sampler, model, device, config, rank = rank) # [N, 3, H, W], # [N, 3, H, W]

def main(argv):
    # suppress annoying inception_v3 initialization warning
    warnings.simplefilter(action='ignore', category=FutureWarning)
    print(f"gpus: {FLAGS.gpus}")
    if FLAGS.sample_img_from_noise_pair:
        if FLAGS.num_procs == 1:
            sample_img_from_noise_pair(0)
        else:
            print("Running DDP")
            mp.spawn(sample_img_from_noise_pair, args=(FLAGS.flag_values_dict(),), nprocs=FLAGS.num_procs, join = True)
        print("Done...")

if __name__ == '__main__':
    app.run(main)