# Load dataset
from torch.utils.data.dataset import Dataset
import numpy as np
import torch
from torchvision import transforms

def read_scp(scp_path):
    res = {}
    with open(scp_path, "r") as f:
        for l in f.readlines():
            l = l.replace("\n", "")
            uttid, val = l.split(" ")
            res[uttid] = val
    return res

class ImageNoisePairDataset(Dataset):

    def __init__(self, img_scp, noise_scp):
        
        uttid_img = read_scp(img_scp)
        uttid_noise = read_scp(noise_scp)

        transform = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        self.uttid_img = uttid_img
        self.uttid_noise = uttid_noise  
        self.transform = transform

    def __len__(self):
        return len(self.uttid_img)

    def __getitem__(self, index):
        _key = list(self.uttid_img.keys())[index]
        _img_path = self.uttid_img[_key]
        _noise_path = self.uttid_noise[_key]
        # [3, H, W]
        img = torch.from_numpy(np.load(_img_path)) # [C, H, W]
        noise = torch.from_numpy(np.load(_noise_path)) # [C, H, W]

        img = self.transform(img)
        return img, noise
