import math
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

def log_cosh_loss(prediction, target):
    loss = torch.log(torch.cosh(prediction - target))
    return loss

class H5Dataset(Dataset):
    def __init__(self, h5_path, augmentation=False):
        super().__init__()
        self.h5_path = h5_path
        self.augmentation = augmentation
        
        self.hf = h5py.File(self.h5_path, 'r')
        self.rgb_data = self.hf['rgb']
        self.chm_data = self.hf['chm']
        self.num_patches = self.rgb_data.shape[0]

        self.normalize_rgb = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        self.chm_min = 0
        self.chm_max = 60
        print(f"Estadísticas del CHM: Min={self.chm_min}, Max={self.chm_max}")

    def __len__(self):
        return self.num_patches

    def __getitem__(self, index): 
        rgb_image = self.rgb_data[index].astype(np.float32)
        chm_image = self.chm_data[index].astype(np.float32)

        chm_image[chm_image < 0] = np.nan
        chm_image[chm_image > self.chm_max] = self.chm_max

        valid_chm_pixels = ~np.isnan(chm_image)
        non_black_rgb_mask = np.any(rgb_image > 0, axis=-1)
        final_mask_np = np.logical_and(valid_chm_pixels, non_black_rgb_mask)
        
        chm_image[np.isnan(chm_image)] = 0
        
        rgb_tensor = torch.from_numpy(rgb_image).permute(2, 0, 1)
        chm_tensor = torch.from_numpy(chm_image).unsqueeze(0)
        final_mask_tensor = torch.from_numpy(final_mask_np).bool()
        
        if self.augmentation:
            if random.random() > 0.5:
                rgb_tensor = F_transforms.hflip(rgb_tensor)
                chm_tensor = F_transforms.hflip(chm_tensor)
                final_mask_tensor = F_transforms.hflip(final_mask_tensor.unsqueeze(0)).squeeze(0)
            if random.random() > 0.5:
                rgb_tensor = F_transforms.vflip(rgb_tensor)
                chm_tensor = F_transforms.vflip(chm_tensor)
                final_mask_tensor = F_transforms.vflip(final_mask_tensor.unsqueeze(0)).squeeze(0)
            if random.random() > 0.5:
                rotation = random.choice([0, 90, 180, 270])
                if rotation > 0:
                    rgb_tensor = F_transforms.rotate(rgb_tensor, rotation)
                    chm_tensor = F_transforms.rotate(chm_tensor, rotation)
                    final_mask_tensor = F_transforms.rotate(final_mask_tensor.unsqueeze(0), rotation, interpolation=F_transforms.InterpolationMode.NEAREST).squeeze(0)
        
        rgb_tensor = self.normalize_rgb(rgb_tensor)
        chm_tensor = (chm_tensor - self.chm_min) / (self.chm_max - self.chm_min)
        
        return rgb_tensor, chm_tensor, final_mask_tensor

    def close(self):
        self.hf.close()


def create_tiles(image_rgb, image_chm, tile_size):
    """
    Genera mosaicos con padding de ceros y los filtra.
    """
    tiles = []
    h, w, c = image_rgb.shape
    
    pad_h = (tile_size - (h % tile_size)) % tile_size
    pad_w = (tile_size - (w % tile_size)) % tile_size

    image_rgb_padded = np.pad(
        image_rgb,
        ((0, pad_h), (0, pad_w), (0, 0)),
        mode='constant',
        constant_values=np.nan
    )
    image_chm_padded = np.pad(
        image_chm,
        ((0, pad_h), (0, pad_w)),
        mode='constant',
        constant_values=np.nan
    )

    h_padded, w_padded, _ = image_rgb_padded.shape
    for y in range(0, h_padded, tile_size):
        for x in range(0, w_padded, tile_size):
            tile_rgb = image_rgb_padded[y:y+tile_size, x:x+tile_size, :]
            tile_chm = image_chm_padded[y:y+tile_size, x:x+tile_size]
            tiles.append((tile_rgb, tile_chm))
    return tiles
