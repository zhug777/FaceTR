import os
import math
import torch
import random
import numpy as np
from core.heatmap import CenterLabelHeatMap, CenterGaussianHeatMap
from core.transforms import get_affine_transform, affine_transform
import glob
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from tqdm import tqdm

segments = ['background', 'skin', 'left_eyebrow', 'right_eyebrow', 'left_eye', 'right_eye',
            'nose', 'upper_lip', 'inner_mouth', 'lower_lip', 'hair']

class FaceDataset(Dataset):
    def __init__(self, root, image_size, is_train=True, scale_factor=0.3, rotation_factor=45):
        super(FaceDataset, self).__init__()
        if is_train:
            train_img_path = os.path.join(root, "train/images/*")
            train_mask_path = os.path.join(root, "train/labels/*")
            train_kpt_path = os.path.join(root, "train/landmarks/*")
            self.img_url = sorted(glob.glob(train_img_path))
            self.mask_url = sorted(glob.glob(train_mask_path))
            self.kpt_url = sorted(glob.glob(train_kpt_path))
        else:
            val_img_path = os.path.join(root, "val/images/*")
            val_mask_path = os.path.join(root, "val/labels/*")
            val_kpt_path = os.path.join(root, "val/landmarks/*")
            self.img_url = sorted(glob.glob(val_img_path))
            self.mask_url = sorted(glob.glob(val_mask_path))
            self.kpt_url = sorted(glob.glob(val_kpt_path))
        self.image_size = np.array(image_size)
        self.scale_factor = scale_factor
        self.rotation_factor = rotation_factor
        self.is_train = is_train

    def __getitem__(self, idx):
        img = cv2.imread(
            self.img_url[idx], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_url[idx], cv2.IMREAD_UNCHANGED | cv2.IMREAD_IGNORE_ORIENTATION)
        kpts = []
        with open(self.kpt_url[idx], 'r') as f:
            line = f.readline()
            if int(line) == 106:
                while line:
                    line = f.readline()
                    kpt = line.split('\n')[0].split(' ')
                    if len(kpt) > 1:
                        for dim in range(len(kpt)):
                            kpt[dim] = int(kpt[dim].split('.')[0])
                        kpts.append(kpt)
        
        center = np.zeros((2), dtype=np.float32)
        center[0] = (img.shape[1] - 1) * 0.5
        center[1] = (img.shape[0] - 1) * 0.5
        scale = np.array([img.shape[1] * 1.0 / 200, img.shape[0] * 1.0 / 200], dtype=np.float32)* 0.7
        rot = 0
        if self.is_train == True:
            scale = scale * np.clip(np.random.randn() * self.scale_factor + 1, 1 - self.scale_factor, 1 + self.scale_factor)
            rot = np.clip(np.random.randn()*self.rotation_factor, -self.rotation_factor*2, self.rotation_factor*2) if random.random() <= 0.6 else 0
            if random.random() <= 0.5: # random hirizontal flip
                cv2.flip(img, 1, img)
                cv2.flip(mask, 1, mask)
                for i, kpt in enumerate(kpts):
                    kpts[i][0] = int(img.shape[1]) - 1 - kpts[i][0]
        trans = get_affine_transform(center, scale, rot, self.image_size)

        img = cv2.warpAffine(img, trans, (int(self.image_size[0]), int(self.image_size[1])), flags=cv2.INTER_NEAREST)
        #img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_NEAREST)
        img_array = np.array(img, dtype=np.float32) / 255
        img_array = img_array.transpose(2, 0, 1)
        img = torch.tensor(img_array)

        mask = cv2.warpAffine(mask, trans, (int(self.image_size[0]), int(self.image_size[1])), flags=cv2.INTER_NEAREST)
        #mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        mask_array = np.array(mask, dtype=np.float32)
        edgemap = self._generate_edgemap(mask_array)
        mask = torch.tensor(mask_array)

        trans_kpts = []
        for i in range(len(kpts)):
            trans_kpt = list(affine_transform(kpts[i], trans))
            for dim in range(len(trans_kpt)):
                trans_kpt[dim] = int(trans_kpt[dim])    
            trans_kpts.append(trans_kpt)

        heatmaps = self._generate_heatmaps(trans_kpts)
        meta = {
            'imagename': self.img_url[idx],
            'maskname': self.mask_url[idx],
            'kptname': self.kpt_url[idx]
        }
        '''
        for i, item in enumerate(segments):
            meta[item] = torch.tensor(mask_array == i).int()
        '''
        return img, mask, heatmaps, edgemap, meta

    def __len__(self):
        return len(self.img_url)

    def _generate_heatmaps(self, kpts):
        heatmaps = np.zeros(shape=(len(kpts), self.image_size[0], self.image_size[1]))
        '''
        for i, kpt in enumerate(kpts):
            heatmap = CenterLabelHeatMap(self.image_size[0], self.image_size[1], kpt[0], kpt[1], sigma=10)
            heatmaps[i] = heatmap
        heatmaps = torch.tensor(heatmaps)
        '''
        return heatmaps
    
    def _generate_edgemap(self, mask):
        edgemap = np.zeros_like(mask)
        '''
        for h in range(self.image_size[1])[:-1]:
            for w in range(self.image_size[0])[1:]:
                if mask[h][w] != mask[h+1][w] or mask[h][w] != mask[h][w-1]:
                    edgemap[h][w] = 1
        for h in range(self.image_size[1])[:-1]:
            if mask[h][0] != mask[h+1][0]:
                edgemap[h][0] = 1
        for w in range(self.image_size[0])[1:]:
            if mask[self.image_size[1]-1][w] != mask[self.image_size[1]-1][w-1]:
                edgemap[self.image_size[1]-1][w] = 1
        edgemap = torch.tensor(edgemap)
        '''
        return edgemap