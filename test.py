from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import pprint
import shutil
import glob
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import math
import numpy as np
import random

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data.distributed
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models import *
from datasets import *
from config import cfg
from config import update_config
from core.heatmap import CenterLabelHeatMap, CenterGaussianHeatMap
from core.evamat import compute_meanf1, compute_mpa, compute_precision
from core.transforms import get_affine_transform, affine_transform
from core.loss import get_loss
from core.utils import get_optimizer, get_scheduler, seed_torch


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    # philly
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')
    args = parser.parse_args()

    return args


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parse_args()
    update_config(cfg, args)
    # cudnn related setting
    seed_torch()

    out_path = os.path.join(os.getcwd(), "checkpoints", cfg.MODEL.NAME)
    checkpoint_file = os.path.join(out_path, cfg.TRAIN.CHECKPOINT)
    out_channels = cfg.MODEL.NUM_SEGMENTS
    total = 0.0
    precs = 0.0
    recs = 0.0
    all_f1 = [0] * 10
    all_prec = [0] * 10
    all_rec = [0] * 10
    n = 0

    # Data loading code
    root = os.path.join(os.getcwd(), "data", cfg.DATASET.ROOT, cfg.DATASET.ROOT)
    #root = "E:\\datasets\\LaPa\\LaPa"
    train_dataset = eval(cfg.DATASET.DATASET)(
        root=root, image_size=cfg.MODEL.IMAGE_SIZE, is_train=True, aug=False)
    val_dataset = eval(cfg.DATASET.DATASET)(
        root=root, image_size=cfg.MODEL.IMAGE_SIZE, is_train=False, aug=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=cfg.TEST.SHUFFLE,
        num_workers=cfg.WORKERS, # num of subprocesses for data loading
        pin_memory=cfg.PIN_MEMORY # default value is False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=cfg.TEST.SHUFFLE,
        num_workers=cfg.WORKERS,  # num of subprocesses for data loading
        pin_memory=cfg.PIN_MEMORY  # default value is False
    )

    model = eval(cfg.MODEL.NAME)(cfg)
    #model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()

    assert os.path.exists(checkpoint_file), "No checkpoint for testing!"
    checkpoint = torch.load(checkpoint_file, map_location=device)
    print(checkpoint.keys())
    model.load_state_dict(checkpoint['state_dict'])
    #best_epoch = checkpoint['best_epoch']
    epoch = checkpoint['cur_epoch']

    model.to(device)

    model.eval()
    with torch.no_grad():
        print("current epoch is %d" % (epoch))
        #print("best epoch is %d" % (best_epoch))
        print("testing weights..")
        for img, mask, heatmaps, edgemap, meta in tqdm(val_loader, total=len(val_loader)):
            n += 1
            val_img = img.to(device)
            val_mask = mask.to(device)
            #pred_img = torch.sigmoid(model(val_img))  # [1, 1, 256, 256]
            pred_img = model(val_img)  # [1, 11, 256, 256]
            if out_channels == 1:
                pred_img = pred_img.squeeze(1)  # [1, 256, 256]
            mean, prec, rec, f1s, precisions, recalls = eval(cfg.TEST.TEST_FUNC)(pred_img, val_mask)
            #print(mean)
            total += mean
            precs += prec
            recs += rec
            for i in range(len(all_f1)):
                all_f1[i] += f1s[i]
                all_prec[i] += precisions[i]
                all_rec[i] += recalls[i]

        total = total / n
        precs = precs / n
        recs = recs / n
        for i in range(len(all_f1)):
            all_f1[i] /= n 
            all_f1[i] = float(all_f1[i].cpu())
            all_prec[i] /= n 
            all_prec[i] = float(all_prec[i].cpu())
            all_rec[i] /= n 
            all_rec[i] = float(all_rec[i].cpu())
        
        print("mean f1 is " + str(total))
        print("precision is " + str(precs))
        print("recall is " + str(recs))
        print("f1 scores are: ")
        for i in range(len(segments[1:])):
            print(segments[i+1], ": ", str(all_f1[i]))
        print("precisions are: ")
        for i in range(len(segments[1:])):
            print(segments[i+1], ": ", str(all_prec[i]))
        print("recalls are: ")
        for i in range(len(segments[1:])):
            print(segments[i+1], ": ", str(all_rec[i]))

if __name__ == '__main__':
    main()
