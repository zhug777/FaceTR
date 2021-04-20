from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
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
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    checkpoint_file = os.path.join(out_path, cfg.TRAIN.CHECKPOINT)
    out_channels = cfg.MODEL.NUM_SEGMENTS
    weights = [1, 1, 1.2, 1.2, 1.2, 1.2, 1, 1.2, 2, 1.2, 1]
    train_loss = 0.0
    test_loss = 0.0
    train_losses = []
    test_losses = []
    all_prec = [0] * 10
    all_rec = [0] * 10
    precisions = []
    recalls = []
    lrs = []
    best_state_dict = []
    best_epoch = 0
    best_perf = 100000.0
    best_model = False
    begin_epoch = cfg.TRAIN.BEGIN_EPOCH # default value is 1

    # Data loading code
    root = os.path.join(os.getcwd(), "data", cfg.DATASET.ROOT, cfg.DATASET.ROOT)
    #root = "E:\\datasets\\LaPa\\LaPa"
    train_dataset = eval(cfg.DATASET.DATASET)(
        root=root, image_size=cfg.MODEL.IMAGE_SIZE, is_train=True, aug=False,
        scale_factor=cfg.DATASET.SCALE_FACTOR, rotation_factor=cfg.DATASET.ROT_FACTOR)
    val_dataset = eval(cfg.DATASET.DATASET)(
        root=root, image_size=cfg.MODEL.IMAGE_SIZE, is_train=False, aug=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=cfg.TRAIN.SHUFFLE,
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

    # define loss function (criterion) and optimizer
    loss_func = get_loss(loss=cfg.TRAIN.LOSS, alpha=None)
    optimizer = get_optimizer(cfg, model)

    if cfg.AUTO_RESUME and os.path.exists(checkpoint_file):
        checkpoint = torch.load(checkpoint_file, map_location=device)       
        model.load_state_dict(checkpoint['state_dict'])
        begin_epoch += checkpoint['cur_epoch']
        #if 'best_perf' in checkpoint.keys():
        best_state_dict = checkpoint['best_state_dict']
        best_epoch = checkpoint['best_epoch']
        best_perf = checkpoint['best_perf']
        optimizer.load_state_dict(checkpoint['optimizer'])
        # 重载optimizer的参数时将所有的tensor都放到cuda上（加载时默认放在cpu上了）
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)
        train_losses.extend(checkpoint['trainloss'])
        test_losses.extend(checkpoint['testloss'])
        precisions.extend(checkpoint['precisions'])
        recalls.extend(checkpoint['recalls'])
        lrs.extend(checkpoint['lrs'])

    scheduler = get_scheduler(cfg, optimizer, begin_epoch)
    model.to(device)

    model.train()
    for epoch in range(begin_epoch, cfg.TRAIN.END_EPOCH + 1):
        print("current epoch is %d" % (epoch))
        print("current learning rate is %.9f" % (optimizer.param_groups[0]['lr']))
        lrs.append(optimizer.param_groups[0]['lr'])

        for img, mask, heatmaps, edgemap, meta in tqdm(train_loader, total=len(train_loader)):
            optimizer.zero_grad()  # reset gradient
            img = img.to(device)
            mask = mask.type(torch.LongTensor).to(device)

            # pred_img (batch, channel, W, H)
            pred_img = model(img)
            if out_channels == 1:
                pred_img = pred_img.squeeze(1) 

            loss = loss_func(pred_img, mask)
            train_loss += loss.item()
            loss.backward() 
            optimizer.step()  

        if scheduler != None:
            scheduler.step() # update learning rate
        print("train loss is " + str(train_loss))
        train_losses.append(train_loss)
        train_loss = 0.0
        # 同时记录测试loss
        model.eval()
        with torch.no_grad():
            for img, mask, heatmaps, edgemap, meta in tqdm(val_loader, total=len(val_loader)):
                test_img = img.to(device)
                test_mask = mask
                test_mask = test_mask.type(torch.LongTensor).to(device)
                pred_img = model(test_img)
                if out_channels == 1:
                    pred_img = pred_img.squeeze(1)
                loss = loss_func(pred_img, test_mask)
                test_loss += loss.item()

                mean, prec, rec, f1s, precisions, recalls = eval(cfg.TEST.TEST_FUNC)(pred_img, test_mask)
                for i in range(len(all_prec)):
                    all_prec[i] += precisions[i]
                    all_rec[i] += recalls[i]

            for i in range(len(all_prec)):
                all_prec[i] /= len(val_loader)
                all_rec[i] /= len(val_loader)
            print("test loss is " + str(test_loss))
            print("precisions are {}".format(all_prec))
            print("recalls are {}".format(all_rec))
            total_prec = 0.0
            total_rec = 0.0
            test_losses.append(test_loss)
            precisions.append(all_prec)
            recalls.append(all_rec)
            perf_indicator = test_loss
            test_loss = 0.0
        model.train()

        if perf_indicator < best_perf:
            best_perf = perf_indicator
            best_model = True
        else:
            best_model = False

        if best_model:
            checkpoint = {
                'state_dict': model.state_dict(),
                'cur_epoch': epoch,
                'best_state_dict': best_state_dict,
                'best_epoch': epoch,
                'best_perf': perf_indicator,
                'optimizer': optimizer.state_dict(),
                'trainloss': train_losses,
                'testloss': test_losses,
                'precisions': precisions,
                'recalls': recalls,
                'lrs': lrs
            }
        else:
            checkpoint = {
                'state_dict': model.state_dict(),
                'cur_epoch': epoch,
                'best_state_dict': model.state_dict(),
                'best_epoch': best_epoch,
                'best_perf': best_perf,
                'optimizer': optimizer.state_dict(),
                'trainloss': train_losses,
                'testloss': test_losses,
                'precisions': precisions,
                'recalls': recalls,
                'lrs': lrs
            }
        torch.save(checkpoint, os.path.join(out_path, cfg.TRAIN.CHECKPOINT))
        

if __name__ == '__main__':
    main()