import os
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import numpy as np
import random

import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset

def plot_data(img, mask, filename):
    img = img.numpy()
    mask = mask.numpy()

    im = plt.subplot(1, 2, 1)
    im.set_title('image')
    plt.imshow(img[0])
    im.set_xticks([])
    im.set_yticks([])

    ma = plt.subplot(1, 2, 2)
    ma.set_title('gd_mask')
    plt.imshow(mask[0])
    ma.set_xticks([])
    ma.set_yticks([])
    #plt.show()
    out_path = os.path.join(os.getcwd(), "data", "processed_data")
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    plt.savefig(os.path.join(out_path, filename[0].split('/')[-1]))


def plot_seg_pred(img, mask, pred, filename):
    img = img.numpy()
    mask = mask.numpy()
    pred = torch.argmax(pred, dim=1)
    pred = pred.int().cpu().numpy()

    im = plt.subplot(1, 3, 1)
    im.set_title('image')
    plt.imshow(img[0])
    im.set_xticks([])
    im.set_yticks([])

    ma = plt.subplot(1, 3, 2)
    ma.set_title('gd_mask')
    plt.imshow(mask[0])
    ma.set_xticks([])
    ma.set_yticks([])

    ma1 = plt.subplot(1, 3, 3)
    ma1.set_title('pred_mask')
    plt.imshow(pred[0])
    ma1.set_xticks([])
    ma1.set_yticks([])
    #plt.show()
    out_path = os.path.join(os.getcwd(), "data", "seg_predictions")
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    plt.savefig(os.path.join(out_path, filename[0].split('/')[-1]))


def plot_kpt_pred(img, pred, kpts, filename):
    kpts = np.array(kpts)
    pred = np.array(pred)
    img = img.numpy()
    
    img2 = img[0].copy()
    img3 = img[0].copy()
    for point in kpts:
        for dim in range(len(point)):
            point[dim] = int(point[dim])
        cv2.circle(img2, tuple(point), radius=1, color=(0, 0, 255), thickness=4)
    kpt = plt.subplot(1, 2, 1)
    kpt.set_title('gd_kpts')
    plt.imshow(img2)
    kpt.set_xticks([])
    kpt.set_yticks([])

    for point in pred[0]:
        for dim in range(len(point)):
            point[dim] = int(point[dim])
        cv2.circle(img3, tuple(point), radius=1, color=(0, 0, 255), thickness=4)
    ma = plt.subplot(1, 2, 2)
    ma.set_title('pred_kpts')
    plt.imshow(img3)
    ma.set_xticks([])
    ma.set_yticks([])
    #plt.show()
    out_path = os.path.join(os.getcwd(), "data", "kpt_predictions")
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    plt.savefig(os.path.join(out_path, filename[0].split('/')[-1]))