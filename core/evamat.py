import os
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_meanf1(input, target):
    '''
    Calculate Mean F1 Score
    :param input: shape[batch, classes, h, w]
    :param target: shape[batch, h, w]
    :return mpa: Mean F1 Score
    '''
    eps = 0.0001
    length = input.size()[1] - 1  # background不计算在内，与原论文一致
    input = torch.argmax(input, dim=1)
    input = input.int()
    mean = torch.zeros(1).to(device)
    prec = torch.zeros(1).to(device)
    rec = torch.zeros(1).to(device)
    f1s = []
    precs = []
    recs = []
    for i in range(length):
        tp = torch.sum((input.view(-1) == i + 1).float() *
                       (target.view(-1) == i + 1).float()).to(device)
        precision = (tp + eps) / (torch.sum((input.view(-1)
                                             == i + 1).float()) + eps).to(device)
        recall = (tp + eps) / (torch.sum((target.view(-1)
                                          == i + 1).float()) + eps).to(device)
        f1 = (2 * precision * recall) / (precision + recall)
        f1s.append(f1)
        precs.append(precision)
        recs.append(recall)
        mean += f1
        prec += precision
        rec += recall
    mean /= torch.tensor([length], dtype=float).to(device)
    prec /= torch.tensor([length], dtype=float).to(device)
    rec /= torch.tensor([length], dtype=float).to(device)
    return mean, prec, rec, f1s, precs, recs


def compute_mpa(input, target):
    '''
    Calculate Mean Pixel Accuracy
    :param input: shape[batch, classes, h, w]
    :param target: shape[batch, h, w]
    :return mpa: Mean Pixel Accuracy
    '''
    eps = 0.0001
    length = input.size()[1]
    input = torch.argmax(input, dim=1)
    input = input.int()
    numerator = denominator = mpa = torch.zeros(1).to(device)
    for i in range(length):
        numerator = torch.sum((input.view(-1) == i).float()
                              * (target.view(-1) == i).float()).to(device)
        denominator = torch.sum((target.view(-1) == i).float()).to(device)
        mpa += (numerator + eps) / (denominator + eps)
    mpa /= torch.tensor([length], dtype=float).to(device)
    return mpa


def compute_precision(input, target):
    '''
    Calculate Precision
    '''
    eps = 0.0001
    # input 是经过了sigmoid 之后的输出。
    input = (input > 0.5).float()
    target = (target > 0.5).float()

    numerator = torch.sum(target.view(-1) * input.view(-1)).to(device)
    denominator = torch.sum((target == 1.0).float()).to(device)

    t = (numerator + eps) / (denominator + eps)
    return t
