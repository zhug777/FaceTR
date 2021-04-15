from torch import nn
import torch
import numpy as np
from torch.nn import functional as F
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class KptsMSELoss(nn.Module):
    def __init__(self):
        super(KptsMSELoss, self).__init__()
        self.loss_func = nn.MSELoss(reduction='mean')

    def forward(self, output, target):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape(
            (batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            loss += 0.5 * self.loss_func(heatmap_pred, heatmap_gt)

        return loss / num_joints

'''
class FocalLoss2d(nn.Module):
    def __init__(self, gamma=0, weight=None, size_average=True):
        super(FocalLoss2d, self).__init__()

        self.gamma = gamma
        self.weight = weight
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.contiguous().view(input.size(0), input.size(1), -1)
            input = input.transpose(1,2)
            input = input.contiguous().view(-1, input.size(2)).squeeze()
        if target.dim()==4:
            target = target.contiguous().view(target.size(0), target.size(1), -1)
            target = target.transpose(1,2)
            target = target.contiguous().view(-1, target.size(2)).squeeze()
        elif target.dim()==3:
            target = target.view(-1)
        else:
            target = target.view(-1, 1)
        
        # compute the negative likelyhood
        weight = Variable(self.weight)
        logpt = -F.cross_entropy(input, target)
        pt = torch.exp(logpt)

        # compute the loss
        loss = -((1-pt)**self.gamma) * logpt

        # averaging (or not) loss
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
'''

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, input, target):
        length = input.size()[1]
        if isinstance(self.alpha, list):
            assert len(self.alpha)==length, "Wrong length of alpha!"
            self.alpha = torch.FloatTensor(self.alpha).to(device)
        elif isinstance(self.alpha, int) or isinstance(self.alpha, float):
            self.alpha = torch.FloatTensor([self.alpha]*length).to(device)
        # important to add reduction='none' to keep per-batch-item loss
        loss_func = nn.CrossEntropyLoss(weight=self.alpha, reduction='none')
        ce_loss = loss_func(input, target)
        pt = torch.exp(-ce_loss).to(device)
        focal_loss = ((1-pt)**self.gamma * ce_loss)
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()


def focal_loss(input, target, alpha=None, gamma=2):
    length = input.size()[1]
    if isinstance(alpha, list):
        assert len(alpha)==length, "Wrong length of alpha!"
        alpha = torch.FloatTensor(alpha).to(device)
    elif isinstance(alpha, int) or isinstance(alpha, float):
        alpha = torch.FloatTensor([alpha]*length).to(device)

    # important to add reduction='none' to keep per-batch-item loss
    loss_func = nn.CrossEntropyLoss(weight=alpha, reduction='none')
    ce_loss = loss_func(input, target)
    pt = torch.exp(-ce_loss).to(device)
    focal_loss = ((1-pt)**gamma * ce_loss).mean()  # mean over the batch
    return focal_loss
    

def get_loss(loss, alpha=None, gamma=2, size_average=True):
    loss_func = None
    if loss == 'bce_logits':
        loss_func = nn.BCEWithLogitsLoss()
    elif loss == 'ce':
        loss_func = nn.CrossEntropyLoss()
    elif loss == 'focal':
        loss_func = FocalLoss(alpha=alpha, gamma=gamma, size_average=size_average)
    return loss_func