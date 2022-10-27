import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.prediction2rgbmask import pred2mask

'''
the metrics for validation and test
'''

def dice_coefficient(pred, mask, smooth=1e-8, ignore_channel=None):
    '''
    :param pred: prediction from model with range [0, 1], shape [batch,channel,h,w]
    :param mask: mask of ground truth with range[0, 1], shape [batch,channel,h,w]
    :param smooth: smooth value
    :param ignore_channel: channel to ignore (to filter out certain class)
    :return: dice coefficient, value between 0, 1. 1 indicates the best segmentation.
    '''
    total_score = torch.tensor(0.0).cuda()
    c = pred.shape[1]
    for i in range(c):
        if i != ignore_channel:
            intersec = torch.sum(pred[:, i, :, :] * mask[:, i, :, :])
            addition = torch.sum(pred[:, i, :, :]) + torch.sum(mask[:, i, :, :])
            score = (intersec * 2 + smooth) / (addition + smooth)
            total_score += score
    return total_score / c


def dice_loss(pred, mask, smooth=1e-8, ignore_channel=None):
    return 1 - dice_coefficient(pred, mask, smooth, ignore_channel)


def pixel_accuracy(pred, mask):
    '''
    :param pred: prediction from model with value 0 or 1, shape [batch,channel,h,w]
    :param mask: mask of ground truth with value 0 or 1, shape [batch,channel,h,w]
    :return: accuracy of prediction compared to ground truth.
    '''
    batch = pred.shape[0]
    acc = 0.0
    for i in range(batch):
        pred = pred[i, :].unsqueeze(0)
        pred_mask = pred2mask(pred)
        pred_mask = torch.permute(pred_mask, (2, 0, 1))
        pred_mask = torch.unsqueeze(pred_mask, dim=0).cuda()
        correct_pixel = torch.sum(pred[pred_mask == mask])
        total_pixel = torch.numel(pred)
        acc += correct_pixel / total_pixel

    return acc / batch


def mIOU(pred, mask):
    '''
    :param pred: prediction from model with value 0 or 1, shape [batch,channel,h,w]
    :param mask: mask of ground truth with value 0 or 1, shape [batch,channel,h,w]
    :return: accuracy of prediction compared to ground truth.
    '''
    intersec = torch.sum(pred * mask)
    addition = torch.sum(pred + mask)
    union = addition - intersec
    return intersec / union
