"""
Utility functions for training

Author: Zhuo Su, Wenzhe Liu
Date: Aug 22, 2020
"""

from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import os
import shutil
import math
import time
import random
import skimage
import numpy as np
from skimage import io
from skimage.transform import resize

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


######################################
#       measurement functions        #
######################################

def get_model_parm_nums(model):
    total = sum([param.numel() for param in model.parameters()])
    total = float(total) / 1e6
    return total



######################################
#         basic functions            #
######################################

def load_checkpoint(args, running_file):

    model_dir = os.path.join(args.savedir, 'save_models')
    latest_filename = os.path.join(model_dir, 'latest.txt')
    model_filename = ''

    if args.evaluate is not None:
        model_filename = args.evaluate
    else:
        if os.path.exists(latest_filename):
            with open(latest_filename, 'r') as fin:
                model_filename = fin.readlines()[0].strip()
    loadinfo = "=> loading checkpoint from '{}'".format(model_filename)
    print(loadinfo)

    state = None
    if os.path.exists(model_filename):
        state = torch.load(model_filename, map_location='cpu')
        # https://blog.csdn.net/Fly2Leo/article/details/122352956
        weights_dict = {}
        for k, v in state["state_dict"].items():
            new_k = k.replace('module.', '') if 'module' in k else k
            weights_dict[new_k] = v
        state["state_dict"] = weights_dict
        
        loadinfo2 = "=> loaded checkpoint '{}' successfully".format(model_filename)
    else:
        loadinfo2 = "no checkpoint loaded"
    print(loadinfo2)
    running_file.write('%s\n%s\n' % (loadinfo, loadinfo2))
    running_file.flush()
    return state


def load_checkpoint_without_runningfile(args):

    model_dir = os.path.join(args.savedir, 'save_models')
    latest_filename = os.path.join(model_dir, 'latest.txt')
    model_filename = ''

    if args.evaluate is not None:
        model_filename = args.evaluate
    else:
        if os.path.exists(latest_filename):
            with open(latest_filename, 'r') as fin:
                model_filename = fin.readlines()[0].strip()
    loadinfo = "=> loading checkpoint from '{}'".format(model_filename)
    print(loadinfo)

    state = None
    if os.path.exists(model_filename):
        state = torch.load(model_filename, map_location='cpu')
        loadinfo2 = "=> loaded checkpoint '{}' successfully".format(model_filename)
    else:
        loadinfo2 = "no checkpoint loaded"
    print(loadinfo2)
    return state


def save_checkpoint(state, epoch, root, saveID, keep_freq=10):

    filename = 'checkpoint_%03d.pth' % epoch
    model_dir = os.path.join(root, 'save_models')
    model_filename = os.path.join(model_dir, filename)
    latest_filename = os.path.join(model_dir, 'latest.txt')

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # write new checkpoint 
    torch.save(state, model_filename)
    with open(latest_filename, 'w') as fout:
        fout.write(model_filename)
    print("=> saved checkpoint '{}'".format(model_filename))

    # remove old model
    if saveID is not None and (saveID + 1) % keep_freq != 0:
        filename = 'checkpoint_%03d.pth' % saveID
        model_filename = os.path.join(model_dir, filename)
        if os.path.exists(model_filename):
            os.remove(model_filename)
            print('=> removed checkpoint %s' % model_filename)

    print('##########Time##########', time.strftime('%Y-%m-%d %H:%M:%S'))
    return epoch


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        #self.sum += val * n
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, epoch, args):
    method = args.lr_type
    if method == 'cosine':
        T_total = float(args.epochs)
        T_cur = float(epoch)
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * T_cur / T_total))
    elif method == 'multistep':
        lr = args.lr
        for epoch_step in args.lr_steps:
            if epoch >= epoch_step:
                lr = lr * 0.1

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    str_lr = '%.6f' % lr
    return str_lr


######################################
#     edge specific functions        #
######################################


def cross_entropy_loss_RCF(prediction, labelf, beta=1.1):
    label = labelf.long()
    mask = labelf.clone()

    num_positive = torch.sum(label == 1).float()
    num_negative = torch.sum(label == 0).float()

    mask[label == 1] = 1.0 * num_negative / (num_positive + num_negative)
    mask[label == 0] = beta * num_positive / (num_positive + num_negative)
    mask[label == 2] = 0
    cost = F.binary_cross_entropy(prediction, labelf, weight=mask, reduction='sum')

    return cost


def weighted_nornal_cross_entropy_loss(prediction, labelf, beta=1.1):
    label = labelf.long()
    mask = labelf.clone()
    num_positive = torch.sum(label > 0).float()
    num_negative = torch.sum(label == 0).float()

    mask[label > 0] = 1.0 * num_negative / (num_positive + num_negative)
    mask[label == 0] = beta * num_positive / (num_positive + num_negative)

    cost = F.binary_cross_entropy(prediction, labelf, weight=mask, reduction='sum')

    return cost

def nornal_cross_entropy_loss(prediction, labelf):
    cost = F.binary_cross_entropy(prediction, labelf, reduction='sum')
    return cost


# loss fcuntions for crisp edge detection
def Dice_Loss(pred, label):
    # pred = torch.sigmoid(pred)
    smooth = 1
    pred_flat = pred.view(-1)
    label_flat = label.view(-1)

    intersecion = pred_flat * label_flat
    unionsection = pred_flat.pow(2).sum() + label_flat.pow(2).sum() + smooth
    loss = unionsection / (2 * intersecion.sum() + smooth)
    loss = loss.sum()
    return loss


def bdrloss(prediction, label, radius):
    '''
    The boundary tracing loss that handles the confusing pixels.
    '''

    filt = torch.ones(1, 1, 2*radius+1, 2*radius+1)
    filt.requires_grad = False
    filt = filt.cuda()

    bdr_pred = prediction * label
    pred_bdr_sum = label * F.conv2d(bdr_pred, filt, bias=None, stride=1, padding=radius)

    texture_mask = F.conv2d(label.float(), filt, bias=None, stride=1, padding=radius)
    mask = (texture_mask != 0).float()
    mask[label == 1] = 0
    pred_texture_sum = F.conv2d(prediction * (1-label) * mask, filt, bias=None, stride=1, padding=radius)

    softmax_map = torch.clamp(pred_bdr_sum / (pred_texture_sum + pred_bdr_sum + 1e-10), 1e-10, 1 - 1e-10)
    cost = -label * torch.log(softmax_map)
    cost[label == 0] = 0

    return cost.sum()


def textureloss(prediction, label, mask_radius):
    '''
    The texture suppression loss that smooths the texture regions.
    '''
    filt1 = torch.ones(1, 1, 3, 3)
    filt1.requires_grad = False
    filt1 = filt1.cuda()
    filt2 = torch.ones(1, 1, 2*mask_radius+1, 2*mask_radius+1)
    filt2.requires_grad = False
    filt2 = filt2.cuda()

    pred_sums = F.conv2d(prediction.float(), filt1, bias=None, stride=1, padding=1)
    label_sums = F.conv2d(label.float(), filt2, bias=None, stride=1, padding=mask_radius)

    mask = 1 - torch.gt(label_sums, 0).float()

    loss = -torch.log(torch.clamp(1-pred_sums/9, 1e-10, 1-1e-10))
    loss[mask == 0] = 0

    return torch.sum(loss)


def tracingloss(prediction, label, tex_factor=0., bdr_factor=0., balanced_w=1.1):
    label = label.float()
    prediction = prediction.float()
    with torch.no_grad():
        mask = label.clone()

        num_positive = torch.sum((mask==1).float()).float()
        num_negative = torch.sum((mask==0).float()).float()
        beta = num_negative / (num_positive + num_negative)
        mask[mask == 1] = beta
        mask[mask == 0] = balanced_w * (1 - beta)
        mask[mask == 2] = 0

    cost = torch.sum(torch.nn.functional.binary_cross_entropy(prediction.float(), label.float(), weight=mask, reduction='none'))

    label_w = (label != 0).float()
    textcost = textureloss(prediction.float(), label_w.float(), mask_radius=2)
    bdrcost = bdrloss(prediction.float(), label_w.float(), radius=2)
    # return cost
    return cost + bdr_factor*bdrcost + tex_factor*textcost


######################################
#         debug functions            #
######################################

# no function currently
