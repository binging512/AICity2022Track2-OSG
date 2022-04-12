#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Loss functions."""

from numpy.core.shape_base import stack
import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftTargetCrossEntropy(nn.Module):
    """
    Cross entropy loss with soft target.
    """

    def __init__(self, reduction="mean"):
        """
        Args:
            reduction (str): specifies reduction to apply to the output. It can be
                "mean" (default) or "none".
        """
        super(SoftTargetCrossEntropy, self).__init__()
        self.reduction = reduction

    def forward(self, x, y):
        loss = torch.sum(-y * F.log_softmax(x, dim=-1), dim=-1)
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "none":
            return loss
        else:
            raise NotImplementedError

class BCE_VIRAT(nn.Module):
    def __init__(self, reduction="mean", hard_thres=-1, weight=None, pos_weight=None, mode = 'personcar'):
        """
        :param hard_thres:
            -1：软标签损失，直接基于标注中的软标签计算BECLoss；
            >0：硬标签损失，将标签大于hard_thres的置为1，否则为0；
        """
        super(BCE_VIRAT, self).__init__()
        self.hard_thres = hard_thres
        self._loss_fn = nn.BCEWithLogitsLoss(reduction=reduction,weight=weight,pos_weight=pos_weight)
        if mode == 'personcar':
            # self.class_list = [[0,1],[2,3],[4,5],[6,7],[8,9],[10,11]]
            # self.class_list = [[0,1],[2,3],[4,5],[6,7],[8,9]]
            self.class_list = []
        elif mode == 'personstructure':
            self.class_list = [[0,1],[2,3]]
        

    def forward(self, x, y):
        if self.hard_thres > 0:  # 硬标签
            mask = y > self.hard_thres
            y[mask] = 1.
            y[~mask] = 0.
        loss = self._loss_fn(x, y)
        x = torch.sigmoid(x)
        loss_plus = torch.zeros(x.shape[0],device='cuda')
        all_weight = torch.zeros(x.shape[0], device='cuda') + 1e-5
        for pair in self.class_list:
            weight, _ = torch.max(torch.stack((y[:,pair[0]], y[:,pair[1]]),dim = 1), dim=1)
            all_weight += weight
            x_margin = torch.abs(x[:,pair[0]]-x[:,pair[1]])
            # x_max,_ = torch.max(torch.stack((x[:,pair[0]],x[:,pair[1]]), dim=1),dim=1)
            loss_p = weight*(1 - x_margin)
            loss_plus += loss_p
        loss_plus = loss_plus/all_weight
        loss = loss + 0.01*torch.mean(loss_plus)
        return loss

class BCE_CFNL(nn.Module):
    def __init__(self, reduction='none', hard_thres=-1, weight=None, pos_weight=None):
        """
        :param hard_thres:
            -1：软标签损失，直接基于标注中的软标签计算BECLoss；
            >0：硬标签损失，将标签大于hard_thres的置为1，否则为0；
        """
        super(BCE_CFNL, self).__init__()
        self.hard_thres = hard_thres
        self._loss_fn = nn.BCEWithLogitsLoss(reduction=reduction,weight=weight,pos_weight=pos_weight)

    def forward(self, x, y, w=None):
        if self.hard_thres > 0:  # 硬标签
            mask = y > self.hard_thres
            y[mask] = 1.
            y[~mask] = 0.
        if w == None:
            w = torch.ones(x.shape[0])
        x = torch.sigmoid(x)
        loss = self._loss_fn(x, y)
        loss = torch.mean(loss,dim=1)*w
        loss = torch.mean(loss)
        return loss

_LOSSES = {
    "cross_entropy": nn.CrossEntropyLoss,
    "bce": nn.BCELoss,
    "bce_logit": nn.BCEWithLogitsLoss,
    "soft_cross_entropy": SoftTargetCrossEntropy,
    "bce_virat": BCE_VIRAT,
}


def get_loss_func(loss_name):
    """
    Retrieve the loss given the loss name.
    Args (int):
        loss_name: the name of the loss to use.
    """
    if loss_name not in _LOSSES.keys():
        raise NotImplementedError("Loss {} is not supported".format(loss_name))
    return _LOSSES[loss_name]
