#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F


#  import ohem_cpp
#  class OhemCELoss(nn.Module):
#
#      def __init__(self, thresh, lb_ignore=255):
#          super(OhemCELoss, self).__init__()
#          self.score_thresh = thresh
#          self.lb_ignore = lb_ignore
#          self.criteria = nn.CrossEntropyLoss(ignore_index=lb_ignore, reduction='mean')
#
#      def forward(self, logits, labels):
#          n_min = labels[labels != self.lb_ignore].numel() // 16
#          labels = ohem_cpp.score_ohem_label(
#                  logits, labels, self.lb_ignore, self.score_thresh, n_min).detach()
#          loss = self.criteria(logits, labels)
#          return loss


class OhemCELoss(nn.Module):

    def __init__(self, thresh, lb_ignore=255):
        '''

        :param thresh: 困难样本的阈值
        :param lb_ignore: 需要忽略的标签值
        '''
        super(OhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, requires_grad=False, dtype=torch.float)).cuda()
        self.lb_ignore = lb_ignore
        self.criteria = nn.CrossEntropyLoss(ignore_index=lb_ignore, reduction='none')

    def forward(self, logits, labels):
        n_min = labels[labels != self.lb_ignore].numel() // 16
        # 计算 n_min，即非忽略标签的数量的 1/16。这个值用于确保每个批次中至少有 n_min 个样本被用于计算损失。
        loss = self.criteria(logits, labels).view(-1)
        # 计算每个值的交叉熵
        loss_hard = loss[loss > self.thresh]
        if loss_hard.numel() < n_min:
            loss_hard, _ = loss.topk(n_min)
        return torch.mean(loss_hard)


if __name__ == '__main__':
    pass

