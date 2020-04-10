#!/usr/bin/python
# -*- coding: UTF-8 -*-
# @Time    : 2020-04-10 17:40
# @Author  : zhangjingbo
# @mail    : jingbo.zhang@yooli.com
# @File    : content_style_loss.py
# @Description: 定义内容损失和风格损失函数
# @Software: PyCharm
import torch
import torch.nn as nn


class ContentLoss(nn.Module):
    def __init__(self, target, weight):
        super(ContentLoss, self).__init__()
        self.criterion = nn.MSELoss()
        self.weight = weight
        self.target = target.detach()

    def forward(self, input_t):
        self.loss = self.weight * self.criterion(input_t, self.target)
        out_put = input_t.clone()
        return out_put


class Gram(nn.Module):
    def __init__(self):
        super(Gram, self).__init__()

    def forward(self, input_t):
        a, b, c, d = input_t.size()
        feature = input_t.view(a * b, c * d)
        gram = torch.mm(feature, feature.t())
        gram /= (a * b * c * d)
        return gram


class StyleLoss(nn.Module):
    def __init__(self, target, weight):
        super(StyleLoss, self).__init__()
        self.target = target.detach()
        self.weight = weight
        self.gram = Gram()
        self.criterion = nn.MSELoss()

    def forward(self, input_t):
        self.loss = self.criterion(self.weight * self.gram(input_t), self.weight * self.gram(self.target))
        out_put = input_t.clone()
        return out_put
