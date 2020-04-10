#!/usr/bin/python
# -*- coding: UTF-8 -*-
# @Time    : 2020-04-10 11:01
# @Author  : zhangjingbo
# @mail    : jingbo.zhang@yooli.com
# @File    : model_loader.py
# @Description: 加载预训练模型，固定参数
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
from pytorch_recepit.image_style_transfer.content_style_loss import ContentLoss, StyleLoss


def cnn_loader():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn = models.vgg19(pretrained=False)
    cnn_paramters = torch.load('./vgg19-dcbb9e9d.pth')
    cnn.load_state_dict(cnn_paramters)
    cnn = cnn.features.to(device).eval()
    for param in cnn.parameters():
        param.requires_grad = False
    return cnn


def get_model_losses(cnn, content_image, style_image, content_weight, style_weight):
    content_layer = ['conv_4']
    style_layer = ['conv_2', 'conv_3', 'conv_4', 'conv_5']
    model = nn.Sequential()
    content_losses = []
    style_losses = []
    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))
        model.add_module(name, layer)
        if name in content_layer:
            m_name = 'content_loss_{}'.format(i)
            target = model(content_image)
            content_loss = ContentLoss(target, content_weight)
            content_losses.append(content_loss)
            model.add_module(m_name, content_loss)
        if name in style_layer:
            m_name = 'style_loss_{}'.format(i)
            target = model(style_image)
            style_loss = StyleLoss(target, style_weight)
            style_losses.append(style_loss)
            model.add_module(m_name, style_loss)
        if name == style_layer[-1]:
            break
    return model, content_losses, style_losses
