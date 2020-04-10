#!/usr/bin/python
# -*- coding: UTF-8 -*-
# @Time    : 2020-04-10 10:55
# @Author  : zhangjingbo
# @mail    : jingbo.zhang@yooli.com
# @File    : image_loader.py
# @Description:  加载训练图片输出tensor
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms


def image_loader(image_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = 512 if torch.cuda.is_available() else 128
    loader = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()])
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


def load_img_tensor():
    # 从图片加载张量并返回
    style_img = image_loader("./picasso.jpg")
    content_img = image_loader("./dancing.jpg")

    assert style_img.size() == content_img.size(), \
        "we need to import style and content images of the same size"
    return content_img, style_img, style_img.size()


def imshow(tensor, title=None):
    # 张量展示图片
    unloader = transforms.ToPILImage()  # reconvert into PIL image
    plt.ion()
    plt.figure()
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


def input_image(image_size):
    input_noise = torch.randn(image_size).clamp(0, 1)
    input_params = nn.Parameter(input_noise, requires_grad=True)
    return input_params
