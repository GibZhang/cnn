#!/usr/bin/python
# -*- coding: UTF-8 -*-
# @Time    : 2020-04-04 10:25
# @Author  : zhangjingbo
# @mail    : jingbo.zhang@yooli.com
# @File    : util.py
# @Description: 文件风格转换的工具类
# @Software: PyCharm

from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_size = 512 if torch.cuda.is_available() else 128

loader = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()])


def image_loader(image_name):
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


style_img = image_loader("./picasso.jpg")
content_img = image_loader("./dancing.jpg")

assert style_img.size() == content_img.size(), \
    "we need to import style and content images of the same size"
unloader = transforms.ToPILImage()  # reconvert into PIL image

plt.ion()


def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# plt.figure()
# imshow(style_img, title='Style Image')
#
# plt.figure()
# imshow(content_img, title='Content Image')


def load_img_tensor():
    style_img = image_loader("./picasso.jpg")
    content_img = image_loader("./dancing.jpg")

    assert style_img.size() == content_img.size(), \
        "we need to import style and content images of the same size"
    return content_img, style_img, style_img.size()



