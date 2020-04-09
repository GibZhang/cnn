#!/usr/bin/python
# -*- coding: UTF-8 -*-
# @Time    : 2020-04-09 10:54
# @Author  : zhangjingbo
# @mail    : jingbo.zhang@yooli.com
# @File    : style_trans_model.py
# @Description: 风格迁移模型
# @Software: PyCharm
import torch
import torch.nn as nn
from torch.optim.lbfgs import LBFGS
import torch.nn.functional as F
import torchvision.models as models
import copy
import torch.optim as optim
from 图片风格转换.util import load_img_tensor
import matplotlib.pyplot as plt
from 图片风格转换.util import imshow


class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()
        self.criterion = nn.MSELoss()

    def forward(self, input_image):
        self.loss = self.criterion(input_image.detach(), self.target).requires_grad_()
        return input_image


class StyleLoss(nn.Module):
    def __init__(self, target):
        super(StyleLoss, self).__init__()
        self.target = target.detach()
        self.criterion = nn.MSELoss()

    def gram_matrix(self, input):
        a, b, c, d = input.size()  # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)
        features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

        G = torch.mm(features, features.t())  # compute the gram product
        return G.__div__(a * b * c * d)

    def forward(self, input_image):
        self.loss = self.criterion(self.gram_matrix(input_image.detach()),
                                   self.gram_matrix(self.target.detach())).requires_grad_()
        return input_image


def load_pretrained_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn = models.vgg19(pretrained=True).features.to(device).eval()
    return cnn


def make_image(image_size):
    input = torch.randn(image_size, device='cpu')
    return input.clamp(0, 1)


def get_model_and_losses(cnn, content_img, style_img):
    content_layers_default = ['conv_4']
    style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
    cnn = copy.deepcopy(cnn)
    i = 0
    content_losses = []
    style_losses = []
    model = nn.Sequential()
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)
        if name in content_layers_default:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            content_losses.append(content_loss)
            model.add_module("content_loss_{}".format(i), content_loss)
        if name in style_layers_default:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            style_losses.append(style_loss)
            model.add_module("style_loss_{}".format(i), style_loss)

        if name == style_layers_default[-1]:
            break
    return model, content_losses, style_losses


def get_input_param_optimier(input_img):
    """
    input_img is a Variable
    """
    input_param = nn.Parameter(input_img.data)
    optimizer = LBFGS([input_param])
    return input_param, optimizer


def image_transfer():
    cnn = load_pretrained_model()
    content_img, style_img, image_size = load_img_tensor()
    input_image = make_image(image_size)  # 生成噪声图片
    input_param, optimizer = get_input_param_optimier(input_image)
    model_vgg, content_losses, style_losses = get_model_and_losses(cnn, content_img, style_img)
    num_iterators = 300
    epoch = [0]
    while epoch[0] < num_iterators:
        def closure():
            style_score = 0
            content_score = 0
            optimizer.zero_grad()
            input_param.data.clamp_(0, 1)
            model_vgg(input_param)
            for cl in content_losses:
                content_score += 1 * cl.loss
            for sl in style_losses:
                style_score += 10000 * sl.loss
            epoch[0] += 1
            if epoch[0] % 50 == 0:
                print(style_score, content_score)
                print()
            loss = style_score + content_score
            loss.backward()
            return loss

        optimizer.step(closure)
    return input_param.data.clamp_(0, 1)


if __name__ == '__main__':
    input = image_transfer()
    plt.figure()
    imshow(input)
