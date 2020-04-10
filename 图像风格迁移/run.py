#!/usr/bin/python
# -*- coding: UTF-8 -*-
# @Time    : 2020-04-10 18:19
# @Author  : zhangjingbo
# @mail    : jingbo.zhang@yooli.com
# @File    : run.py
# @Description: 执行训练任务
# @Software: PyCharm
from matplotlib.pylab import plt
from torch.optim.lbfgs import LBFGS
import torch.nn as nn
from pytorch_recepit.image_style_transfer.image_loader import imshow
from pytorch_recepit.image_style_transfer.image_loader import load_img_tensor, input_image
from pytorch_recepit.image_style_transfer.model_loader import cnn_loader, get_model_losses


def train():
    content_image, style_image, image_size = load_img_tensor()
    # imshow(content_image)
    # imshow(style_image)
    # input_params = input_image(image_size)
    input_params = nn.Parameter(content_image, requires_grad=True)
    cnn = cnn_loader()
    model, content_losses, style_losses = get_model_losses(cnn, content_image, style_image, 1, 1000)
    epoch = [0]
    num_epoches = 100
    optimizer = LBFGS([input_params])
    content_loss_list = []
    style_loss_list = []
    while epoch[0] < num_epoches:
        def closure():
            optimizer.zero_grad()
            model(input_params)
            content_score = 0
            style_score = 0
            for cs in content_losses:
                content_score += cs.loss
            for ss in style_losses:
                style_score += ss.loss
            loss = content_score + style_score
            loss.backward()
            epoch[0] += 1
            if epoch[0] % 50 == 1:
                print('content score: {}, style score: {}'.format(content_score, style_score))
            content_loss_list.append(content_score)
            style_loss_list.append(style_score)
            return loss

        optimizer.step(closure)
    return input_params, content_loss_list, style_loss_list


def losses_plot(losses, xtitle):
    plt.figure()
    plt.plot(losses)
    plt.xlabel(xtitle)
    plt.show()


if __name__ == '__main__':
    input_image, content_loss_list, style_loss_list = train()
    imshow(input_image.clamp(0, 1))
    # losses_plot(content_loss_list, 'content_loss')
    # losses_plot(style_loss_list, 'style_loss')
