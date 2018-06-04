# -*- coding: utf-8 -*-

"""
scc_net | cell-scan | 4/06/18
Scan a single cell input image and decide whether it is positive, negative, or neutral.
"""
import torch

from torch.nn import Module
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from torchvision import transforms

__author__ = "Jakrin Juangbhanich"
__email__ = "juangbhanich.k@gmail.com"


class SCCNet(Module):
    """ With this network, I don't care much about its accuracy. It is only used to provide extra information. In
    line with this goal, I have tried to minimize the parameter count, whilst maintaining the best score possible."""

    K_PADDING = 1

    def __init__(self, target_size=64):
        super().__init__()

        self.target_size = target_size

        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(self.K_PADDING),
            nn.Conv2d(3, 6, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(6),

            nn.ReflectionPad2d(self.K_PADDING),
            nn.Conv2d(6, 12, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(12),

            nn.ReflectionPad2d(self.K_PADDING),
            nn.Conv2d(12, 24, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(24),
        )

        # Calculate the flattened layer size, and create the FC layers with it.
        o1 = self._get_output_size(self.target_size, 4, 2, self.K_PADDING)
        o2 = self._get_output_size(o1, 2, 2, self.K_PADDING)
        o3 = self._get_output_size(o2, 2, 2, self.K_PADDING)
        final_layer_size = o3 * o3 * 24

        self.fc1 = nn.Sequential(
            nn.Linear(final_layer_size, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 3),
            nn.Softmax()
        )

    @staticmethod
    def _get_output_size(width, filter_size, stride, padding):
        """ Returns the output size for the convolution layer."""
        return (width - filter_size + 2 * padding) // stride + 1

    def forward(self, x):
        x = self.cnn1(x)
        x = x.view(x.size()[0], -1)
        output = self.fc1(x)
        # output = F.normalize(output, p=2, dim=1)
        return output

    def process(self, image):

        trans = transforms.ToTensor()
        tensor = trans(image)

        img = tensor.view(1, 3, 64, 64)
        img = Variable(img).cuda()
        result = self.forward(img)
        result = int(torch.argmax(result))
        return result
