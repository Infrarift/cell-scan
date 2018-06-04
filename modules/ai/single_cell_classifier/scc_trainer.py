# -*- coding: utf-8 -*-

"""
scc_trainer | cell-scan | 4/06/18
<ENTER DESCRIPTION HERE>
"""
import os
import random

import cv2
import torch
from typing import List

import torchvision
from torch.autograd import Variable

from modules.ai.single_cell_classifier.scc_net import SCCNet
from modules.data.cell import Cell
from tools.util.logger import Logger

__author__ = "Jakrin Juangbhanich"
__email__ = "juangbhanich.k@gmail.com"


class SCCUnit:
    def __init__(self):
        self.image_path = None
        self.type = Cell.NEUTRAL

    @property
    def image(self):
        return cv2.imread(self.image_path)


class SCCTrainer:
    def __init__(self):
        self.net: SCCNet = SCCNet().cuda()
        pass

    def process(self, input_path):

        units = []
        folders = os.listdir(input_path)
        for folder in folders:

            type = Cell.NEUTRAL

            if folder == "positive":
                type = Cell.POSITIVE

            if folder == "negative":
                type = Cell.NEGATIVE

            folder_path = os.path.join(input_path, folder)
            files = os.listdir(folder_path)
            for file in files:
                unit = SCCUnit()
                unit.image_path = os.path.join(folder_path, file)
                unit.type = type
                units.append(unit)

        Logger.log_field("Units Created", len(units))
        Logger.log_field("Positive", sum(1 for unit in units if unit.type == Cell.POSITIVE))
        Logger.log_field("Negative", sum(1 for unit in units if unit.type == Cell.NEGATIVE))
        Logger.log_field("Neutral", sum(1 for unit in units if unit.type == Cell.NEUTRAL))

        self._train(units)
        self._eval(units)
        self._save("output/scc.pt")

    def _save(self, path):
        torch.save(self.net, path)

    def _train(self, units: List[SCCUnit]):

        epochs = 10

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.net.parameters())
        trans = torchvision.transforms.ToTensor()

        # Loss
        loss_factor = 0.3
        loss_recip = 1.0 - loss_factor

        # batch_max = 10

        for e in range(epochs):
            random.shuffle(units)
            # batch_num = min(len(units), batch_max)
            for unit in units:

                img, label = trans(unit.image), torch.Tensor([unit.type])
                img = img.view(1, 3, 64, 64)
                img, label = Variable(img).cuda(), Variable(label).long().cuda()

                output = self.net.forward(img)
                print(output)
                optimizer.zero_grad()
                loss = criterion(output, label)
                loss.backward()
                optimizer.step()
                new_loss = loss.data[0]
                if new_loss != 0:
                    loss = loss * loss_recip + new_loss * loss_factor

                print("Epoch number {}\n Current loss {}\n".format(e, loss))

    def _eval(self, units: List[SCCUnit]):

        correct = 0

        for unit in units:
            result = self.net.process(unit.image)
            if result == unit.type:
                correct += 1

        Logger.log_field("Accuracy", "{:.1f}%".format(100 * (correct/len(units))))
