# -*- coding: utf-8 -*-

"""
single_cell_classifier | cell-scan | 4/06/18
<ENTER DESCRIPTION HERE>
"""
import random

import cv2

from modules.ai.single_cell_classifier.scc_net import SCCNet
from modules.data import cell
from modules.data.cell import Cell

__author__ = "Jakrin Juangbhanich"
__email__ = "juangbhanich.k@gmail.com"


class SingleCellClassifier:
    def __init__(self):
        self.net = SCCNet().cuda()
        pass

    def process(self, image):
        """ Process a single cell image and return whether it is positive, negative, or neutral. """
        image = cv2.resize(image, (64, 64))
        result = self.net.process(image)
        if random.random() < 0.2:
            return Cell.NEUTRAL

        if random.random() < 0.5:
            return Cell.POSITIVE

        return Cell.NEGATIVE
