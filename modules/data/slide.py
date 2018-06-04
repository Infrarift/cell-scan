# -*- coding: utf-8 -*-

"""
slide | cell-scan | 3/06/18
A container for the single slide (image) of one scan.
"""
import uuid
from typing import List

import cv2

from modules.data.cell import Cell

__author__ = "Jakrin Juangbhanich"
__email__ = "juangbhanich.k@gmail.com"


class Slide:
    def __init__(self):
        self.image_path = None
        self.name = uuid.uuid4().hex
        self.slide_path = None
        self.cells: List[Cell] = []
        self._purity_factor = None

    @property
    def image(self):
        return cv2.imread(self.image_path)

    @property
    def purity(self):
        if self._purity_factor is None:
            pos_area = sum(cell.region.area for cell in self.cells if cell.type == Cell.POSITIVE)
            neg_area = sum(cell.region.area for cell in self.cells if cell.type == Cell.NEGATIVE)
            total_area = pos_area + neg_area
            self._purity_factor = pos_area / total_area
        return self._purity_factor


