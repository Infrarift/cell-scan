# -*- coding: utf-8 -*-

"""
slide | cell-scan | 3/06/18
A container for the single slide (image) of one scan.
"""
import uuid

import cv2

__author__ = "Jakrin Juangbhanich"
__email__ = "juangbhanich.k@gmail.com"


class Slide:
    def __init__(self):
        self.image_path = None
        self.name = uuid.uuid4().hex

    @property
    def image(self):
        return cv2.imread(self.image_path)

