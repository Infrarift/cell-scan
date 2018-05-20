#!$(which python)
# -*- coding: utf-8 -*-

"""
scanner | cell-scan | 20/05/18
<ENTER DESCRIPTION HERE>
"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import colorsys
import csv
import json
import random

import numpy as np

import cv2

from tools.util.logger import Logger

__author__ = "Jakrin Juangbhanich"
__email__ = "juangbhanich.k@gmail.com"
__version__ = "0.0.0"


class Cell:
    def __init__(self):
        self.points = None
        self.x_points = []
        self.y_points = []

    def calibrate_points(self):
        size = len(self.x_points)
        self.points = np.zeros((size, 2), dtype=np.int32)
        for i in range(size):
            self.points[i][0] = self.x_points[i]
            self.points[i][1] = self.y_points[i]


class Scanner:
    def __init__(self):
        pass

    def scan(self, image_path, label_path):

        image = cv2.imread(image_path)
        Logger.log_field("Image Loaded Size", image.shape)

        cells = []

        # Read the CSV labels.
        with open(label_path) as file:
            reader = csv.reader(file, delimiter=",")
            skip_row = True
            for row in reader:

                # Skip the header.
                if skip_row:
                    skip_row = False
                    continue

                region_shape = json.loads(row[5])
                # region_attributes = json.loads(row[6])

                cell = Cell()
                cell.x_points = region_shape["all_points_x"]
                cell.y_points = region_shape["all_points_y"]
                cell.calibrate_points()
                cells.append(cell)

        overlay = np.zeros_like(image)

        colors = self.random_colors(len(cells))
        i = 0
        for cell in cells:
            cv2.fillConvexPoly(overlay, cell.points, colors[i] * 0.2)
            cv2.polylines(overlay, [cell.points], True, colors[i], 24)
            i += 1
            pass

        compose = cv2.addWeighted(image, 1.0, overlay, 1.0, 0.0)
        compose = cv2.resize(compose, (512, 512))
        overlay = cv2.resize(overlay, (512, 512))

        final = np.zeros((512, 1024, 3), dtype=np.uint8)
        final[:512, :512] = compose
        final[:512, 512:] = overlay

        cv2.imwrite("input/sample.png", final)

        # cv2.imshow("Cell Scan", image)
        # cv2.waitKey(5000)
        # cv2.destroyAllWindows()

    @staticmethod
    def random_colors(n):
        """
        Generate random colors.
        To get visually distinct colors, generate them in HSV space then
        convert to RGB.
        """
        brightness = 1.0
        hsv = [(i / n, 1, brightness) for i in range(n)]
        colors_raw = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        colors = np.array(colors_raw)
        colors *= 255
        np.random.shuffle(colors)
        # random.shuffle(colors)
        return colors

