# -*- coding: utf-8 -*-

"""
Main AI class. Will run the neural network on the cells, and do some post processing.
"""

import cv2
import numpy as np
from .equalizer import Equalizer
from .mask_rcnn import MaskRCNN
from .prediction import Prediction, PredictionUnit
from tools.util.logger import Logger
from tools.util.region import Region

__author__ = "Jakrin Juangbhanich"
__copyright__ = "Copyright 2018, Jakrin Juangbhanich"
__email__ = "juangbhanich.k@gmail.com"


class CellNet:

    K_SCORE_LIMIT = 0.70
    K_VOLUME_LIMIT = 0.85
    K_MAX_CYCLES = 1  # Maybe 3 is good?
    K_USE_MORPH = True  # If true, will post process the output mask with a morph transform.

    def __init__(self, net_name):
        self.m_rcnn = MaskRCNN()
        self.m_rcnn.load("models/{}.pth".format(net_name))
        self.equalizer = Equalizer()

    def _get_max_volume(self, height, width):
        return width * self.K_VOLUME_LIMIT * height * self.K_VOLUME_LIMIT

    def cycle_predict(self, image, equalizer_mask=None):

        prediction = Prediction()
        extract_image = np.copy(image)
        extract_mask = np.ones_like(equalizer_mask, dtype=np.bool)

        width = image.shape[1]
        height = image.shape[0]
        max_volume = self._get_max_volume(height, width)

        attempts = 0
        while attempts < self.K_MAX_CYCLES:
            attempts += 1
            units, extract_image, extract_mask = self.predict_and_extract(extract_image, extract_mask, max_volume)
            if len(units) > 0:
                for unit in units:
                    prediction.units.append(unit)
            else:
                break

        Logger.log_field("Prediction Attempts", attempts)
        Logger.log_field("Total Cells Predicted", len(prediction.units))
        return prediction

    def predict_and_extract(self, extract_image, extract_mask, max_volume):

        # Predict the top bounding boxes in this image.
        try:
            results = self.m_rcnn.process(extract_image)
        except Exception:
            # Prediction failed.
            return [], extract_image, extract_mask

        # Split out the results.
        masks = results["masks"]
        scores = results["scores"]
        rois = results["rois"]
        regions = []
        units = []

        # Create the regions for this image.
        for i in range(len(rois)):

            mask = masks[:, :, i].astype(np.bool)
            roi = rois[i]
            score = scores[i]

            if score > self.K_SCORE_LIMIT:

                region: Region = Region()
                region.top = int(roi[0])
                region.bottom = int(roi[2])
                region.left = int(roi[1])
                region.right = int(roi[3])

                if region.area < max_volume:

                    sub_mask = extract_mask * self._morph(mask)
                    if np.any(sub_mask):
                        region.score = score
                        regions.append(region)
                        unit = PredictionUnit(region, score, sub_mask)
                        units.append(unit)
                        erase_mask = np.logical_not(sub_mask)
                        extract_mask = extract_mask * erase_mask

                        # If we are going to predict again, we need to cut out
                        # some of the image that we don't want to use.
                        r = region
                        e_pad = 2
                        e_top = int(max(r.top - e_pad, 0))
                        e_bot = int(min(r.bottom + e_pad, extract_image.shape[0]))
                        e_left = int(max(r.left - e_pad, 0))
                        e_right = int(min(r.right + e_pad, extract_image.shape[1]))
                        extract_image[e_top:e_bot, e_left:e_right, :] = np.zeros((e_bot - e_top, e_right - e_left, 3))

        return units, extract_image, extract_mask

    def _morph(self, mask):
        if not self.K_USE_MORPH:
            return mask
        img = mask.astype(np.uint8)
        kernel_size = 2
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        result = img
        result = cv2.morphologyEx(result, cv2.MORPH_DILATE, kernel, iterations=1)
        return result.astype(np.bool)

