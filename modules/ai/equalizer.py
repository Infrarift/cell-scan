# -*- coding: utf-8 -*-

"""
The equalizer is an image pre-processor that should be called on our inputs before they are fed into the neural network.
The goal of this process is to boost the signals of nuclei, normalize the signal range, and distinguish it from the BG.
"""

import cv2
import numpy as np

__author__ = "Jakrin Juangbhanich"
__copyright__ = "Copyright 2018, Jakrin Juangbhanich"
__email__ = "juangbhanich.k@gmail.com"


class Equalizer:

    # Use this to tune the signal strength.
    K_GAIN = 1.5
    K_DECAY = 1.0
    K_SIGNAL_THRESHOLD = 0.05

    def __init__(self):
        pass

    def brighten(self, x):
        return x * self.K_GAIN if x > self.K_SIGNAL_THRESHOLD else x * self.K_DECAY

    def create_equalized_image(self, image):

        # Find the value median.
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        v_median = np.median(hsv_image[:, :, 2])

        # Find the saturation mode.
        s_only = hsv_image[:, :, 1]
        s_bins = np.histogram(s_only, bins=50)
        s_mode_index = np.argmax(s_bins[0], axis=0)
        s_mode = s_bins[1][s_mode_index]

        # Prepare the replacement image.
        # v_func = np.vectorize(self.brighten)
        delta_mask = np.zeros_like(hsv_image)
        delta_mask = delta_mask.astype(np.float16)

        # Red channel is the value delta.
        delta_mask[:, :, 0] = abs(hsv_image[:, :, 2] - v_median)
        max_delta = np.max(delta_mask[:, :, 0])
        max_delta = 1 / max_delta if max_delta > 0 else 0
        delta_mask[:, :, 0] *= max_delta

        # Green channel is the saturation delta.
        delta_mask[:, :, 1] = abs(hsv_image[:, :, 1] - s_mode)
        max_delta = np.max(delta_mask[:, :, 1])
        max_delta = 1 / max_delta if max_delta > 0 else 0
        delta_mask[:, :, 1] *= max_delta

        # Combine Masks
        delta_mask[:, :, 0] = np.maximum(delta_mask[:, :, 0], delta_mask[:, :, 1])
        delta_mask[:, :, 1] = delta_mask[:, :, 0]
        delta_mask[:, :, 2] = delta_mask[:, :, 0]

        # Increase signal strength.
        # delta_mask = v_func(delta_mask)

        # Clip and convert back to uint8.
        delta_mask = np.clip(delta_mask, 0, 1)
        delta_mask *= 255
        delta_mask = delta_mask.astype(np.uint8)

        return delta_mask
