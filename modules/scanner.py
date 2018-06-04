# -*- coding: utf-8 -*-

"""
scanner | cell-scan | 3/06/18
<ENTER DESCRIPTION HERE>
"""
import os
import torch
from typing import List
import numpy as np
import cv2

from modules.ai.cell_detector.equalizer import Equalizer
from modules.ai.cell_detector.prediction import Prediction
from modules.ai.single_cell_classifier.single_cell_classifier import SingleCellClassifier
from modules.data.cell import Cell
from modules.data.slide import Slide
from modules.reporter import Reporter
from tools.util import visual, pather
from tools.util.logger import Logger

__author__ = "Jakrin Juangbhanich"
__email__ = "juangbhanich.k@gmail.com"


class Scanner:
    def __init__(self, output_path: str):
        self.output_path: str = output_path
        from modules.ai.cell_detector.cell_net import CellNet

        self.equalizer = Equalizer()
        self.net = CellNet("cell_model")
        self.scc = torch.load("models/scc.pt")
        self.reporter = Reporter()

    def process(self, slides: List[Slide]):
        """ Predict the results, and create a report for each scan. """
        for slide in slides:
            self._process_slide(slide)

    def _process_slide(self, slide: Slide):
        slide_path = os.path.join(self.output_path, slide.name)
        slide.slide_path = slide_path
        os.mkdir(slide_path)

        # Draw the slide image.
        image = slide.image
        image_path = os.path.join(slide_path, "image.png")
        Logger.log_special("Scanning {}".format(slide.name), with_gap=True)

        # Predict the cell masks from an image.
        predict_image = image
        if True:
            equalizer_image = self.equalizer.create_equalized_image(image)
            predict_image = equalizer_image

        # Get the sample prediction.
        prediction = self.net.cycle_predict(predict_image, None)

        slide.cells = self._process_prediction(slide, slide_path, prediction)
        self._draw_prediction_mask(image, slide_path, prediction)

        pather.create("output/summary")
        self.reporter.produce(slide, "output/summary")

        cv2.imwrite(image_path, equalizer_image)

    def _process_prediction(self, slide: Slide, path, prediction: Prediction) -> List[Cell]:
        # Extract an image of each cell.
        n = 0
        pad = 50
        pad_image = cv2.copyMakeBorder(slide.image, pad, pad, pad, pad, cv2.BORDER_REFLECT)
        cells: List[Cell] = []

        for unit in prediction.units:
            r = unit.region.clone()
            max_size = max(r.width, r.height)
            r.width = max_size
            r.height = max_size
            r.x += pad
            r.y += pad
            n += 1
            ex_image = pad_image[r.top:r.bottom, r.left:r.right]
            full_path = os.path.join(path, "{}_cell_{}.png".format(slide.name, n))
            ex_image = cv2.resize(ex_image, (64, 64))
            cv2.imwrite(full_path, ex_image)

            cell = Cell()
            cell.image_path = full_path
            cell.image = ex_image
            cell.mask = unit.mask
            cell.region = unit.region

            # Predict Cell Health
            cell.type = self.scc.process(cell.image)
            cells.append(cell)

        return cells

    def _draw_prediction_mask(self, image, path: str, prediction: Prediction):
        # For each prediction I also want to draw a mask.
        mask = np.zeros_like(image, dtype=np.uint8)

        n_units = len(prediction.units)
        colors = visual.random_colors(n_units)
        pad = 8

        for i in range(n_units):
            unit = prediction.units[i]
            r = unit.region
            cv2.rectangle(image, (r.left - pad, r.top - pad), (r.right + pad, r.bottom + pad),
                          color=colors[i], thickness=2)
            mask[unit.mask] = colors[i] // 3

            # Find and draw the contour as well
            c_mask = np.zeros((image.shape[0], image.shape[1], 1), dtype=np.uint8)
            c_mask[unit.mask] = 255
            _, contours, _ = cv2.findContours(c_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            tam = 0
            contornoGrande = None

            for contorno in contours:
                if len(contorno) > tam:
                    contornoGrande = contorno
                    tam = len(contorno)

            if contornoGrande is not None:
                cv2.drawContours(mask, contornoGrande.astype('int'), -1, colors[i], 3)

        mask_path = os.path.join(path, "mask.png")
        final = cv2.addWeighted(image, 1.0, mask, 0.5, 0.0)
        cv2.imwrite(mask_path, final)



