# -*- coding: utf-8 -*-

"""
reporter | cell-scan | 4/06/18
<ENTER DESCRIPTION HERE>
"""
import math
import os
from typing import List

import cv2
import numpy as np
from modules.data.cell import Cell
from modules.data.slide import Slide
from tools.util.region import Region

__author__ = "Jakrin Juangbhanich"
__email__ = "juangbhanich.k@gmail.com"


class Reporter:
    def __init__(self):
        self.edge = 30
        self.pad = 10
        self.border = 2

        # self.slide_width = 500
        self.content_width = 1024
        self.report_width = self.content_width + self.edge * 2
        self.cell_columns = 12
        self.cell_size = int((self.content_width - (self.cell_columns - 1) * self.pad) / self.cell_columns)
        self.analysis_bar_height = 36

        self.report_bg_color = (50, 40, 30)
        pass

    def produce(self, slide: Slide, output_path: str) -> None:
        """ Produce and size a report for these cells and image. """

        # Create the header.
        header = self._create_header(100)

        # Create the slides.
        slide_images = self._create_slide_images(slide)
        slide_size = slide_images[0].shape[0]

        # Create the cell images.
        cell_images = self._create_cell_images(slide)
        cells_size = int(math.ceil(len(cell_images) / self.cell_columns)) * (cell_images[0].shape[0] + self.pad)

        analysis_bar = self._create_analysis_bar(slide.purity)

        # Create the report.
        report_height = header.shape[0] + slide_size + cells_size + \
                        self.analysis_bar_height + self.pad * 2 + self.edge * 2

        report = np.zeros((report_height, self.report_width, 3), dtype=np.uint8)
        report[:, :] = self.report_bg_color

        # Join the elements.
        self._composite(report, [header], x=0, y=0)

        slide_y = header.shape[0] + self.edge
        self._composite(report, slide_images, x=self.edge, y=slide_y, p=self.pad)

        analysis_y = slide_y + slide_size + self.pad
        self._composite(report, [analysis_bar], x=self.edge, y=analysis_y)

        cell_y = analysis_y + self.analysis_bar_height + self.pad
        self._composite(report, cell_images, x=self.edge, y=cell_y, p=self.pad, c=self.cell_columns)

        # Save the report.
        report_path = os.path.join(slide.slide_path, "report.png")
        cv2.imwrite(report_path, report)

        if output_path is not None:
            report_path = os.path.join(output_path, "{}_report.png".format(slide.name))
            cv2.imwrite(report_path, report)

    def _create_analysis_bar(self, purity: float):
        image = np.zeros((self.analysis_bar_height, self.content_width, 3), dtype=np.uint8)
        t_height = 28

        tag = "DANGER"
        p_color = (0, 50, 255)

        if purity > 0.5:
            tag = "CAUTION"
            p_color = (0, 150, 255)

        if purity > 0.8:
            tag = "SAFE"
            p_color = (255, 255, 255)

        label = "PURITY: {:.0f}% ({})".format(purity * 100, tag)

        self._create_text_box(image, label, x=self.content_width//2, y=t_height//2, width=300, height=t_height,
                              color=p_color)

        # Draw Bar
        b_pad = 4
        b_width_max = self.content_width - b_pad * 2
        b_height = 2
        b_p_x = b_pad
        b_p_width = int(b_width_max * purity)
        b_n_width = b_width_max - b_p_width
        b_n_x = b_p_x + b_p_width

        b_y_start = self.analysis_bar_height - b_height - b_pad - 1
        b_y_end = b_y_start + b_height

        cv2.rectangle(image, (b_n_x, b_y_start), (b_n_x + b_n_width, b_y_end), color=(0, 0, 255), thickness=-1)
        cv2.rectangle(image, (b_p_x, b_y_start), (b_p_x + b_p_width, b_y_end), color=(0, 255, 0), thickness=-1)
        return image

    def _create_cell_images(self, slide: Slide):
        cells = slide.cells
        images = []
        n_cells = len(cells)

        num_pad = 4
        num_size = 22
        nx = num_size // 2 + num_pad

        green = (0, 255, 0)
        red = (0, 0, 255)
        grey = (255, 255, 255)

        for i in range(n_cells):
            cell = cells[i]
            image = cv2.resize(cell.image, (self.cell_size, self.cell_size))
            self._create_text_box(image, str(i + 1), nx, nx, num_size, num_size, centered=True)

            cell_color = grey
            if cell.type == Cell.POSITIVE:
                cell_color = green

            if cell.type == Cell.NEGATIVE:
                cell_color = red

            cv2.rectangle(image, (0, 0), (self.cell_size, self.cell_size), color=cell_color, thickness=4)
            images.append(image)
        return images

    def _create_slide_images(self, slide: Slide):

        n_images = 2
        image_size = int((self.content_width - (n_images - 1) * self.pad) / n_images)
        t_height = 24
        t_pad = 2

        original_image = cv2.resize(np.copy(slide.image), (image_size, image_size))
        self._create_text_box(original_image, "ORIGINAL",
                              x=image_size // 2,
                              y=t_height // 2 + t_pad,
                              width=image_size - 2 * t_pad,
                              height=t_height)

        analysis_image = self._create_labelled_pos_neg_scan(slide, image_size)
        self._create_text_box(analysis_image, "ANALYSIS",
                              x=image_size // 2,
                              y=t_height//2 + t_pad,
                              width=image_size - 2 * t_pad,
                              height=t_height)

        images = [original_image, analysis_image]

        return images

    def _create_header(self, height=100) -> np.array:
        header = cv2.imread("cell_logo.png")
        # header = np.zeros((height, self.report_width, 3), dtype=np.uint8)
        # header[:, :] = (255, 0, 0)
        return header

    def _composite(self, base: np.array, elements: List[np.array], x: int, y: int, p: int=0, c: int=99):
        """
        Composite the element image into the base image at location x, y (left, top).
            p: Padding between each element
            c: Max number of columns.
        """

        n_elements = len(elements)

        for i in range(n_elements):

            element = elements[i]
            nx = x + (i % c) * (element.shape[0] + p)
            ny = y + math.floor(i/c) * (element.shape[0] + p)

            left = nx
            top = ny
            right = left + element.shape[1]
            bottom = top + element.shape[0]

            base[top:bottom, left:right] = element

    def _create_labelled_pos_neg_scan(self, slide: Slide, target_size: int):

        image = np.copy(slide.image)
        f = target_size / image.shape[0]  # Image size factor.

        image = cv2.resize(image, (target_size, target_size))
        mask = np.zeros_like(image)

        n_cells = len(slide.cells)
        red = [0, 0, 255]
        green = [0, 255, 0]

        for i in range(n_cells):
            cell = slide.cells[i]
            if cell.type == Cell.NEUTRAL:
                continue

            cell_color = red
            if cell.type == Cell.POSITIVE:
                cell_color = green

            r: Region = cell.region.clone()
            r.set_rect(
                left=int(f * r.left),
                right=int(f * r.right),
                top=int(f * r.top),
                bottom=int(f * r.bottom))

            pad = 5
            cv2.rectangle(image, (r.left - pad, r.top - pad), (r.right + pad, r.bottom + pad),
                          color=cell_color, thickness=2)
            c_mask_resize = cell.mask.astype(np.uint8)
            c_mask_resize = cv2.resize(c_mask_resize, (target_size, target_size))
            c_mask_resize = c_mask_resize[:, :] > 0
            mask[c_mask_resize] = np.array(cell_color) // 3

            # Find and draw the contour as well
            c_mask = np.zeros((image.shape[0], image.shape[1], 1), dtype=np.uint8)
            c_mask[c_mask_resize] = 255
            _, contours, _ = cv2.findContours(c_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            tam = 0
            main_contour = None

            for contour in contours:
                if len(contour) > tam:
                    main_contour = contour
                    tam = len(contour)

            if main_contour is not None:
                cv2.drawContours(mask, main_contour.astype('int'), -1, cell_color, 2)

        image = cv2.addWeighted(image, 1.0, mask, 0.5, 0.0)

        for i in range(n_cells):
            cell = slide.cells[i]
            if cell.type == Cell.NEUTRAL:
                continue

            r: Region = cell.region.clone()
            r.set_rect(
                left=int(f * r.left),
                right=int(f * r.right),
                top=int(f * r.top),
                bottom=int(f * r.bottom))
            self._create_text_box(image, str(i + 1), r.x, r.y, 12, 12, circular=True)

        return image

    @staticmethod
    def _create_text_box(image, text, x, y, width, height, color=(255, 255, 255), bg_color=(0, 0, 0), centered=True,
                         right=False, circular=False):
        """ Create a text box, with the text written in the center."""

        # Find the corners of the box that we want to draw.
        if centered:
            bot_left = (x - width // 2, y + height // 2)
            top_right = (x + width // 2, y - height // 2)
        else:
            bot_left = (x, y + height)
            top_right = (x + width, y)

        # Create a black BG plate for the text.
        if bg_color is not None:
            if circular:
                cv2.circle(image, (x, y), width, color=bg_color, thickness=-1)
            else:
                cv2.rectangle(image, bot_left, top_right, color=bg_color, thickness=-1)

        # Assign the font and get the boundary of the text.
        font = cv2.FONT_HERSHEY_PLAIN
        text_size = cv2.getTextSize(text, font, 1, 1)[0]

        # Get the text co-ordinates based on the boundary.
        if centered:
            tx = (width - text_size[0]) // 2 + (x - width // 2)
            ty = (height + text_size[1]) // 2 + (y - height // 2)
        else:
            if right:
                tx = top_right[0] - text_size[0] - 10
                ty = (height + text_size[1]) // 2 + y
            else:
                tx = x + 10
                ty = (height + text_size[1]) // 2 + y

        # Add the text, centered to the area.
        cv2.putText(image, text, (tx, ty), font, 1, color)
