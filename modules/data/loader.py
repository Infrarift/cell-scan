# -*- coding: utf-8 -*-

"""
loader | cell-scan | 3/06/18
Use this module to load slides, cells, training and testing data for our app.
"""

import os
from typing import List
from modules.data.slide import Slide

__author__ = "Jakrin Juangbhanich"
__email__ = "juangbhanich.k@gmail.com"


def load_testing_slides(path) -> List[Slide]:
    """ Load slides from a directory for testing. """
    slides = []
    for file in os.listdir(path):
        full_path = os.path.join(path, file)
        slide = Slide()
        slide.image_path = full_path
        slides.append(slide)

    return slides


