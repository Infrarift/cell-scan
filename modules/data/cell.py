# -*- coding: utf-8 -*-

"""
cell | cell-scan | 3/06/18
<ENTER DESCRIPTION HERE>
"""

__author__ = "Jakrin Juangbhanich"
__email__ = "juangbhanich.k@gmail.com"


class Cell:

    NEUTRAL = 0
    POSITIVE = 1
    NEGATIVE = 2

    def __init__(self):
        self.image_path = None
        self.image = None
        self.type = self.NEUTRAL
        self.mask = None
        self.region = None

