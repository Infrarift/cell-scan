# -*- coding: utf-8 -*-

"""
Here's a stand-alone logger class to help me write good output to the terminal.
"""
from typing import List

from tools.util.region import Region

__author__ = "Jakrin Juangbhanich"
__copyright__ = "Copyright 2018, Jakrin Juangbhanich"
__email__ = "juangbhanich.k@gmail.com"


class PredictionUnit:
    def __init__(self, region, score, mask):
        self.mask = mask
        self.region: Region = region
        self.score = score


class Prediction:
    def __init__(self):
        self.units: List[PredictionUnit] = []
        self.total_mask = None

    def add_unit(self, region, score, mask):
        unit = PredictionUnit(region, score, mask)
        self.units.append(unit)
