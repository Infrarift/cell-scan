#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
run_trainer | cell-scan | 3/06/18
The trainer is about loading the labelled samples, visualizing them, and preparing them for training into our models.
"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import argparse
import os

from modules.ai.single_cell_classifier.scc_trainer import SCCTrainer
from tools.util.logger import Logger

__author__ = "Jakrin Juangbhanich"
__email__ = "juangbhanich.k@gmail.com"
__version__ = "0.0.1"

parser = argparse.ArgumentParser(description='<Script Info>')
parser.add_argument('-i', '--input', default='data', type=str, help="<help>")
parser.add_argument('-f', '--flag', action="store_true", help="<help>")
args = parser.parse_args()

if __name__ == "__main__":

    input_path = os.path.join("input", args.input)

    Logger.log_header("Running Cell-Scan Trainer", with_gap=True)
    Logger.log_field("Version", __version__)
    Logger.log_field("Input Folder", input_path)

    scc_trainer = SCCTrainer()
    scc_trainer.process(input_path)
