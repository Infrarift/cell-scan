#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
run_scanner | cell-scan | 3/06/18
Run the scanner on all images from the targeted input file.
"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import argparse
import os
from typing import List

from modules.data import loader
from modules.data.slide import Slide
from modules.scanner import Scanner
from tools.util import pather
from tools.util.logger import Logger

__author__ = "Jakrin Juangbhanich"
__email__ = "juangbhanich.k@gmail.com"
__version__ = "0.0.0"

parser = argparse.ArgumentParser(description='<Script Info>')
parser.add_argument('-i', '--input', default='test', type=str, help="<help>")
parser.add_argument('-o', '--output', default='batch1', type=str, help="<help>")
parser.add_argument('-f', '--flag', action="store_true", help="<help>")
args = parser.parse_args()

if __name__ == "__main__":

    input_path = os.path.join("input", args.input)
    output_path = os.path.join("output", args.output)
    pather.create(output_path, clear=True)

    Logger.log_header("Running Scanner", with_gap=True)
    Logger.log_field("Version", __version__)
    Logger.log_field("Input Path", input_path)
    Logger.log_field("Input Path", output_path)

    scanner = Scanner()
    slides: List[Slide] = loader.load_testing_slides(input_path)
    scanner.process(slides)

