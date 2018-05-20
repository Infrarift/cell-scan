#!$(which python)
# -*- coding: utf-8 -*-

"""
run_cell_scan | cell-scan | 20/05/18
<ENTER DESCRIPTION HERE>
"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import argparse

from modules.scanner import Scanner

__author__ = "Jakrin Juangbhanich"
__email__ = "juangbhanich.k@gmail.com"
__version__ = "0.0.0"

parser = argparse.ArgumentParser(description='<Script Info>')
parser.add_argument('-i', '--input', default='data', type=str, help="<help>")
parser.add_argument('-f', '--flag', action="store_true", help="<help>")
args = parser.parse_args()

if __name__ == "__main__":
    print("Hello World")
    scanner = Scanner()
    scanner.scan("input/cell_image.jpg", "input/region_data.csv")