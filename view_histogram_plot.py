"""
A simple script to load and view a plot. Just pass the name.
"""
import sys

import utils


fname = sys.argv[1]
utils.HistogramPlotter( fname = fname, load = True ).show( show = True, save = False )
