"""
A simple script to load and view a plot. Just pass the name.
"""
import sys
import logging as l

import matplotlib

import utils

utils.init_logging()

l.info("Using matplotlib backend: {}".format( matplotlib.get_backend() ))

fname = sys.argv[1]
utils.Plotter( fname = fname, load = True ).show( show = True, save = False )
