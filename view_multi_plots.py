"""
A script to view multiple plots. Does not support showing veritcal sep lines.
"""
import sys
import logging as l

import matplotlib
from matplotlib import pyplot as plt

import utils

utils.init_logging()

l.info("Using matplotlib backend: {}".format( matplotlib.get_backend() ))

for fno, fname in enumerate( sys.argv[1:] ):
    l.info("Plotting file {}".format( fno ))
    plotter = utils.Plotter( fname = fname, load = True )
    for plot_str, vals in plotter.data_dict.items():
        plt.plot( vals, label = "{}: {}".format( fno, plot_str ))

plt.legend()
plt.show()
