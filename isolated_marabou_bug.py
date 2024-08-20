
import logging as l
import os.path

import torch
import numpy as np
from numpy import random

import parametrized_abstract_network
import train_loop
import interval_propagation
import config
import utils
from marabou_query import marabou_query 
import network

net = network.load_npz("net_dumps/2024-04-25_19-10-53.npz")

ip_bounds = [( 0.6,   0.6798577687),(  -0.5,   0.5),(  -0.5,   0.5), (  0.45,   0.5), (  -0.5,   -0.45)]

cex = marabou_query(net, ip_bounds, None)

print(cex)
out, _ = net.eval(np.array([0.639928936958313,0.0, 0.0, 0.4749999940395355,-0.4749999940395355]))
print(out)