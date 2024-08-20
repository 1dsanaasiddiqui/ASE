import numpy as np
import torch
from torch import nn

import config
import logging as l
import network
from network import Network


def get_interval_propagations(orig_net : Network,
                              lower_bound_ip, upper_bound_ip):
    """
    Returns: List of pairs (lb, ub), one for each layer, post-relu. 
    """
    propagated_intervals = []
    lower_bounds = np.array(lower_bound_ip, dtype=config.FLOAT_TYPE_NP)
    upper_bounds = np.array(upper_bound_ip, dtype=config.FLOAT_TYPE_NP)
    propagated_intervals.append((lower_bounds, upper_bounds))
    for i in range(orig_net.num_layers - 1):

        lin_lower_bound = np.where(
                orig_net.weights[i] >= 0,
                np.expand_dims( lower_bounds, -1 ), 
                np.expand_dims( upper_bounds, -1 )
        )
        if config.DEBUG:
            l.debug("shape after where: {}".format( lin_lower_bound.shape ))
        lin_lower_bound = np.sum( 
                lin_lower_bound * orig_net.weights[i], axis = 0 )
        if config.DEBUG:
            l.debug("shape after mult: {}".format( lin_lower_bound.shape ))
        lin_lower_bound += orig_net.biases[i]

        lin_upper_bound = np.where(
                orig_net.weights[i] >= 0,
                np.expand_dims( upper_bounds, -1 ), 
                np.expand_dims( lower_bounds, -1 )
        ) 
        lin_upper_bound = np.sum( 
                lin_upper_bound * orig_net.weights[i], axis = 0 )
        lin_upper_bound += orig_net.biases[i]

        if config.DEBUG:
            l.debug("lin_u,l_bound shape: {} {}".format( 
                lin_lower_bound.shape, lin_upper_bound.shape)) 

        lower_bounds = np.maximum(0, lin_lower_bound) #instead of Relu
        upper_bounds = np.maximum(0, lin_upper_bound) #instead of Relu

        if config.DEBUG:
            l.debug("u,l_bound shape: {} {}".format( 
                lower_bounds.shape, upper_bounds.shape)) 

        propagated_intervals.append((lower_bounds, upper_bounds))
    if config.DEBUG:
        l.debug("Propagated intervals : ")
        l.debug(propagated_intervals)
    return propagated_intervals
