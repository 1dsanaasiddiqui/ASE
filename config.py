"""
Contains various configurations
"""
import os
from math import exp, sqrt, log
import time
from types import FunctionType

import numpy as np
import torch

# TODO Move to proper config object that can be loaded from files / command line
# args

# Directories. Creates them if they don't exist.
LOG_PATH = "logs/"
os.makedirs(LOG_PATH, exist_ok=True)
PLOT_PATH = "plots/"
os.makedirs(PLOT_PATH, exist_ok=True)
DUMP_PATH = "dumps/"
os.makedirs(DUMP_PATH, exist_ok=True)
DSETS_PATH = "dsets/"
os.makedirs(DSETS_PATH, exist_ok=True)

# Turn on/off debugging, asserts, etc.
ASSERTS = True
DEBUG = True
VISUALIZE = True
TIMING = True
WARN = True
PROFILE = True
SHOW_PLOT = True

# Number of steps after which to save plot
PLOT_SAVE_IVAL = 20

# Intervals after which to print loss. Set negative to disable printing
LOSS_LOG_IVAL = 1

# Lambda visualization: enable, and number of bins for lambda value histograms
LAMBDA_VIZ_ENABLE = True and VISUALIZE
LAMBDA_VIZ_NUM_BINS = 250

# Seed for all rngs. Set to None to pull seed from OS entropy.
RNG_SEED = int(time.monotonic() * 1000) #10 #5 #1
#RNG_SEED = 104653 # A particular seed that may be interesting for alternating

#This is gain in variance due to activation function
WEIGHT_INIT_GAIN = sqrt(2) # TODO this has been set because init is not
                                # working correctly, fix

# Learning rate config. See train_loop.train_abs_net()
LR_INIT = 0.05   
LR_BACKOFF = [ (10, 0.01), (1, 0.001) ] 

# Various types 
FLOAT_TYPE_NP = np.float32          # Type of the floating point values to use
FLOAT_TYPE_PYTORCH = torch.float32  # Type of the floating point values to use
INDEX_TYPE_PYTORCH = torch.long     # Type of variables used to index 

# Various fp constants
FLOATING_POINT_ADJUST = 1e-5 # Small value added to dividend to make sure a/a
                             # <=1
LARGE_LOSS_VAL = 1e5         # Large fp value denoting loss = inf. Should be
                             # used in a way that doesnt affect grads 

# Settings to take loss of cex for training during verify net. 
LOG_CEX_LOSS_ENABLE = True   # Enable taking loss
LOG_CEX_LOSS_ZPOINT = 1e-2 #1 # log graph applied to loss is shifted so that 
                             # (this point, log( this point )) is at new origin

# Scaling factors applied to gradients
W_GRAD_SCALE = 1
B_GRAD_SCALE = 1
LAM_GRAD_SCALE = 1
ALPHA_GRAD_SCALE = 1

# The number of iterations to use to calculate LS per GD iteration. Set 1 to do
# single step (fastest), and -1 to calculate precise LS
NUM_LS_ITERS = 1

# Options for LS calculation
# TODO: Disable LS calc for now due to bug, scale factor coming zero, see
# sound_loss_proj
SCALE_LS = False    # If true, tries to apply some normalizing scaling to LS
ADHOC_LS = True     # Enable adhoc calc where if max diff of abs & conc appears
                    # on active side of both neurons, its taken as ls
CALC_L2_LS = False  # Used to disable l2 calculation. l2 can be disabled if
                    # adhoc is enabled. TODO prove.

# If true, initialize lambda as random, else init to uniform values.
LAMBDA_RANDOM_INIT = False

# Max size of weights as a multiple of original network weights
WB_CLAMP_ENABLE = False
WB_RANGE = 100

# Set default ls calculator
import sound_loss_proj
LS_CALCULATOR = sound_loss_proj.SoundnessLossCalculator

# LS_LC alternation
LS_LC_ALTER_ENABLE = False #True 
LS_LC_ALTER_MIN_ITER = 20       # Minimum and maximum iterations after which to
LS_LC_ALTER_MAX_ITER = 100      # alternate
LS_LC_ALTER_CONST_EPS = 1.0     # If loss does not change more than EPS for NUM
LS_LC_ALTER_CONST_NUM = 10      # alternate
LS_LC_ALTER_VIZ = True
LS_LC_ALTER_VIZ_LS_COL = 'blue' # Colors when transitioning to LS and LC
LS_LC_ALTER_VIZ_LC_COL = 'red' 

# Use this to disable progress bars on certain terminals
SHOW_PBARS = False

FLOAT_TOL = 1e-5 

# Type of net to consider for projection. `net_snd` chooses best soundness loss,
# `net_cex` best counterexample loss.
ABS_NET_TYPE = 'net_cex'

# Min max initialization
MIN_MAX_INIT = False

# Project at each step of gd
PROJECT_WHILE_GD = False

# If true, add residuals before last relu. Does not affect behavior if last
# layer does not have relu.
# NOTE: Setting this to True may cause training to get stuck in a situation 
# where output of the network is almost always zero (for all but critical class,
# residual adjustment pushes value to < 0, which is = 0 after relu).
# On the other hand, if False, during projection an extra layer needs to be
# added to end of network to add residuals. 
ADD_RESIDUAL_BEFORE_LAST_RELU = False

# If set to true, last layers of loaded networks are force-dropped.
FORCE_NO_END_RELU = False
