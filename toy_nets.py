"""
Script to run training on a bunch of toy networks.

First three parameters are:
    1.  Which network to use
    2.  Number of abstract nodes
    3.  Number of epochs to train for
"""
import sys
from datetime import datetime
import logging as l
import argparse

import numpy as np
import torch

import verify_net
import parametrized_abstract_network
import train_loop
import sound_loss_proj
import min_max_ls
import network
import property_encode
import split_and_merge
import utils
import interval_propagation
import config



utils.init_logging()

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--test-no', type = int,
        help = "Test number to load",)
parser.add_argument('-a', '--abs-nodes', type = int,
        help = "Number of abstract nodes per layer", )
parser.add_argument('-e', '--epochs', type = int,
        help = "Number of epochs to train for", )
parser.add_argument('-d', '--data', type = int,
        help = "Number of data points in L_C", 
        default = 1000,)
parser.add_argument('--ls-calculator', type = str,
        help = "Class to use to calculate ls",
        default = 'sound_loss_proj.SoundnessLossCalculator',
        choices = [
            'sound_loss_proj.SoundnessLossCalculator',
            'min_max_ls.MinMaxLossCalculator'
        ],)
args = parser.parse_args()

# Seed everything
rng = np.random.default_rng( seed = config.RNG_SEED )
torch.manual_seed( seed = config.RNG_SEED )
if config.DEBUG:
    torch.use_deterministic_algorithms( mode = True )

tst_no = args.test_no
n_abs = args.abs_nodes
num_epochs = args.epochs
num_data = args.data
ls_calc_class = eval( args.ls_calculator )

plotter_fname = 'plots/test_{}_with_ls_calc_{}_abs_neurons_{}_epoch_{}_on_{}.plot'.format( 
    tst_no, args.ls_calculator, n_abs, num_epochs, datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
)

if tst_no == 0:
    """
    Network from slides, with alpha values multiplied in.
    """

    net = network.Network(
        weights = [
            np.array([  
                [ 3., 100., 102. ],
                [ 2., 102., 100. ],
            ]),
            np.array([  
                [3.], 
                [1.], 
                [0.02],
            ]),
        ],
        biases = [
            np.array([ 0., 0., 0. ]),
            np.array([0.]),
        ],
        end_relu = False
    )

    prop = {
        'input' : [
            (0, {'Lower': -1, 'Upper': 1}),
            (1, {'Lower': -1, 'Upper': 1}),
        ],
        'output' : [
            ( [(1., 0)], {'Lower': 300} ),
        ]
    }

elif tst_no == 1:
    """
    Bigger version of network from slides, with alpha values multiplied in.
    """
    
    net = network.Network(
        weights = [
            np.array([  
                [ 002.    , 002.    ],
                [ 002.    , 002.    ],
                [ 002.    , 003.    ],
                [ 003.    , 002.    ],
                [ 003.    , 003.    ],
                [ 003.    , 003.    ],
                [ 002.    , 003.    ],
                [ 003.    , 002.    ],
                [ 003.    , 003.    ],
                [ 003.    , 003.    ],
                [ 052.    , 052.    ],
                [ 052.    , 052.    ],
                [ 052.    , 053.    ],
                [ 053.    , 052.    ],
                [ 053.    , 053.    ],
                [ 053.    , 053.    ],
                [ 052.    , 053.    ],
                [ 053.    , 052.    ],
                [ 053.    , 053.    ],
                [ 053.    , 053.    ],
                [ 102.    , 102.    ],
                [ 102.    , 102.    ],
                [ 102.    , 103.    ],
                [ 103.    , 102.    ],
                [ 103.    , 103.    ],
                [ 103.    , 103.    ],
                [ 102.    , 103.    ],
                [ 103.    , 102.    ],
                [ 103.    , 103.    ],
                [ 103.    , 103.    ],
                [ 152.    , 152.    ],
                [ 152.    , 152.    ],
                [ 152.    , 153.    ],
                [ 153.    , 152.    ],
                [ 153.    , 153.    ],
                [ 153.    , 153.    ],
                [ 152.    , 153.    ],
                [ 153.    , 152.    ],
                [ 153.    , 153.    ],
                [ 153.    , 153.    ],
                [ 152.    , 152.    ],
                [ 202.    , 202.    ],
                [ 202.    , 203.    ],
                [ 203.    , 202.    ],
                [ 203.    , 203.    ],
                [ 203.    , 203.    ],
                [ 202.    , 203.    ],
                [ 203.    , 202.    ],
                [ 203.    , 203.    ],
                [ 203.    , 203.    ],
                [ 202.    , 202.    ],
                [ 202.    , 202.    ],
                [ 252.    , 253.    ],
                [ 253.    , 252.    ],
                [ 253.    , 253.    ],
                [ 253.    , 253.    ],
                [ 252.    , 253.    ],
                [ 253.    , 252.    ],
                [ 253.    , 253.    ],
                [ 253.    , 253.    ],
                [ 252.    , 252.    ],
                [ 252.    , 252.    ],
                [ 302.    , 303.    ],
                [ 303.    , 302.    ],
                [ 303.    , 303.    ],
                [ 303.    , 303.    ],
                [ 302.    , 303.    ],
                [ 303.    , 302.    ],
                [ 303.    , 303.    ],
                [ 303.    , 303.    ],
                [ 302.    , 302.    ],
                [ 302.    , 302.    ],
                [ 352.    , 353.    ],
                [ 353.    , 352.    ],
                [ 353.    , 353.    ],
                [ 353.    , 353.    ],
                [ 352.    , 353.    ],
                [ 353.    , 352.    ],
                [ 353.    , 353.    ],
                [ 353.    , 353.    ],
                [ 352.    , 352.    ],
                [ 352.    , 352.    ],
                [ 402.    , 403.    ],
                [ 403.    , 402.    ],
                [ 403.    , 403.    ],
                [ 403.    , 403.    ],
                [ 402.    , 403.    ],
                [ 403.    , 402.    ],
                [ 403.    , 403.    ],
                [ 403.    , 403.    ],
                [ 402.    , 402.    ],
                [ 402.    , 402.    ],
                [ 452.    , 453.    ],
                [ 453.    , 452.    ],
                [ 453.    , 453.    ],
                [ 453.    , 453.    ],
                [ 452.    , 453.    ],
                [ 453.    , 452.    ],
                [ 453.    , 453.    ],
                [ 453.    , 453.    ],
            ]).T,
            np.array([  
                [1.],
                [1.],
                [1.],
                [1.],
                [1.],
                [1.],
                [1.],
                [1.],
                [1.],
                [1.],
                [1.],
                [1.],
                [1.],
                [1.],
                [1.],
                [1.],
                [1.],
                [1.],
                [1.],
                [1.],
                [1.],
                [1.],
                [1.],
                [1.],
                [1.],
                [1.],
                [1.],
                [1.],
                [1.],
                [1.],
                [1.],
                [1.],
                [1.],
                [1.],
                [1.],
                [1.],
                [1.],
                [1.],
                [1.],
                [1.],
                [1.],
                [1.],
                [1.],
                [1.],
                [1.],
                [1.],
                [1.],
                [1.],
                [1.],
                [1.],
                [1.],
                [1.],
                [1.],
                [1.],
                [1.],
                [1.],
                [1.],
                [1.],
                [1.],
                [1.],
                [1.],
                [1.],
                [1.],
                [1.],
                [1.],
                [1.],
                [1.],
                [1.],
                [1.],
                [1.],
                [1.],
                [1.],
                [1.],
                [1.],
                [1.],
                [1.],
                [1.],
                [1.],
                [1.],
                [1.],
                [1.],
                [1.],
                [1.],
                [1.],
                [1.],
                [1.],
                [1.],
                [1.],
                [1.],
                [1.],
                [1.],
                [1.],
                [1.],
                [1.],
                [1.],
                [1.],
                [1.],
                [1.],
                [1.],
                [1.],
            ]),
        ],
        biases = [
            np.zeros( (100,) ),
            np.array([0.]),
        ],
        end_relu = False
    )

    prop = {
        'input' : [
            (0, {'Lower': -1, 'Upper': 1}),
            (1, {'Lower': -1, 'Upper': 1}),
        ],
        'output' : [
            ( [(1., 0)], {'Lower': 2250} ),
        ]
    }

elif tst_no == 2:
    """
    Bigger version of network from slides, with alpha values multiplied in.
    """
    
    net = network.Network(
        weights = [
            np.array([  
                [ 002.    , 003.    ],
                [ 003.    , 002.    ],
                [ 102.    , 103.    ],
                [ 103.    , 102.    ],
                [ 202.    , 203.    ],
                [ 203.    , 202.    ],
                [ 302.    , 303.    ],
                [ 303.    , 302.    ],
                [ 402.    , 403.    ],
                [ 403.    , 402.    ],
            ]).T,
            np.array([  
                [1.],
                [1.],
                [1.],
                [1.],
                [1.],
                [1.],
                [1.],
                [1.],
                [1.],
                [1.],
            ]),
        ],
        biases = [
            np.zeros( (10,) ),
            np.array([0.]),
        ],
        end_relu = False
    )

    prop = {
        'input' : [
            (0, {'Lower': -1, 'Upper': 1}),
            (1, {'Lower': -1, 'Upper': 1}),
        ],
        'output' : [
            ( [(1., 0)], {'Lower': 4500} ),
        ]
    }

else:
    l.error("Unknown test number: {}".format( tst_no ))
    sys.exit( -1 )
    
# Common code for all test cases
encoded_network, ip_bounds = property_encode.encode_property(net, prop)
splitted_net, inc_dec_vects = split_and_merge.split_net( encoded_network)

lb_ip = [ l for l,u in ip_bounds ]
ub_ip = [ u for l,u in ip_bounds ]
ival_props = interval_propagation.get_interval_propagations( 
        splitted_net, lb_ip, ub_ip )

abs_net = parametrized_abstract_network.ParametrizedAbstractNet(
        splitted_net, n_abs, inc_dec_vects, ival_props,
        #init_w_min = 110., init_w_max = 150., 
        ls_calc_class = ls_calc_class, )

# Generate random data
data = rng.uniform(low=lb_ip, high=ub_ip, size=(num_data, abs_net.ip_size))
data = torch.from_numpy(config.FLOAT_TYPE_NP(data))

(
    best_snd_save,
    best_cex_save,
) = train_loop.train_abs_net(
        abs_net,
        data,
        num_epochs, 
        verify_net.loss_c_fn,
        plotter_fname = plotter_fname,
)

# Print lambdas after training
l.info("For abs_net")
l.info("Lambdas after training: {}".format( abs_net.lam ))
l.info("Fixed lambdas: {}".format( abs_net.lam_fix_abs_cnc ))
l.info("Weights after training: {}".format( abs_net.w_prm))
l.info("Biases  after training: {}".format( abs_net.b_prm))
#l.info("Concrete weights: {}".format( abs_net.w_org ))
#l.info("Concrete biases: {}".format( abs_net.b_org ))
l.info("Inc dec vects: {}".format( abs_net.inc_dec_vects ))

# pytype: disable=attribute-error

l.info("For best snd at epoch {} (loss was snd {}, cex {} total {})".format( 
    best_snd_save.epoch, best_snd_save.snd_loss, best_snd_save.cex_loss, 
    best_snd_save.tot_loss))
l.info("Lambdas after training: {}".format( best_snd_save.abs_net.lam ))
l.info("Fixed lambdas: {}".format( best_snd_save.abs_net.lam_fix_abs_cnc ))
l.info("Weights after training: {}".format( best_snd_save.abs_net.w_prm))
l.info("Biases  after training: {}".format( best_snd_save.abs_net.b_prm))

l.info("For best cex at epoch {} (loss was snd {}, cex {} total {})".format( 
    best_cex_save.epoch, best_cex_save.snd_loss, best_cex_save.cex_loss, 
    best_cex_save.tot_loss))
l.info("Lambdas after training: {}".format( best_cex_save.abs_net.lam ))
l.info("Fixed lambdas: {}".format( best_cex_save.abs_net.lam_fix_abs_cnc ))
l.info("Weights after training: {}".format( best_cex_save.abs_net.w_prm))
l.info("Biases  after training: {}".format( best_cex_save.abs_net.b_prm))

#l.info("For best tot at epoch {} (loss was snd {}, cex {} total {})".format( 
#    best_tot_save.epoch, best_tot_save.snd_loss, best_tot_save.cex_loss, 
#    best_tot_save.tot_loss))
#l.info("Lambdas after training: {}".format( best_tot_save.abs_net.lam ))
#l.info("Fixed lambdas: {}".format( best_tot_save.abs_net.lam_fix_abs_cnc ))
#l.info("Weights after training: {}".format( best_tot_save.abs_net.w_prm))
#l.info("Biases  after training: {}".format( best_tot_save.abs_net.b_prm))

# pytype: enable=attribute-error
