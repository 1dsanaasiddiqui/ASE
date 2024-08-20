"""
A file to use parametrised abstract network to attempt verification for a given
network and property
"""
import logging as l
import os
import os.path
import math
import itertools

import torch
import numpy as np
from numpy import random

import parametrized_abstract_network
import train_loop
import interval_propagation
import config
import utils
from marabou_query import marabou_query 



# Auto calcs
_log_cex_loss_zx = config.LOG_CEX_LOSS_ZPOINT
_log_cex_loss_zy = -math.log( config.LOG_CEX_LOSS_ZPOINT )
def loss_c_fn( pred, target, log_loss_enable = config.LOG_CEX_LOSS_ENABLE ):
    # TODO Think about smooth minimum with 0 => no cex property

    if config.DEBUG:
        l.debug("Preds are: {}".format( pred ))

    relu_op = torch.nn.functional.relu(pred)
    loss_vals = torch.amin( relu_op, dim = -1 )
    loss_c = torch.sum(loss_vals)/pred.size()[0]

    if log_loss_enable:
        loss_c = torch.log( loss_c + _log_cex_loss_zx ) + _log_cex_loss_zy

    if config.DEBUG and loss_c > 1e7:
        bad_idx = torch.nonzero( loss_vals > 1e7, as_tuple = True )[0][0]
        l.debug("loss_val: {}".format( loss_vals[bad_idx] ))
        l.debug("relu_op: {}".format( relu_op[bad_idx] ))
        #assert False

    return loss_c


def verify_net( net, prop, n_abs, num_steps,
        lr_init = config.LR_INIT,
        lr_backoff = config.LR_BACKOFF,
        dump_trained = True,
        dump_fname = None,
        load_dump = None,
        plots_fname_pfx = None, 
        call_solver = True,
        show_plot = True,
        ):
    """
    Attempts to verify the network by learning an abstract network

    Arguments:
    
    net             -   Instance of Network to verify
    prop            -   A structure similar to a .prop file to verify `net`
                        against
    n_abs           -   Number of abstract nodes per layer in the attempted
                        abstract network
    num_steps       -   Number of steps to attempt training for
    lr_init         -   The initial learning rate to start training with.
    lr_backoff      -   List of pair of values, representing how lr is backed
                        off as training progresses. The first value is the loss
                        value at which the learning rate change will be
                        triggered, and the second value is the new learning
                        rate.
    dump_trained    -   Dump the trained pre-projection pan
    dump_fname      -   Where to dump the trained pre-projection pan to.
                        Defaults to timestamp.
    load_dump       -   If given, loads dump from given file instead of training.
    plots_fname_pfx -   Prefix to store plots at, optional
    call_solver     -   Whether to call solver or not, disable for debugging 
    show_plot       -   If true, show final plots, else only dumps plot data

    Returns: EITHER ONE of the following:
    1.  A string 'SAT', and the counterexample, if actual cex found
    2.  A string 'UNK', and the final prediction values for best returned net if
        inconclusive
    """
    # Filename for dumps
    if dump_trained:
        if dump_fname is None:
            time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            dump_fname = os.path.join(
                    config.DUMP_PATH,
                    "on-{}-pre-proj.npz".format( time_str ))
            dump_pfx = os.path.join(
                    config.DUMP_PATH,
                    "on-{}".format( time_str ))
        else:
            dump_pfx = dump_fname
        l.info("Dumps will be stored to fname {} and prefix {}".format(
            dump_fname, dump_pfx ))

    # Seed everything
    rng = np.random.default_rng( seed = config.RNG_SEED )
    torch.manual_seed( seed = config.RNG_SEED )
    if config.DEBUG:
        torch.use_deterministic_algorithms( mode = True )

    # Get encoded network
    encoded_network, ip_bounds = property_encode.encode_property(net, prop)
    ip_lb = [ lb for lb, ub in ip_bounds ]
    ip_ub = [ ub for lb, ub in ip_bounds ]

    # Split network
    splitted_net, inc_dec_vects = split_and_merge.split_net( encoded_network )

    # Perform interval propagation
    ival_props = interval_propagation.get_interval_propagations(
            splitted_net, ip_lb, ip_ub)

    # Init abstract net
    abs_net = parametrized_abstract_network.ParametrizedAbstractNet(
            splitted_net, n_abs, inc_dec_vects,
            ival_props)

    # Load pre-trained params
    if load_dump is not None:
        l.info("Loading params from {}".format( load_dump ))
        abs_net.load_params( load_dump )
        returned_net = abs_net

    # Generate random data
    data_np = config.FLOAT_TYPE_NP( rng.uniform( low=ip_lb, high=ip_ub, 
            size=(args.data_size, abs_net.ip_size )))
    data = torch.from_numpy( data_np )
    stub_tgt = torch.empty( (args.data_size,),  # TODO unbounded local
            dtype = config.FLOAT_TYPE_PYTORCH )
    data_iter = [( data, stub_tgt )]

    # Check if data has actual cex
    enc_outs, _ = encoded_network.eval( data_np )
    sat_mask = np.all( enc_outs > 0, axis = 1 )
    if np.any( sat_mask ):
        l.info("Found actual cex from samples")
        sat_idx = np.nonzero( sat_mask )[0]
        cex = data_np[ sat_idx, : ]
        l.info("Cex: {}\n Output: {}\n Encoded: {}".format( 
            cex, net.eval( cex )[0], enc_outs[ sat_idx, : ] ))
        return 'SAT', cex

    # Run training loop 
    if num_steps > 0:
        net_cex = train_loop.train_abs_net( 
            abs_net, data_iter, num_steps, loss_c_fn,
            plotter_fname = plots_fname_pfx,
            lr_init = lr_init,
            lr_backoff = lr_backoff,
            dump_pfx = dump_pfx,
            show_plot = show_plot
        )  
    
        l.info("Choosing best net obtained from step {}".format(
            net_cex.step )) # pytype: disable=attribute-error
        returned_net = net_cex.abs_net # pytype: disable=attribute-error    

    # Dump result of training
    if dump_trained:
        l.info("Dumping pre-projection to {}".format( dump_fname ))
        returned_net.dump_params( dump_fname )
    
    abs_net_verify = returned_net.project_get_net_residuals()
    l.info("Projection done")

    if call_solver:
        l.info("Calling solver")
        cex = marabou_query(abs_net_verify, ip_bounds, 0)
        if config.DEBUG:
            if(cex is not None):
                l.debug("Cex, out: {},{}".format( cex, abs_net_verify.eval(cex)[0] ))
            else:
                l.debug("Cex is none")
            
    # Return avg preds
    avg_preds = loss_c_fn( abs_net_verify.eval( data )[0], stub_tgt, 
            log_loss_enable = False )
    return 'UNK', avg_preds
    

##     if config.DEBUG:   
##         #l.debug(encoded_network.eval(np.array([0.639928936958313,0.0, 0.0, 0.4749999940395355, -0.4749999940395355]))[0])
##         l.debug(returned_net(torch.tensor([0.639928936958313,0.0, 0.0, 0.4749999940395355, -0.4749999940395355])))
##         l.debug(abs_net_verify.eval(np.array([0.639928936958313,0.0, 0.0, 0.4749999940395355, -0.4749999940395355]))[0])
#    l.info(encoded_network.eval(np.array([0.639928936958313,0.0, 0.0, 0.4749999940395355, -0.4749999940395355]))[0])
#    l.info("Doing residuals projection")
#    return cex
##     abs_net_verify = returned_net.project_get_net_residuals() 
#
##     marabou_query(abs_net_verify, ip_bounds, 0)
#




    

if __name__ == "__main__":
    """
    Basic test with an .nnet and .prop file that attempts to train a
    parametrized abstract network with given settings.
    """

    import split_and_merge
    from datetime import datetime
    import sys
    import argparse

    import network
    import property_encode
    import sound_loss_proj
    import min_max_ls
    import train_loop
    
    utils.init_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--networks',
            help = "Networks to load, should be .nnet file dir with only .nnet files",)
    parser.add_argument('-p', '--properties',
            help = "Property to load, should be .prop file dir with only .prop files",)
    parser.add_argument('-a', '--abs-nodes', type = int,
            help = "Number of abstract nodes per layer", )
    parser.add_argument('-s', '--steps', type = int,
            help = "Number of steps to train for", )
    parser.add_argument('-d', '--data-size', type = int, default = 1000,
            help = "Number of data points to generate", )
    parser.add_argument('--load-dump',
            required = False, type = str, default = None,
            help = "If given, loads dump from here instead of training",)
    parser.add_argument('--table-file',
            default = "table.txt",
            help = "Table where resutls will go to",)
    parser.add_argument('--call-solver', default = True, type = eval,
            help = "Call solver?")
    parser.add_argument('--show-plots', default = True, type = eval,
            help = "Show plots?")
    parser.add_argument('-l', '--lr-program', type = str, 
            default = "{},{}".format( config.LR_INIT, config.LR_BACKOFF ),
            help = "If given, sets the learning program. Should be a string with two comma seperated elements, the first being initial lr, the second being a lr backoff list as documeted in train_abs_net()",
    )
    args = parser.parse_args()
    l.info("Args are {}".format( args ))
    
    # Get lrs
    lr_init, lr_backoff = eval( args.lr_program )

    # Prepare list of nets and props
    if os.path.isdir( args.networks ):
        l.info("Networks is a dir")
        nets = [ os.path.join( args.networks, f ) 
                for f in os.listdir( args.networks ) 
                if os.path.isfile( os.path.join( args.networks, f )) ]
    else:
        nets = [ args.networks ]
    if os.path.isdir( args.properties ):
        l.info("Properties is a dir")
        props = [ os.path.join( args.properties, f ) 
                for f in os.listdir( args.properties ) 
                if os.path.isfile( os.path.join( args.properties, f )) ]
    else:
        props = [ args.properties ]
    net_prop_pairs = list( itertools.product( nets, props ))
    if config.DEBUG:
        l.debug("Nets, props: {},{}".format( nets, props ))
    l.info("Net prop pairs: {}".format( net_prop_pairs ))

    # Table header
    with open( args.table_file, 'at' ) as f:
        f.write( "Run From {}\n".format(
            datetime.now().strftime("%Y-%m-%d_%H-%M-%S") ))
        f.write("Network\t Property\n Result\n")

    # Load network
    for net_fname, prop_fname in net_prop_pairs:
        l.info("Netname {} propname {}".format( net_fname, prop_fname ))

        if net_fname.endswith( 'nnet' ):
            net = network.load_nnet( net_fname )
        elif net_fname.endswith('.onnx'):
            net = network.load_onnx( net_fname )

        # Load property
        with open( prop_fname, 'r' ) as f:
            prop = eval( f.read() )

        # Get plot prefix and dump fname
        time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        plotter_pfx = os.path.join(
                config.PLOT_PATH,
                'plot_net_{}_prop_{}_abs_neurons_{}_on_{}.plot'.format( 
                    os.path.basename( net_fname ), 
                    os.path.basename( prop_fname ), 
                    args.abs_nodes, time_str ))
        dump_fname = os.path.join(
                config.DUMP_PATH,
                'dump_net_{}_prop_{}_abs_neurons_{}_on_{}'.format( 
                    os.path.basename( net_fname ), 
                    os.path.basename( prop_fname ), 
                    args.abs_nodes, time_str ))

        # Call
        res = verify_net( net, prop, args.abs_nodes, args.steps,
                lr_init = lr_init,
                lr_backoff = lr_backoff,
                load_dump = args.load_dump,
                dump_fname = dump_fname,
                plots_fname_pfx = plotter_pfx,
                call_solver = args.call_solver,
                show_plot = args.show_plots,
        )
        l.info("Verification result: {}".format( res ))

        with open( args.table_file, 'at' ) as f:
            f.write( "{}\t {}\t {}\n".format( net_fname, prop_fname, res ))

