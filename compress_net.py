"""
A script to compress a given network wrt given data with given budget
"""
import logging as l

import torch
import numpy as np
import tqdm

import verify_net
import parametrized_abstract_network
import train_loop
import interval_propagation
import config
import utils
from functools import partial
import os

def false_positive_rate( preds, actual_class ):
    # TODO Measure the actual false positive rate
    
    if config.DEBUG:
        l.debug(" pred, tgt shape {} {}".format( preds.shape, actual_class.shape ))
    
    return torch.nn.functional.cross_entropy( preds, actual_class ) 


def get_accuracy(dset, func, crit_class, num_points = None):
    """
    Utility for measuring accuracy. func should be a function wrapping the model
    taking an input and providing the output of the model as a numpy array. If
    num_points is given, accuracy measurement is stopped after going through
    that many points. num_points is ignored if config.DEBUG is False

    NOTE: This expects dset to be an iterator over img, tgt with batch size 1.
    """
    if not config.DEBUG and num_points is not None:
        l.warning( "Ignoring num_points in non-debug mode" )
        num_points = None
    n_correct = 0
    n_tot = 0 
    n_true_pos = 0
    n_true_neg = 0
    n_pos = 0
    n_false_pos = 0
    n_false_neg = 0
    n_neg = 0
    for point_idx, (img, tgt) in enumerate( tqdm.tqdm( dset )):

        if num_points is not None and n_tot > num_points:
            break

        out = func( img )
        assert out.shape[0] == 1
        assert out.shape[1] == 10
        out = out.flatten()
        
        #if config.DEBUG:
        #    l.debug("Outs: {}".format( out ))

        pred = np.argmax( out )
        n_tot += 1
        n_correct += 1 if pred == tgt else 0            
        pred_pos = pred == crit_class
        tgt_pos = tgt == crit_class
        n_pos += 1 if tgt_pos else 0
        n_neg += 1 if not tgt_pos else 0
        n_true_pos  += 1 if     pred_pos and     tgt_pos else 0
        n_true_neg  += 1 if not pred_pos and not tgt_pos else 0
        n_false_pos += 1 if     pred_pos and not tgt_pos else 0
        n_false_neg += 1 if not pred_pos and     tgt_pos else 0

    l.info( "Accuracy: {}".format( n_correct / n_tot ))
    l.info( "True positive rate {}".format(  n_true_pos / n_pos ))
    l.info( "True negative rate {}".format(  n_true_neg / n_neg ))
    l.info( "False positive rate {}".format( n_false_pos / n_neg))
    l.info( "False negative rate {}".format( n_false_neg / n_neg))


def compress_net( net, crit_class, data_iter, ip_lb, ip_ub, n_abs, num_steps,
        plots_fname_pfx = None, 
        lr_init = config.LR_INIT,
        lr_backoff = config.LR_BACKOFF,
        init_min_max = config.MIN_MAX_INIT,
        project_each_step = config.PROJECT_WHILE_GD,
        dump_trained = True,
        dump_fname = None,
        load_dump = None,
    ):
    """
    Attempts to verify the network by learning an abstract network

    Arguments:
    
    net             -   Instance of Network to verify
    crit_class      -   The critical class
    data_iter       -   An iterator producing batches of input and target data,
                        like a DataLoader.
    ip_l/ub         -   Input bounds for interval analysis
    n_abs           -   Number of abstract nodes per layer in the attempted
                        abstract network
    num_steps       -   Number of steps to attempt training for
    plots_fname_pfx -   Prefix to store plots at, optional
    lr_init         -   The initial learning rate to start training with.
    lr_backoff      -   List of pair of values, representing how lr is backed
                        off as training progresses. The first value is the loss
                        value at which the learning rate change will be
                        triggered, and the second value is the new learning
                        rate.
    init_min_max    -   Initialize to min-max 
    project_each_step-  Project w,b at each step
    dump_trained    -   Dump the trained pre-projection pan
    dump_fname      -   Where to dump the trained pre-projection pan to.
                        Defaults to timestamp.
    load_dump       -   If given, loads dump from given file instead of training.
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
            l.info("Dumps will be stored to fname {} and prefix {}".format(
                dump_fname, dump_pfx ))
    
    # Split network
    out_lyr_idv = [ -1 for _ in range( net.out_size ) ]
    out_lyr_idv[crit_class] = 1
    out_lyr_idv = np.array(out_lyr_idv)
    splitted_net, inc_dec_vects = split_and_merge.split_net( 
            net, out_lyr_idv )

    # Perform interval propagation
    ival_props = interval_propagation.get_interval_propagations(
            splitted_net, ip_lb, ip_ub)

    # Init abstract net
    abs_net = parametrized_abstract_network.ParametrizedAbstractNet(
            splitted_net, n_abs, inc_dec_vects, ival_props, 
            sound_wb_init = init_min_max)

    # Load pre-trained params
    if load_dump is not None:
        l.info("Loading params from {}".format( load_dump ))
        abs_net.load_params( load_dump )
        resultant_net = abs_net

    # Train net
    if num_steps > 0:
        net_cex = train_loop.train_abs_net( 
            abs_net, data_iter, num_steps, false_positive_rate,
            plotter_fname = plots_fname_pfx,
            project_wb = project_each_step,
            dump_pfx = dump_pfx,
            lr_init = lr_init,
            lr_backoff = lr_backoff,
        )  

        # Choose a net.
        l.info("Choosing best net obtained from step {}".format(
            net_cex.step )) # pytype: disable=attribute-error
        resultant_net = net_cex.abs_net # pytype: disable=attribute-error

    if dump_trained:
        l.info("Dumping pre-projection to {}".format( dump_fname ))
        resultant_net.dump_params( dump_fname )
    
    l.info("Accuracy of resultant net:")
    get_accuracy( mnist_test_dloader, 
        (lambda x : 
            resultant_net( x ).numpy( force = True )), 
        args.crit_class,
        num_points = 10 #100 
    )

    l.info("Doing residuals projection")
    abs_net = resultant_net.project_get_net_residuals() 

    return abs_net 


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
    parser.add_argument('-n', '--network',
            required = True,
            help = "Network to load, should be .nnet file",)
    parser.add_argument('-a', '--abs-nodes', type = int,
            required = True,
            help = "Number of abstract nodes per layer", )
    parser.add_argument('-s', '--steps', type = int,
            required = True,
            help = "Number of steps to train for", )
    parser.add_argument('-b', '--batch-size', type = int,
            default = 1000,
            help = "The batch size to train for", )
    parser.add_argument('-d', '--data-set', type=str,
            choices = ['mnist'],
            default = 'mnist',
            help = "Which dataset to compress against", )
    parser.add_argument('-c', '--crit-class', type = int,
            required = True,
            help = "The critical class to maintain soundness for", )
    parser.add_argument('-N', '--data-size', type = int, default = 1000,
            help = "Number of data points to generate", )
    parser.add_argument('--load-dump',
            required = False, type = str, default = None,
            help = "If given, loads dump from here instead of training",)
    parser.add_argument('-l', '--learning-rate-program',
            required = False, type = str, 
            default = "{},{}".format( config.LR_INIT, config.LR_BACKOFF ),
            help = "If given, sets the learning program. Should be a string with two comma seperated elements, the first being initial lr, the second being a lr backoff list as documeted in train_abs_net()",
            )
    parser.add_argument('--force-no-end-relu',
            required = False, type = bool, default = config.FORCE_NO_END_RELU,
            help = "If true, relus at output layer are dropped even if input network has them",)
    args = parser.parse_args()
    l.info("Args are {}".format( args ))
    
    # Seed everything
    rng = np.random.default_rng( seed = config.RNG_SEED )
    torch.manual_seed( seed = config.RNG_SEED )
    if config.DEBUG:
        torch.use_deterministic_algorithms( mode = True )

    # Load network
    if args.network.endswith( 'nnet' ):
        net = network.load_nnet( args.network )
    elif args.network.endswith('.onnx'):
        net = network.load_onnx(args.network)

    # Get lrs
    lr_init, lr_backoff = eval( args.learning_rate_program )

    # Force network to not end with relu
    if args.force_no_end_relu:
        net.end_relu = False

    # Generate data via mnist
    if args.data_set == 'mnist':
        import torchvision
        import torchvision.datasets
        import torchvision.transforms as T

        mnist_train_dset = torchvision.datasets.MNIST( 
                root = config.DSETS_PATH, 
                download = True, train = True, 
                transform = T.Compose([ 
                    T.ToTensor(), 
                    T.Lambda( lambda x: torch.flatten( x, start_dim = 0 ) ) ]) )
        mnist_train_dloader = torch.utils.data.DataLoader(
                mnist_train_dset,
                batch_size = args.batch_size,
                shuffle = True, )
        mnist_test_dset = torchvision.datasets.MNIST( 
                root = config.DSETS_PATH, 
                download = True, train = False, 
                transform = T.Compose([ 
                    T.ToTensor(), 
                    T.Lambda( lambda x: torch.flatten( x, start_dim = 0 ) ) ]) )
        mnist_test_dloader = torch.utils.data.DataLoader(  
                mnist_test_dset,
                batch_size = 1,
                shuffle = False, )

        # Bounds are trivial
        ip_lb = np.zeros( (net.in_size,), dtype = config.FLOAT_TYPE_NP )
        ip_ub = np.ones( (net.in_size,), dtype = config.FLOAT_TYPE_NP )
    
    else:
        raise NotImplementedError()

    # Get plot prefix
    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plotter_pfx = os.path.join(
            config.PLOT_PATH,
            '{}_abs_neurons_{}.plot'.format( args.abs_nodes, time_str )
    )

    # Measure original net accuracy
    l.info("Measuring accuracy of original network")
    get_accuracy( 
            mnist_test_dloader, lambda x: net.eval(x)[0], args.crit_class, 1000 )

    # Call
    resultant_net = compress_net( 
            net, args.crit_class, mnist_train_dloader, ip_lb, ip_ub, 
            args.abs_nodes,
            args.steps, 
            plots_fname_pfx = plotter_pfx,
            load_dump = args.load_dump,
            lr_init = lr_init,
            lr_backoff = lr_backoff,
    )

    l.info("Final accuracy")
    get_accuracy( mnist_test_dloader, 
        lambda x: resultant_net.eval(x)[0], 
        args.crit_class,
        num_points = 1000,
    )
    
