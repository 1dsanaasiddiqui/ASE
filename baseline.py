"""
A script with various fns associated with the training loop
"""
import logging as l
import copy
from math import sqrt
from collections import namedtuple
import itertools

import torch
import torch.nn as nn
#import torch.profiler
from tqdm import tqdm

import config
import utils
import numpy as np
import os


class NormalizedStepOptim( torch.optim.Optimizer ):
    """
    A class to do single step small updates

    Members:
    params  -   List of parameters, assumed to have .grad
    lr      -   Interpreted as constant step size
    """
    def __init__( self, params, lr ):
        """ Initialize """
        self.params = params
        self.lr = lr

    def zero_grad( self ):
        """ Set params to zero """

        for p in self.params:
            p.grad = None

    def step( self ):
        """ Move small fixed amount in direction of grad """

        # Calc sum of squares
        sq_sum = sum(( torch.sum( p.grad * p.grad ) for p in self.params ))
        if sq_sum <= 0:
            return
        norm = sqrt( sq_sum )

        for p in self.params:
            p.data = p.data - p.grad * self.lr / sq_sum


# Typedefs for returnning saved nets
SavedNet = namedtuple( 'SavedNet', [
    'abs_net',      # The network
    # Various losses for that network
    'cex_loss',
    'step',        # Epoch at which net was saved
])


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
    that many points

    NOTE: This expects dset to be an iterator over img, tgt with batch size 1.
    """
    n_correct = 0
    n_tot = 0 
    n_true_pos = 0
    n_true_neg = 0
    n_pos = 0
    n_false_pos = 0
    n_false_neg = 0
    n_neg = 0
    for point_idx, (img, tgt) in enumerate( tqdm( dset )):

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


def train_abs_net(abs_net, data_iter, no_of_steps, cex_loss_fn,
        plotter_fname = None,
        lr = config.LR_INIT,
        vis_loss = config.VISUALIZE,
        dump_pfx = None,
        optim_class = torch.optim.Adam
        ):
    """
    Arguments:
    
    abs_net         -   The abstract network to train
    data_iter       -   An iterator producing batches of input and target data,
                        Could be a DataLoader
    no_of_steps     -   The number of gradient descent steps to train for
    cex_loss_fn     -   The function to calculate cex loss from the output and
                        target values. This defines if some undesirable event
                        has happened.
    plotter_fname   -   The file name to store plots to. Various plots are
                        stored at file names generated via adding various
                        suffixes to this. 
    *_loss_scale    -   Scaling factors applied to components of loss
    *_grad_scale    -   Amount by which various components of gradient are
                        scaled afer gradient descent
    lr              -   Learning rates
    vis_loss/lam    -   Enable/disable plotting of lambdas and loss
    dump_pfx        -   If given, dumps saved nets to given prefix
    optim_class     -   The optimizer class to use, Adam by default.
    
    Returns: Multiple saved net structures:
    1.  Network with best soundness loss.
    2.  Network with best counterexample loss.
    """
    #absnet-- nn.sequential

    # TODO More documentation
    l.info("Starting training")

    if vis_loss:
        plotter = utils.Plotter( fname = plotter_fname )
    
    # Two optimizers, one for wb, one for lambas
    #Pass in the network paramters
    optim = optim_class(abs_net.parameters(), lr=lr)

    # Keep track of best so far for soundness and cex losses
    #best_snd_save = None
    best_cex_save = None
    #best_tot_save = None

    #Disable lambda fixing
    # Threshold for fixing lambda
    #lam_fix_thres = 1
    #step_sinc_impr = 0

    # State for keeping track of ls-lc alternations
    # if config.LS_LC_ALTER_ENABLE:
    #     current_ls = True
    #     iter_since_last_alter = 0
    #     iter_since_loss_change = 0
    #     last_loss_val = None

    loop_iter = itertools.islice( itertools.cycle( data_iter ), no_of_steps )
    if config.SHOW_PBARS:
        loop_iter = tqdm( loop_iter )
    for step, (inp, tgt) in enumerate( loop_iter ):
        l.info("Step {} of {}".format( step, no_of_steps ))

        #if config.DEBUG:
        #    l.debug("Input, target shape: {}, {}".format( inp.shape, tgt.shape ))

        # Uncomment for torch profiling
        #with torch.profiler.profile(
        #        activities = [ torch.profiler.ProfilerActivity.CPU ],
        #        record_shapes = True,
        #        with_stack = True,
        #        profile_memory = True, ) as prof:

        # Compute prediction error
        #Result of network on that inp
        
        #TODO Change this
        pred = abs_net( inp )
        #Simply the cross entropy loss

        cex_loss = cex_loss_fn( pred, tgt )
        #snd_loss *= sound_loss_scale
        #lam_loss *= lambda_loss_scale
        # if config.LS_LC_ALTER_ENABLE:
        #     loss = ( snd_loss + lam_loss + cex_loss ) if current_ls else cex_loss
        # else:
        loss =  cex_loss 

        # if config.LOSS_LOG_IVAL >= 0 and step % config.LOSS_LOG_IVAL == 0:
        #     l.info("Sound loss: {}, lam loss: {}, cex: {}, sum: {}".format( 
        #         snd_loss, lam_loss, cex_loss, loss))

        if config.ASSERTS:
            #assert torch.isfinite( snd_loss )
            assert torch.isfinite( cex_loss )
            #assert torch.isfinite( lam_loss )

        # Save best yet, reset lambda threshold TODO do something with nets
        any_saved = False
       
        if (
                best_cex_save is None or 
                cex_loss < best_cex_save.cex_loss or
                (
                    torch.isclose( cex_loss, best_cex_save.cex_loss ) 
            )):
            l.info("Saving net with best lc")
            best_cex_save = SavedNet( 
                copy.deepcopy( abs_net ), cex_loss, step )
            #TODO change this
            # if dump_pfx is not None:
            #     fname = "{}_best_cex".format( dump_pfx )
            #     l.info("Dumping best counterexample to {}".format( fname ))
            #     abs_net.dump_params( fname )
            any_saved = True
        # TODO Save using proper tot loss.. not loss gd is done on.
        #if best_tot_save is None or loss < best_tot_save.tot_loss:
        #    l.info("Saving net with best lc")
        #    best_tot_save = SavedNet( 
        #        copy.deepcopy( abs_net ),
        #        snd_loss, cex_loss, lam_loss, loss, step )
        #    any_saved = True

        # Post save
        if any_saved:
            # lam_fix_thres = 1
            step_since_impr = 0
            any_saved = False

        # Backpropagation
        loss.backward()
        
        # if config.DEBUG:
        #     #TODO change this
        #     for i, p in enumerate( abs_net.parameters_wb() ):
        #         g = p.grad
        #         l.debug( "wb grads at index {}: {} of {} zero".format(
        #             i, torch.count_nonzero( torch.isclose( g, torch.tensor(0.) )), 
        #             torch.numel(g) ))

        # Uncomment for step debugging
        #if config.DEBUG:
        #    wb = abs_net.parameters_wb()
        #    lm = abs_net.lam
        #    l.debug("W,b: {}".format( wb ))
        #    l.debug("W,b grads: {}".format( [t.grad for t in wb] ))
        #    l.debug("lams: {}".format( lm ))
        #    l.debug("lams grads: {}".format( [t.grad for t in lm] ))
        #    input()

        optim.step()

        #reset phase
        optim.zero_grad()

        if vis_loss:
            plotter.record( 'Cex loss',  cex_loss.detach().item() ) #DEBUG

        # Ls-lc alternation manage state
        # Uncomment for torch profiling 
        # PROFILE
        #l.info( "Profiling table: {}".format(
        #    prof.key_averages( 
        #        group_by_stack_n = 5,
        #        group_by_input_shape = True,
        #    ).table( 
        #        sort_by = 'self_cpu_memory_usage',
        #        row_limit = 10
        #)))

        # Uncomment for step profiling 
        #if config.DEBUG
        #    input()

            
        
    l.info("At End,  cex: {}".format( 
        cex_loss))

    if vis_loss:
        plotter.show( show = True )

    # Return best yet networks
    return (
        best_cex_save
    )

if __name__ == "__main__":
    """
    Basic test with an .nnet and .prop file that attempts to train a
    parametrized abstract network with given settings.
    """

    from datetime import datetime
    import sys
    import argparse

    import train_loop
    
    utils.init_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument('-n-in', '--net-input', type = int,
            required = True,
            help = "Number of Input neurons", )
    parser.add_argument('-n-out', '--net-output', type = int,
            required = True,
            help = "Number of Output neurons", )
    parser.add_argument('-l', '--layer', type = int,
            required = True,
            help = "Number of hidden layers in the network", )
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
    args = parser.parse_args()
    l.info("Args are {}".format( args ))
    
    # Seed everything
    rng = np.random.default_rng( seed = config.RNG_SEED )
    torch.manual_seed( seed = config.RNG_SEED )
    if config.DEBUG:
        torch.use_deterministic_algorithms( mode = True )

    # Load network
    layers = []
    hidden_layers = [args.abs_nodes for i in range(args.layer)]
    current_size = args.net_input

    for hidden_size in hidden_layers:
        layers.append(nn.Linear(current_size, hidden_size))
        layers.append(nn.ReLU())
        current_size = hidden_size

    layers.append(nn.Linear(current_size, args.net_output))

    net = nn.Sequential(*layers)


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
        mnist_test_dset = torchvision.datasets.MNIST( 
                root = config.DSETS_PATH, 
                download = True, train = False, 
                transform = T.Compose([ 
                    T.ToTensor(), 
                    T.Lambda( lambda x: torch.flatten( x, start_dim = 0 ) ) ]) )
        mnist_train_dloader = torch.utils.data.DataLoader(
                mnist_train_dset,
                batch_size = args.batch_size,
                shuffle = True, )
        # TODO move to test dset
        mnist_test_dloader = torch.utils.data.DataLoader(  
                mnist_test_dset,
                batch_size = 1,
                shuffle = False, )

        # Bounds are trivial
        ip_lb = np.zeros( (args.net_input,), dtype = config.FLOAT_TYPE_NP )
        ip_ub = np.ones( (args.net_input,), dtype = config.FLOAT_TYPE_NP )
    
    else:
        raise NotImplementedError()

    # Get plot prefix
    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plotter_pfx = os.path.join(
            config.PLOT_PATH,
            '{}_abs_neurons_{}.plot'.format( args.abs_nodes, time_str )
    )

    # Measure original net accuracy

    # Call
    resultant_net  =train_abs_net(net, mnist_train_dloader, args.steps, false_positive_rate,
        plotter_fname = None,
        lr = config.LR_INIT,
        vis_loss = config.VISUALIZE,
        dump_pfx = None,
        optim_class = torch.optim.Adam
        )

    l.info("Final accuracy")
    get_accuracy( mnist_test_dloader, 
        lambda x: net(x).detach(), 
        args.crit_class,
        num_points = 100000 )
