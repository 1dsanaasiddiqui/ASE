"""
A script with various fns associated with the training loop
"""
import logging as l
import copy
from math import sqrt
from collections import namedtuple
import itertools

import torch
#import torch.profiler
from tqdm import tqdm

import config
import utils


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
    'loss',         # The loss
    'step',         # Epoch at which net was saved
])

def train_abs_net(abs_net, data_iter, no_of_steps, loss_fn,
        plotter_fname = None,
        w_grad_scale = config.W_GRAD_SCALE, 
        b_grad_scale = config.B_GRAD_SCALE, 
        lam_grad_scale = config.LAM_GRAD_SCALE, 
        alph_grad_scale = config.ALPHA_GRAD_SCALE, 
        lr_init = config.LR_INIT,
        lr_backoff = config.LR_BACKOFF,
        vis_loss = config.VISUALIZE,
        vis_lam = config.LAMBDA_VIZ_ENABLE,
        project_wb = False,
        dump_pfx = None,
        optim_class = torch.optim.Adam,
        loss_check_hook = None,
        show_plot = True,
        ):
    """
    Arguments:
    
    abs_net         -   The abstract network to train
    data_iter       -   An iterator producing batches of input and target data,
                        Could be a DataLoader
    no_of_steps     -   The number of gradient descent steps to train for
    loss_fn     -   The function to calculate cex loss from the output and
                        target values. This defines if some undesirable event
                        has happened.
    plotter_fname   -   The file name to store plots to. Various plots are
                        stored at file names generated via adding various
                        suffixes to this. 
    *_grad_scale    -   Amount by which various components of gradient are
                        scaled afer gradient descent
    lr_init         -   The initial learning rate to start training with.
    lr_backoff      -   List of pair of values, representing how lr is backed
                        off as training progresses. The first value is the loss
                        value at which the learning rate change will be
                        triggered, and the second value is the new learning
                        rate.
    vis_loss/lam    -   Enable/disable plotting of lambdas and loss
    dump_pfx        -   If given, dumps saved nets to given prefix
    optim_class     -   The optimizer class to use, Adam by default.
    loss_check_hook -   If given, this is called after each iteration to on the
                        loss. If this returns False, training is stopped.
    show_plot       -   If true, show final plots, else only dumps plot data
    
    Returns: A saved net structure with best loss network
    1.  Network with best soundness loss.
    2.  Network with best counterexample loss.
    """
    # TODO More documentation
    l.info("Starting training")
    if config.DEBUG:l.debug("scales are {},{},{},{}".format(  
            w_grad_scale, b_grad_scale, lam_grad_scale, alph_grad_scale ))

    if vis_loss:
        plotter = utils.Plotter( fname = plotter_fname )
        res_plotter = utils.Plotter( fname = '{}_residuals'.format( plotter_fname ))

    if vis_lam:
        # One plotter for each layer with lambda
        lam_plots = [ 
            utils.HistogramPlotter( 
                num_bins = config.LAMBDA_VIZ_NUM_BINS,
                fname = '{}_lambdas_{}'.format(plotter_fname, i),
            ) for i in range(len(abs_net.lam_inc)) ]
    
    
    # Two optimizers, one for wb, one for lambas
    if config.DEBUG:
        l.debug("Setting up optimizer with lr {}".format( lr_init ))
    param_groups = [{'params': abs_net.parameters(), 'lr': lr_init }]
    optim = optim_class( param_groups, lr=lr_init )

    # Keep track of best so far 
    best_save = None

    loop_iter = itertools.islice( itertools.cycle( data_iter ), no_of_steps )
    if config.SHOW_PBARS:
        loop_iter = tqdm( loop_iter )
    for step, (inp, tgt) in enumerate( loop_iter ):
        l.info("Step {} of {}".format( step, no_of_steps ))

        # Uncomment for torch profiling
        #with torch.profiler.profile(
        #        activities = [ torch.profiler.ProfilerActivity.CPU ],
        #        record_shapes = True,
        #        with_stack = True,
        #        profile_memory = True, ) as prof:

        # Compute prediction error
        pred = abs_net( inp )
        loss = loss_fn( pred, tgt )

        if config.LOSS_LOG_IVAL >= 0 and step % config.LOSS_LOG_IVAL == 0:
            l.info("Loss: {}".format( loss ))

        if config.ASSERTS:
            assert torch.isfinite( loss )

        # Save best yet, reset lambda threshold TODO do something with nets
        if best_save is None or loss < best_save.loss :
            l.info("Saving net with best loss")
            best_save = SavedNet( copy.deepcopy( abs_net ), loss, step )
            if dump_pfx is not None:
                fname = "{}_best_cex.npz".format( dump_pfx )
                l.info("Dumping best counterexample to {}".format( fname ))
                abs_net.dump_params( fname )
            # TODO Save plot so far
            plotter.save()
            for l_plt in lam_plots: l_plt.save()

        # Call loss hook, exit if needed.
        if loss_check_hook is not None and not loss_check_hook( loss ):
            l.info("Loss check hook returned false on loss {}".format( loss ))
            break

        # Backoff learning rate
        if len( lr_backoff ) > 0:
            trigger, lr_new = lr_backoff[0]
            if loss <= trigger:
                l.info("Setting new lr {}".format( lr_new ))
                for param_group in optim.param_groups:
                    param_group['lr'] = lr_new 
                lr_backoff = lr_backoff[1:]

        # Backpropagation
        loss.backward()
        abs_net.scale_grads( 
                w_grad_scale, b_grad_scale, lam_grad_scale, alph_grad_scale )
        optim.step()

        #reset phase
        optim.zero_grad()
        abs_net.lambda_projection()
        if project_wb: 
            abs_net.project_wb_sound()

        if vis_loss:
            plotter.record( 'Loss', loss.detach().item() )

            residuals, _ = abs_net.propagate_residuals()
            for i, res in enumerate( residuals ):
                res_plotter.record( 'Res_{}'.format( i ), abs( 
                    res.detach().item() ))

        if vis_lam:
            for l_plt, vals in zip( lam_plots, abs_net.get_full_lams() ):
                max_vals = torch.amax( vals.detach(), dim = 0 )
                l_plt.record( max_vals.numpy() )

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

            
        
    l.info("At End, Loss: {}".format( loss ))

    if vis_loss:
        plotter.show( show = show_plot, save = True )
        res_plotter.show( show = show_plot, save = True )

    if vis_lam:
        for i, l_plt in enumerate( lam_plots ):
            l_plt.show( title = 'Lambdas_{}'.format(i), 
                    show = show_plot, save = True )

    # Return best yet networks
    return best_save

