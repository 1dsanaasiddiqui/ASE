import logging as l
import copy
from math import sqrt
from functools import partial

import numpy as np
import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as f
import torch.nn.parameter

import network
import config
import utils



# Convention: vectors are multiplied to the left, vectors are row vects.
def uniform_dist(matrix_rows, matrix_cols, bounds, generator):
    """
    Generates a matrix with given rows and cols from uniform distribution with
    given bounds 
    
    Arguments:
    matrix_rows     -   Rows generated matrix should have
    matrix_cols     -   Cols generated matrix should have
    bounds          -   Generates values from -bounds to bounds
    generator       -   Torch rng state
    """
    v = torch.rand( (matrix_rows, matrix_cols), 
            generator=generator, dtype=config.FLOAT_TYPE_PYTORCH )
    return (v - 0.5) * (2. * bounds)



class ParametrizedAbstractNet( nn.Module ):
    """
    Class representing trainable parametrized abstract network.

    Node arranging convention :
        [(all concrete nodes for one abs node), 
        (all conc nodes for next abs node)
        ... ]

    Members:

    w_org           -   List of original weight matrices of incoming weights for
                        each non-input layer - Not a parameter
    b_org           -   List of original bias vector for each non-input layer -
                        Not a parameter
    w_prm           -   List of primed weight matrices of incoming weights for
                        each non-input layer. Doesnt have anything for last
                        layer, since that layer uses concrete weights. Is a
                        parameter
    b_prm           -   List of primed bias vector for each non-input layer.
                        Doesnt have anything for last layer, since that layer
                        uses concrete weights. Is a parameter
    w_range         -   Values used to clamp weights
    b_range         -   Values used to clamp weights
    lam_inc/dec     -   Matrix of lambda values [0, 1] which determine merging
                        of neurons in abs net k-1 X n. lam[i] is for layer i+1.
                        None if inc/dec not present in that layer
    ip_size         -   Size of input
    op_size         -   Size of output
    conc_lyr_sizes  -   Sizes of original concrete network
    no_of_abs_nodes -   No of abs nodes for each of inc and dec.
    ival_bounds     -   lb, ub for each layer, including input
    inp_lb          -   lb for input
    inp_ub          -   ub for input
    inc_dec_vects   -   The inc-dec vects. inc_dec_vects[i] corresponds to layer
                        i, where layer 0 is the input layer. Is None for the
                        input layer.
    id_rev_maps     -   Inverse of a mapping that would permute concrete nodes
                        in a way that all incs come before all decs
    end_relu        -   Does the network end with a relu
    no_of_layers    -   Number of layers, including input and output
    """
    def __init__( self, concrete_net, no_of_abs_nodes, inc_dec_vects,
             ival_bounds, 
             sound_wb_init = False ):
        """
        Arguments that aren't members:
        sound_wb_init   -   If enabled, wb initialized to min/max
        """
        super(ParametrizedAbstractNet, self).__init__()

        # Store inc dec vects and indices
        self.inc_dec_vects = inc_dec_vects

        self.inp_lb, self.inp_ub = ival_bounds[0]

        # Inverse of a mapping that would permute concrete nodes in a way that
        # all incs come before all decs
        self.id_rev_maps = []
        num_incs = []
        num_decs = []
        for idv in inc_dec_vects[1:]:
            ix = np.nonzero( idv > 0 )[0]
            dx = np.nonzero( idv < 0 )[0]
            ni, nd = ix.shape[0], dx.shape[0]
            num_incs.append(ni)
            num_decs.append(nd)
            rev_map = np.empty( (ni+nd,), dtype = int )
            if ni > 0: rev_map[ ix ] = np.arange( 0, ni, dtype = int )
            if nd > 0: rev_map[ dx ] = np.arange( ni, ni+nd, dtype = int )
            self.id_rev_maps.append( rev_map )

        # Some asserts to check input makes sense
        if config.ASSERTS:
            for b, idv in zip( concrete_net.biases, inc_dec_vects[1:] ):
                assert b.shape[0] == idv.shape[0]
        
        self.end_relu = concrete_net.end_relu
        # Piece 1: Parameters -- members of the class.

        # Convert interval bounds
        self.ival_bounds = [ 
            ( 
                torch.from_numpy( config.FLOAT_TYPE_NP( lb )), 
                torch.from_numpy( config.FLOAT_TYPE_NP( ub )), 
            ) for lb, ub in ival_bounds 
        ]

        # Create copies
        self.w_org = [ torch.from_numpy( config.FLOAT_TYPE_NP( w )) 
                for w in concrete_net.weights]
        # similar for b_org
        self.b_org = [ torch.from_numpy( config.FLOAT_TYPE_NP( b )) for b in concrete_net.biases]
        self.no_of_layers = concrete_net.num_layers
        self.ip_size = concrete_net.in_size
        self.op_size = concrete_net.out_size
        self.conc_lyr_sizes = concrete_net.layer_sizes
        self.no_of_abs_nodes = no_of_abs_nodes

        # Get values for clamping w,b
        self.w_range = []
        self.b_range = []
        for w,b in zip(self.w_org, self.b_org):
            self.w_range.append( config.WB_RANGE * torch.max( torch.abs( w )))
            self.b_range.append( config.WB_RANGE * torch.max( torch.abs( b )))

        # Set up rng
        generator = torch.Generator()
        generator.manual_seed(config.RNG_SEED)
        
        # Alphas
        self.alpha = []
        for i in range(self.no_of_layers - 2):
            self.alpha.append( nn.parameter.Parameter( torch.zeros(
                (no_of_abs_nodes*2 * concrete_net.layer_sizes[i+1]),
                dtype=config.FLOAT_TYPE_PYTORCH )))

        # Lambdas and counts
        self.lam_inc = []
        self.lam_dec = []
        for i in range(self.no_of_layers - 2):
            #for each layer, create a lambda matrix
            if config.LAMBDA_RANDOM_INIT:
                new_lam_inc = ( torch.rand(
                        (no_of_abs_nodes - 1, num_incs[i]),
                        generator=generator,dtype=config.FLOAT_TYPE_PYTORCH)
                    if num_incs[i] > 0 else None )
                new_lam_dec = ( torch.rand(
                        (no_of_abs_nodes - 1, num_decs[i]),
                        generator=generator,dtype=config.FLOAT_TYPE_PYTORCH)
                    if num_decs[i] > 0 else None )
            else:
                new_lam_inc = ( torch.ones( 
                        (no_of_abs_nodes - 1, num_incs[i]),
                        dtype=config.FLOAT_TYPE_PYTORCH, ) / no_of_abs_nodes
                    if num_incs[i] > 0 else None )
                new_lam_dec = ( torch.ones( 
                        (no_of_abs_nodes - 1, num_decs[i]),
                        dtype=config.FLOAT_TYPE_PYTORCH, ) / no_of_abs_nodes
                    if num_decs[i] > 0 else None )
                
            if num_incs[i] > 0:
                self.lam_inc.append( nn.parameter.Parameter( new_lam_inc ))
            if num_decs[i] > 0:
                self.lam_dec.append( nn.parameter.Parameter( new_lam_dec ))
            
        self.lambda_projection()

        # Initialize wb
        if sound_wb_init:
            init_w, init_b, _ = self.get_min_max_replaced_wb( 
                    lam_zero_threshold = 0 )
            self.w_prm = [ nn.parameter.Parameter( torch.from_numpy( w )) 
                    for w in init_w[:-1] ]
            self.b_prm = [ nn.parameter.Parameter( torch.from_numpy( b ))
                    for b in init_b[:-1] ]
            if config.ASSERTS:
                assert len(self.w_prm) == self.no_of_layers - 2
                assert len(self.b_prm) == self.no_of_layers - 2

        else:
            w_prm_vals, b_prm_vals = self.get_initial_wb()
            self.w_prm = [ nn.parameter.Parameter(v) for v in w_prm_vals ]
            self.b_prm = [ nn.parameter.Parameter(v) for v in b_prm_vals ]

    def get_initial_wb( self ):
        """
        Returns random values from which w,b may be initialized / reset to.
        
        Returns:
        1.  List of values to initialize w_prm from, same order as w_prm
        2.  List of values to initialize b_prm from, same order as b_prm
        """
        # Set up rng
        generator = torch.Generator()
        generator.manual_seed(config.RNG_SEED)

        # Parameter w via glorot
        w_prm_vals  = [ uniform_dist(              # First layer
            self.ip_size, self.no_of_abs_nodes*2,
            sqrt( 6.0 / (
                ( self.ip_size + self.no_of_abs_nodes*2 ) *
                self.conc_lyr_sizes[1]
            )) * config.WEIGHT_INIT_GAIN,
            generator,
        )]
        for i in range(1, self.no_of_layers - 2):   # Other non last layers
            w_prm_vals.append( uniform_dist(
                self.conc_lyr_sizes[i] * self.no_of_abs_nodes*2, 
                self.no_of_abs_nodes*2,
                sqrt( 3 / (
                    self.no_of_abs_nodes*2 * 
                    self.conc_lyr_sizes[i] * 
                    self.conc_lyr_sizes[i+1] 
                )) * config.WEIGHT_INIT_GAIN,
                generator,
            ))
        # Nothing for last layer

        # Parameter b
        b_prm_vals = []
        for i in range(0, self.no_of_layers - 2):   # Hidden layers except last
            b_prm_vals.append(
                torch.zeros((self.no_of_abs_nodes*2,),
                    dtype=config.FLOAT_TYPE_PYTORCH)
            )
        # Nothing for last layer

        return w_prm_vals, b_prm_vals

    def parameters(self, recurse: bool = True):
        return (
            self.w_prm + self.b_prm + 
            [ l for l in self.lam_inc if l is not None ] + 
            [ l for l in self.lam_dec if l is not None ] + 
            self.alpha )

    def parameters_wb( self ):
        """ Returns the w', b' parameters """
        return self.w_prm + self.b_prm

    def parameters_lam( self ):
        return ([ l for l in self.lam_inc if l is not None ] + 
                [ l for l in self.lam_dec if l is not None ])

    def lambda_projection(self):
        """
        Ensure that lambdas are in right range, and that sum is 1
        """
        for i in range(self.no_of_layers - 2):
            lam_t_i = self.lam_inc[i]
            if lam_t_i is not None:
                lam_t_i = torch.min(torch.tensor(1.0), lam_t_i).data
                lam_t_i = torch.max(torch.tensor(0.0), lam_t_i).data
                buff_lam_i = torch.sum(lam_t_i, dim=0)
                val_to_be_distributed_i = torch.where(
                        buff_lam_i > 1.0, 
                        buff_lam_i + config.FLOATING_POINT_ADJUST, 
                        torch.tensor( 1.0, dtype = config.FLOAT_TYPE_PYTORCH )
                )
                lam_t_i_copy = lam_t_i / val_to_be_distributed_i
                self.lam_inc[i].data[:,:] = lam_t_i_copy
                if config.ASSERTS:
                    if not torch.all(lam_t_i_copy >= 0):
                        l.debug("lam_t_i, lam_t_i_compy, dist: {} {} {}".format(
                            lam_t_i, lam_t_i_copy, val_to_be_distributed_i ))
                        assert False

            lam_t_d = self.lam_dec[i]
            if lam_t_d is not None:
                lam_t_d = torch.min(torch.tensor(1.0), lam_t_d).data
                lam_t_d = torch.max(torch.tensor(0.0), lam_t_d).data
                buff_lam_d = torch.sum(lam_t_d, dim=0)
                val_to_be_distributed_d = torch.where(
                        buff_lam_d > 1.0, 
                        buff_lam_d + config.FLOATING_POINT_ADJUST, 
                        torch.tensor( 1.0, dtype = config.FLOAT_TYPE_PYTORCH )
                )
                lam_t_d_copy = lam_t_d / val_to_be_distributed_d
                self.lam_dec[i].data[:,:] = lam_t_d_copy
                if config.ASSERTS:
                    if not torch.all(lam_t_d_copy >= 0):
                        l.debug("lam_t_d, lam_t_d_compy, dist: {} {} {}".format(
                            lam_t_d, lam_t_d_copy, val_to_be_distributed_d ))
                        assert False

    def scale_grads( self, 
            w_grad_scale, b_grad_scale, lam_grad_scale, alph_grad_scale ):
        """
        Applies configured scaling factor to gradients of parameters.
        """
        for lyr, (wp, bp) in enumerate( zip( self.w_prm, self.b_prm )):
            if wp.grad is not None:
                wp.grad = wp.grad * w_grad_scale
            if bp.grad is not None:
                bp.grad = bp.grad * b_grad_scale

        for lyr, (lmd_i, lmd_d) in enumerate( zip( self.lam_inc, self.lam_dec )):
            if lmd_i is not None and lmd_i.grad is not None:
                lmd_i.grad = lmd_i.grad * lam_grad_scale
            if lmd_d is not None and lmd_d.grad is not None:
                lmd_d.grad = lmd_d.grad * lam_grad_scale

        for lyr, alph in enumerate( self.alpha ):
            if alph.grad is not None:
                alph.grad = alph.grad * alph_grad_scale

    def get_full_lams( self, lam_inc = None, lam_dec = None ):
        """ 
        Returns the lambdas with the buffer lambdas multiplied and
        concatenated, and placed where they need to be. 
        Returns a list of tensors, one for each layer. For each
        layer, the [i,j] component corresponds to lambda for sending concrete
        node j to i-th merge group.
        """
        # TODO This returns a very sparse set of lambdas for compatibility. 
        # Optimize TODO: very sparse and inefficient, do better

        lam_inc = lam_inc if lam_inc is not None else self.lam_inc
        lam_dec = lam_dec if lam_dec is not None else self.lam_dec

        full_lams = []
        for lyr, (free_lam_i, free_lam_d, rev_map ) in enumerate( zip( 
                                lam_inc, lam_dec, self.id_rev_maps )):
            norm_lams = []
            if free_lam_i is not None:
                buff_lam_i = 1 - torch.sum( free_lam_i, dim=0, keepdim = True)
                norm_lam_i = torch.cat( (
                    free_lam_i, buff_lam_i,
                    torch.zeros( (self.no_of_abs_nodes, buff_lam_i.shape[1]),
                        dtype = config.FLOAT_TYPE_PYTORCH )), axis = 0 )
                norm_lams.append( norm_lam_i )

            if free_lam_d is not None:
                buff_lam_d = 1 - torch.sum( free_lam_d, dim=0, keepdim = True)
                norm_lam_d = torch.cat( (
                    free_lam_d, buff_lam_d,
                    torch.zeros( (self.no_of_abs_nodes, buff_lam_d.shape[1]),
                        dtype = config.FLOAT_TYPE_PYTORCH )), axis = 0 )
                norm_lams.append( norm_lam_d )

            arranged_lams = torch.cat( norm_lams, axis = 1 )
            #if config.DEBUG:
            #    l.debug("Reverse map: {}".format( rev_map ))
            #    l.debug("arranged lams shape: {}".format( arranged_lams.shape ))
            full_lam = arranged_lams[ :, rev_map ]
            full_lams.append( full_lam )

        return full_lams

    def get_wb_cnc( self, wb_idx ):
        """
        Returns the copies of concrete weights for each copy of concrete node
        for layer given by wb_idx + 1. L_S is calculated against this.
        """
        # Make concrete weights
        max_wb_idx = len(self.w_org)
        w_cnc = self.w_org[ wb_idx ]
        b_cnc = self.b_org[ wb_idx ]
        if wb_idx > 0:
            w_cnc = w_cnc.repeat( self.no_of_abs_nodes*2, 1 )
        if wb_idx < max_wb_idx - 1:
            w_cnc = w_cnc.repeat( 1, self.no_of_abs_nodes*2 )         
            b_cnc = b_cnc.repeat( self.no_of_abs_nodes*2 )
        w_cnc = torch.t( w_cnc )

        # Alpha mult
        if wb_idx < max_wb_idx - 1:
            w_cnc *= torch.exp(self.alpha[wb_idx].view(-1, 1))
            b_cnc *= torch.exp(self.alpha[wb_idx])
        if wb_idx > 0:
            w_cnc /= torch.exp(self.alpha[wb_idx-1])
    
        return w_cnc, b_cnc

    def calc_abs_ival_bounds( self ):
        """
        Calculates and returns the interval bounds for abstract values using the
        current abstract weights
        """
        # Get full lambdas
        full_lams = self.get_full_lams()
        
        # Init ivals
        lb = torch.from_numpy( self.inp_lb )  
        ub = torch.from_numpy( self.inp_ub )  
        ival_bounds = [( lb, ub )]

        # Layer loop
        for wp, bp, lams in zip( self.w_prm, self.b_prm, full_lams ):

            # Linear
            cen = (lb + ub) / 2
            ext = (ub - lb) / 2
            cen = cen @ wp + bp
            ext = ext @ torch.abs( wp )
            
            # Relu
            lb = f.relu( cen - ext )
            ub = f.relu( cen + ext )

            # Lambda mult
            lb = torch.flatten( torch.unsqueeze( lb, dim = 1 ) * lams )
            ub = torch.flatten( torch.unsqueeze( ub, dim = 1 ) * lams )

            if config.ASSERTS:
                assert torch.all( lb >= 0 )
                assert torch.all( ub >= 0 )

            ival_bounds.append(( lb, ub ))

        return ival_bounds

    def forward( self, x ):
        """
        Defines behavior and loss (Piece 2 & 3)

        Arguments:
        x           -   Input value x

        Returns y for given value of x for current value of (w,b,etc)
        """
        
        # Calculates y
        # Calculate buffer lambda and multiply
        unpacked_lambdas = [ torch.flatten(l) for l in self.get_full_lams() ]

        _temp_val = x

        # Calc residuals and adjustment
        residuals, _ = self.propagate_residuals()
        #if config.DEBUG:
        #    l.debug("residual adjustment in forward {}".format(
        #        utils.str_with_idx( 
        #            residuals * torch.from_numpy( self.inc_dec_vects[-1] )) ))
        res_adjustment = residuals * torch.from_numpy( self.inc_dec_vects[-1] )

        # TODO Clean up indexing
        max_wb_idx = len(self.w_org)
        for wb_idx in range(max_wb_idx):

            # Get bounds
            lb, ub = self.ival_bounds[ wb_idx ]

            npp_lyr_size = self.w_org[ wb_idx ].shape[1]
            if wb_idx < max_wb_idx - 1:       # Not last layer
                npp_lyr_size *= self.no_of_abs_nodes*2
            
            # If not last layer, weights and biases come from primed vars
            if wb_idx < max_wb_idx - 1:       

                # Never last layer, copy cols  
                temp_w = torch.repeat_interleave( 
                        self.w_prm[wb_idx], 
                        self.conc_lyr_sizes[wb_idx+1], dim = 1 )
                temp_b = torch.repeat_interleave( 
                        self.b_prm[wb_idx], 
                        self.conc_lyr_sizes[wb_idx+1], dim = 0 )
            
            # Otherwise, forward pass w,b is just origial weights
            else:
                temp_w, temp_b = self.get_wb_cnc( wb_idx )
                temp_w = temp_w.T

            # Clamp w,b
            if config.WB_CLAMP_ENABLE:
                temp_w = torch.clamp( 
                        temp_w, 
                        min = -self.w_range[wb_idx], 
                        max = self.w_range[wb_idx] )
                temp_b = torch.clamp( 
                        temp_b, 
                        min = -self.b_range[wb_idx], 
                        max = self.b_range[wb_idx] )
                
            _temp_val = _temp_val @ temp_w
            #if config.DEBUG:
            #    l.debug("Output on orig net post w mult, layer {} val {}".format( 
            #        wb_idx, utils.str_with_idx(_temp_val) ))
            _temp_val += temp_b
            #if config.DEBUG:
            #    l.debug("Output on orig net post b add, layer {} val {}".format( 
            #        wb_idx, utils.str_with_idx(_temp_val) ))

            # Residual adjustment if needed
            if wb_idx >= max_wb_idx - 1 and config.ADD_RESIDUAL_BEFORE_LAST_RELU:
                _temp_val += res_adjustment 

            #if config.DEBUG:
            #    l.debug("Output on orig net post res adj, layer {} val {}".format( 
            #        wb_idx, utils.str_with_idx(_temp_val) ))

            # Do relu if needed
            if wb_idx < max_wb_idx - 1 or self.end_relu:       
                _temp_val = f.relu( _temp_val )

            #if config.DEBUG:
            #    l.debug("Output on orig net post relu, layer {} val {}".format( 
            #        wb_idx, utils.str_with_idx(_temp_val) ))
            #    l.debug("max_wb_idx, end_relu: {}, {}".format(
            #        max_wb_idx, self.end_relu ))

            if config.ASSERTS:
                assert torch.all( torch.isfinite( _temp_val ))

            # If not last layer, handle lambda mult
            if wb_idx < max_wb_idx - 1:
                _temp_val = _temp_val * unpacked_lambdas[ wb_idx ]

        y = _temp_val

        # Add residuals
        if not config.ADD_RESIDUAL_BEFORE_LAST_RELU:
            y = y + res_adjustment

        #if config.DEBUG:
        #    l.debug("Output on orig net post residual adj, val {}".format( 
        #        utils.str_with_idx( y ) ))

        # TODO Remove soundness loss calc
        return y

    def propagate_residuals( self, ival_bounds = None ):
        """
        Propagates the residuals from unsoundness to output layer.

        Arguments:
        ival_bounds -   Ival bounds to use, if not given recalculates from
                        current w',b',lambda

        Returns: 
        1.  The residual at output layer
        2.  Residuals at each layer, starting from first hidden, including
            output.
        """
        # TODO optimize out redundant idv mults, in gen, simplify
        lams = self.get_full_lams()
        unpacked_lambdas = [ torch.flatten(l) for l in lams ]

        # Generate ival bounds
        if ival_bounds is None:
            ival_bounds = self.calc_abs_ival_bounds()

        # Layer loop
        residuals = 0
        residuals_list = []
        for lyr in range( len( self.w_prm )):

            # conc weights
            w_cnc, b_cnc = self.get_wb_cnc( lyr )
            a_w_cnc = torch.abs( w_cnc )
            
            # Abs weights
            w_abs = torch.repeat_interleave( 
                    self.w_prm[ lyr ].T, 
                    self.conc_lyr_sizes[ lyr + 1 ], dim = 0 )
            b_abs = torch.repeat_interleave( 
                    self.b_prm[ lyr ], 
                    self.conc_lyr_sizes[ lyr + 1 ], dim = 0 )
            
            # Ivals
            lb, ub = ival_bounds[ lyr ]

            # Inc dec vects
            idv = torch.from_numpy( self.inc_dec_vects[ lyr+1 ] ).repeat(
                self.no_of_abs_nodes * 2 )

            # Calculate deltas TODO use loss fn here
            sm_cnc_abs = (w_cnc - w_abs) * torch.unsqueeze( idv, 1 )
            deltas = f.relu( torch.sum(
                    torch.where( sm_cnc_abs > 0, ub, lb ) * sm_cnc_abs, dim = 1 
                ) + idv * (b_cnc - b_abs))

            #if config.DEBUG:
            #    l.debug("Deltas at layer {} are {}, max {}".format( lyr, deltas,
            #        torch.max( deltas ) ))

            # Get residuals
            residuals = ( deltas + torch.sum( residuals * a_w_cnc, dim = 1 )
                ) * unpacked_lambdas[ lyr ] 
            residuals_list.append( residuals )

            #if config.DEBUG and lyr == 0:
            #    l.debug("Residuals at layer 0 are {}".format(
            #        utils.str_with_idx( residuals ) ))

            if config.ASSERTS and not torch.all( residuals >= 0 ):
                l.error("Nonzero residuals {} at layer {}".format( residuals,
                    lyr ))
                bad_idxs = torch.nonzero( residuals < 0 )[0]
                bad_vals = residuals[ bad_idxs ]
                l.error("Indices, values of nonzero: {},{}".format(
                    bad_idxs, bad_vals ))
                l.error("Corresponding idv: {}".format( idv[ bad_idxs ] ))
                assert False

        # Last layer 
        idv = torch.from_numpy( self.inc_dec_vects[ -1 ] )
        prv_idv = torch.from_numpy( self.inc_dec_vects[ -2 ] ).repeat(
            self.no_of_abs_nodes * 2 )
        w_lst, _ = self.get_wb_cnc( len( self.w_prm ) )
        residuals = torch.sum( residuals * torch.abs( w_lst ), dim = 1 )
        residuals_list.append( residuals )
        #if config.DEBUG:
        #    l.debug("Last layer residuals: {}".format( residuals ))

        if config.ASSERTS and not torch.all( residuals >= 0 ):
            l.error("Nonzero residuals {} at last layer".format( residuals ))
            assert False

        return residuals, residuals_list

    def merge_weights( self, weights, biases, lams ):
        """
        Merges given weights and biases based on given lambda and returns
        """
        # Layer loop to sum outgoing
        for lyr in range( len( weights ) - 1):

            w_merged = []

            # Loop over abstract nodes
            for abs_idx in range( 2 * self.no_of_abs_nodes ):

                # Merge outgoing weights
                out_w_start = abs_idx * self.conc_lyr_sizes[ lyr+1 ]  
                out_w_end = (abs_idx+1) * self.conc_lyr_sizes[ lyr+1 ]  
                w = ( weights[ lyr+1 ][ out_w_start:out_w_end, : ] * 
                    np.expand_dims( lams[ lyr ][ abs_idx, : ], axis = 1 ))
                w_merged.append( np.sum( w, axis = 0 ))

            weights[ lyr+1 ] = np.stack( w_merged, axis = 0 )
            
        return weights, biases
    
    def project_get_net_residuals( self ):
        """
        Performs a projection to get abstract network via residuals method
        """
        # TODO Use better calculation for deltas
        if config.DEBUG:
            import numpy.random
            rng = np.random.default_rng( seed = 0 )
            rand_inp = config.FLOAT_TYPE_NP( rng.uniform( size = (self.ip_size,) ))
            l.debug("Trying random input: {}".format( rand_inp ))
            out_orig = self( torch.from_numpy( rand_inp )).detach().numpy(
                    force = True )
            l.debug("Output on current net: {}".format( out_orig ))
        
        # Get residuals
        residuals, _ = self.propagate_residuals()
        residuals = residuals.detach().numpy( force = True ) # pytype: disable=attribute-error

        # Weights upto last layer
        weights = [ w.detach().numpy( force = True ) for w in self.w_prm ]
        biases =  [ b.detach().numpy( force = True ) for b in self.b_prm ]

        # Last layer
        w,b = self.get_wb_cnc( self.no_of_layers - 2 )
        weights.append( np.transpose( w.numpy( force = True )))
        if config.ADD_RESIDUAL_BEFORE_LAST_RELU:
            biases.append( b.numpy( force = True ) + 
                            residuals * self.inc_dec_vects[-1] )
        else:
            biases.append( b.numpy( force = True ))

        if config.DEBUG:
            l.debug("idvs: {}".format( self.inc_dec_vects[-1] ))
            l.debug("residuals: {}".format( residuals ))
            l.debug("residual adjustment in project {}".format(
                utils.str_with_idx( residuals * self.inc_dec_vects[-1] ) ))

        # Lambdas
        lams = [ l.detach().numpy( force = True ) for l in self.get_full_lams() ]

        # Merge
        weights, biases = self.merge_weights( weights, biases, lams )
        if config.DEBUG:
            l.debug("Merged weight shapes: {}".format(
                [ w.shape for w in weights ]))
        proj_net = network.Network( weights, biases, end_relu = self.end_relu )

        # If residuals are to be added to layer after last, do it
        if not config.ADD_RESIDUAL_BEFORE_LAST_RELU: 
            l.info("Appending residual addition layer")
            proj_net.append_layer(
                np.eye( proj_net.out_size ), 
                residuals * self.inc_dec_vects[-1],
                relu = False )

        if config.DEBUG:
            cval = rand_inp
            # Evaluate inner layers
            for i, (w, b) in enumerate( zip( 
                    proj_net.weights[:-1], proj_net.biases[:-1] )):
                cval = cval @ w
                l.debug("Vals after w mul on projected layer {} vals {}".format(
                    i, utils.str_with_idx( cval ) ))
                cval = cval + b
                l.debug("Vals after b add on projected layer {} vals {}".format(
                    i, utils.str_with_idx( cval ) ))
                cval = np.maximum( cval, 0 )
                l.debug("Vals after relu add on projected layer {} vals {}".format(
                    i, utils.str_with_idx( cval ) ))

            # Evaluate last layer
            cval = cval @ proj_net.weights[-1]
            l.debug("Vals after w mul on projected last layer vals {}".format(
                utils.str_with_idx( cval ) ))
            cval = cval + proj_net.biases[-1]
            l.debug("Vals after b add on projected last layer vals {}".format(
                utils.str_with_idx( cval ) ))
            if proj_net.end_relu:
                cval = np.maximum( cval, 0 )
                l.debug("Vals after relu add on projected last layer vals {}".format(
                    utils.str_with_idx( cval ) ))

            # This is a very weak check... but some fp wierdness is happening. 
            # TODO Modify network.Network to properly support residual addition
            if config.ASSERTS and not np.allclose( cval, out_orig, atol=1e-2 ):
                l.debug(" cval {}, out_orig {}, diff {}".format(
                    cval, out_orig, cval-out_orig ))
                raise AssertionError()

        # Return network
        return proj_net

    def dump_params( self, fname ):
        """
        Dump all params to given npz filename. If filename does not end with
        .npz, it will be appended.
        """
        data = {}
        for i, wp in enumerate( self.w_prm ):
            data[ 'w_prm_{}'.format(i) ] = wp.data.detach().numpy( force=True )
        for i, bp in enumerate( self.b_prm ):
            data[ 'b_prm_{}'.format(i) ] = bp.data.detach().numpy( force=True )
        for i, li in enumerate( self.lam_inc ):
            if li is not None:
                data[ 'lam_inc_{}'.format(i) ] = li.data.detach().numpy( 
                        force=True )
        for i, ld in enumerate( self.lam_dec ):
            if ld is not None:
                data[ 'lam_dec_{}'.format(i) ] = ld.data.detach().numpy( 
                        force=True )
        for i, a in enumerate( self.alpha ):
            data[ 'alpha_{}'.format(i) ] = a.data.detach().numpy( force=True )
        np.savez( fname, **data )

    def load_params( self, fname ):
        """
        Load all params from given npz filename. 
        """
        data = np.load( fname )
        l.info( "Loading npz with items: ".format( list( data.keys() )))
        for i, wp in enumerate( self.w_prm ):
            wp.data = torch.from_numpy( data[ 'w_prm_{}'.format(i) ] )
        for i, bp in enumerate( self.b_prm ):
            bp.data = torch.from_numpy( data[ 'b_prm_{}'.format(i) ] )
        for i, li in enumerate( self.lam_inc ):
            li.data = torch.from_numpy( data[ 'lam_inc_{}'.format(i) ] )
        for i, ld in enumerate( self.lam_dec ):
            ld.data = torch.from_numpy( data[ 'lam_dec_{}'.format(i) ] )
        for i, a in enumerate( self.alpha ):
            a.data = torch.from_numpy( data[ 'alpha_{}'.format(i) ] )
