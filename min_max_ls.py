"""
An implementation of L_S that enforces only component-wise min and max
"""
import logging as l

import torch
from torch.nn.functional import relu as torch_relu

import config
import utils

class MinMaxLossCalculator:
    """
    A class to calculate soundness loss based on min/max.
    
    Members:
    npp_prev_lyr_size   -   Size of the previous layer in N''
    no_of_abs_nodes     -   Number of abstract nodes 
    is_inc_rep_msk      -   Indices of inc nodes in N'' (factors in the repetitions)
    is_dec_rep_msk      -   Indices of dec nodes in N'' (factors in the repetitions)
    no_of_cnc_nodes     -   Number of concrete nodes
    prev_lyr_hidden     -   If true, calculate as if the previous layer is a
                            hidden layer, else consider it to be the input layer
    """
    def __init__( self, no_of_abs_nodes, inc_dec_vect, prev_lyr_hidden,
            npp_prev_lyr_size ):
        """
        Arguments:

        no_of_abs_nodes     -   Number of abstract nodes
        inc_dec_vect        -   A vector with +1 for inc and -1 for dec nodes in
                                the concrete network (that is, N)
        prev_lyr_hidden     -   If true, calculate as if the previous layer is a
                                hidden layer, else consider it to be the input
                                layer
        npp_prev_lyr_size   -   Size of the previous layer in N''
        """
        self.no_of_abs_nodes = no_of_abs_nodes
        self.no_of_cnc_nodes = inc_dec_vect.shape[0]
        self.npp_prev_lyr_size = npp_prev_lyr_size
        self.prev_lyr_hidden = prev_lyr_hidden
        self.is_inc_rep_msk = (inc_dec_vect > 0).repeat( self.no_of_abs_nodes )
        self.is_dec_rep_msk = torch.logical_not( self.is_inc_rep_msk )

    def calc_ls( self, w_cnc, b_cnc, lb, ub, w_abs, b_abs, 
            sub_idxs = None,
            check_idv = None,
            num_iters = config.NUM_LS_ITERS, 
            ):
        """
        Calculates and returns the soundness losses for an entire layer.

        Arguments:

        w_cnc, b_cnc    -   Weights and bias of concrete nodes (in N'')
        lb, ub          -   lb and ub of concrete net values in prev layer
        w_abs, b_abs    -   The proposed abstract weights and biases for each
                            node in N''
        sub_idxs        -   The indices of N'' nodes for which L_S calculation
                            must be done. Size of w_org and b_org must match
                            this.
        check_idv       -   If passed, this is matched with stored inc-dec vects
                            via an assert.
        num_iters       -   Ignored - only present for compatibility reasons

        Returns: 
        1.  A vector with the corresponding soundness loss value for each neuron in
            N'' in that layer.
        """
        # TODO Scale returned loss
            
        # Get layer sizes in N''
        num_nodes = w_cnc.shape[0]
        npp_prev_lyr_size = w_cnc.shape[1]

        # Do slicing
        if sub_idxs is not None:

            if config.ASSERTS:
                assert sub_idxs.shape[0] == w_cnc.shape[0]
                assert sub_idxs.shape[0] == b_cnc.shape[0]
                assert sub_idxs.shape[0] == w_abs.shape[0]
                assert sub_idxs.shape[0] == b_abs.shape[0]

            is_inc_rep_msk = self.is_inc_rep_msk[ sub_idxs ]
            is_dec_rep_msk = self.is_dec_rep_msk[ sub_idxs ]

            if config.ASSERTS and check_idv is not None and (
                    not torch.all( is_inc_rep_msk == ( check_idv > 0 ))):
                l.error( "Inc dec vect slices don't match")
                l.error( "Sub idx: {}".format( sub_idxs ))
                l.error( "is_inc_rep_msk: {}".format( is_inc_rep_msk ))
                l.error( "check_idv: {}".format( check_idv ))
                assert False
                    
        else:
            if config.ASSERTS:
                assert self.no_of_cnc_nodes * self.no_of_abs_nodes == w_cnc.shape[0]
                assert b_cnc.shape[0] == w_cnc.shape[0]
                assert b_abs.shape[0] == w_cnc.shape[0]
                assert w_abs.shape[0] == w_cnc.shape[0]

            is_inc_rep_msk = self.is_inc_rep_msk
            is_dec_rep_msk = self.is_dec_rep_msk
            

        is_inc_rep_idx = torch.nonzero( is_inc_rep_msk, as_tuple = True )[0]
        is_dec_rep_idx = torch.nonzero( is_dec_rep_msk, as_tuple = True )[0]

        # Split into inc and dec
        utils.start_timer('indec split')
        w_cnc_inc = w_cnc[ is_inc_rep_idx, : ]
        b_cnc_inc = b_cnc[ is_inc_rep_idx ]
        w_abs_inc = w_abs[ is_inc_rep_idx, : ]
        b_abs_inc = b_abs[ is_inc_rep_idx ]
        w_cnc_dec = w_cnc[ is_dec_rep_idx, : ]
        b_cnc_dec = b_cnc[ is_dec_rep_idx ]
        w_abs_dec = w_abs[ is_dec_rep_idx, : ]
        b_abs_dec = b_abs[ is_dec_rep_idx ]
        utils.record_time('indec split')

        # Get violations for inc and dec
        #if config.DEBUG:
        #    l.debug("Max inc viol: {}, max conc {}".format(
        #        torch.max( torch_relu( w_cnc_inc - w_abs_inc )),
        #        torch.max( torch.abs( w_cnc_inc )), ))
        ls_viol = torch.empty( 
                (num_nodes,), 
                dtype = config.FLOAT_TYPE_PYTORCH )
        ls_viol = ls_viol.index_put( (is_inc_rep_idx,),
                torch.max( torch_relu( w_cnc_inc - w_abs_inc ),
                                dim = -1 )[0] / torch.max( 
                            torch.abs( w_cnc_inc )) +
                torch_relu( b_cnc_inc - b_abs_inc ) / torch.max(  
                            torch.abs( b_cnc_inc )) )
        ls_viol = ls_viol.index_put( (is_dec_rep_idx,),
                torch.max( torch_relu( w_abs_dec - w_cnc_dec ),
                            dim = -1 )[0] / torch.max( 
                            torch.abs( w_cnc_dec )) +
                torch_relu( b_abs_dec - b_cnc_dec ) / torch.max(  
                            torch.abs( b_cnc_dec )) )
        #if config.DEBUG:
        #    l.debug("Max ls viol: {}".format( torch.max( ls_viol )))

        return ls_viol
