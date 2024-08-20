"""
Defines several functions to help compute the soundnss loss and soundness
ensuring projection function

NOTE:

    1. Compare planes algo may produce bounds corresponding to infeasible points
       if there are multiple solutions to the underlying LP. This happens
       because in such a case, for each iteration, there are multiple choices
       for some components, and only some of these choices may actually be
       feasible. However, the values produced at these infeasible points will be
       exactly the same as in the feasible optimals, so this bound will still be
       tight, and the loss correct. The only ramification of this is that in the
       ref_impl, if each iteration produces an infeasible optimal x, LP may
       still be feasible. So feasability check needs to be done seperately
       up-front.

    2.  NOTE: Regarding floating point differences due to different order of
        operations b/w various impls of dot. Using different implementations of
        dot (eg, torch.dot(a,b) vs torch.sum( a*b, -1 )) gives slightly
        different results. With float32, this hopefully shouldnt matter too
        much.
        TODO: Check this, and remove all occurrences of torch.dot for
        consistency.
"""
import logging as l
import collections

import torch
import torch.nn.functional

import config
import utils

# TODO Check directions
# TODO Replace artificial gradient dependency injection, having grads be None
# should be fine with optimizers.

"""
A simple namedtuple to contain some pre-calculated data about interval bounds
"""
IvalBounds = collections.namedtuple( 'IvalBounds', [
    'lb',           # The lower bounds
    'ub',           # The upper bounds
    'neg_lu_diff',  # lb - ub
    'cen_b',        # (lb + ub)/2
    'inc_dir',      # (ub - lb)/2
])
def get_ival_bounds( lb, ub ):
    """
    Returns the interval bounds structure for given lb, ub.
    """
    return IvalBounds(
        lb, ub,
        lb - ub,
        (ub + lb) / 2.,
        (ub - lb) / 2.,
    )


def compare_planes_ref_impl( ival, a, b, p, q ):
    """
    Converts a quantified query comparing planes to a loss function form. The
    returned loss function is 0 iff the following holds:

        forall lb <= x <= ub : x . a + b = 0 => x . p + q >= 0

    Further, the loss represents the distance this function is from holding.

    This is a reference implementation, kept around only for potential future
    debugging.

    NOTE: 

    1.  This function is entirely differenciable, and should handle torch
        tensors seamlessly.
    2.  This DOES NOT SUPPORT BROADCASTING, only pass vectors into a,p and
        scalars into b,q. 
    3.  Components of a may appear in the denominator of the returned value.
        This can have consequences on the numerical stability, especially if a
        is being optimized with gradients from the output of this value.

    Regarding potential oscillation issues:

        When optimal value calculated by ignoring different constraints are very
        close, gradients can still be drastically different. Then, a situation
        can happen where these two points get shifted in alternate way. This
        shouldn't be an issue, but if it does become an issue, one can take
        softmax.

    Returns: 

    1.  The loss.

    Note: This is a reference, known to be correct implementation
    """
    # TODO Remove to reduce tech debt 
    lb = ival.lb
    ub = ival.ub
    
    if config.ASSERTS:
        assert len( a.size() ) == 1
        assert len( b.size() ) == 0
        assert len( p.size() ) == 1
        assert len( q.size() ) == 0
        assert len( lb.size() ) == 1
        assert len( ub.size() ) == 1
        assert lb.size()[0] == ub.size()[0] 
        assert lb.size()[0] == a.size()[0] 
        assert lb.size()[0] == p.size()[0] 

    # Calcs for feasibility check
    max_a_x = torch.where( a > 0, ub, lb )
    min_a_x = torch.where( a > 0, lb, ub )
    max_a = torch.dot( max_a_x, a ) + b
    min_a = torch.dot( min_a_x, a ) + b

    # Case wehn a is 0
    if torch.allclose( a, torch.tensor(0.) ):
        utils.start_timer('Shortcut zero a')
        
        # If b is also 0, lhs of => is True, so just minimize x.p + q
        if torch.allclose( b, torch.tensor(0.) ):
            min_val = torch.dot( torch.where( p > 0, lb, ub ), p ) + q

        # Else, lhs of => is False, condition always holds, have some 
        else:
            utils.record_time('Shortcut zero a')
            return a[0] * 0.    # Cant directly put 0, hack to get back_fn

        utils.record_time('Shortcut zero a')

    # Case when a,b are not feasible, then lhs of => is true, return 0.
    # This needs to happen here, can't be inferred from general case. See notes
    # at head of file.
    elif max_a < 0 or min_a > 0:
        return a[0] * 0. # Can't directly say 0, hack to get back_fn.


    # General case
    else:

        # Get all nonzero indices
        utils.start_timer('Calculate nonzero indices')
        nz_idxs = torch.nonzero( a )[:,0]
        utils.record_time('Calculate nonzero indices')

        # Try for various nonzero idexes, collect lower bounds, take a tight one
        min_val = None
        for itr_no, nz_idx in enumerate( nz_idxs ): 

            # We do a variable change to y, x_0 = y_0 ... x_nz_idx = y . vcc +
            # vcb, x_(nz_idx+1) = y_nz_idx ... so that for all y, x . a + b = 0
            # holds.
            utils.start_timer('Calculating Variable Transform')
            vcc = torch.cat( (a[ : nz_idx ], a[ nz_idx + 1 : ]), dim = 0 )
            vcc = - vcc / a[ nz_idx ]
            vcb = - b / a[ nz_idx ]
            utils.record_time('Calculating Variable Transform')

            # Now, x.p + q = y_0 p_0 + ... (y.vcc + vcb) . p_nz_idx + y_nz_idx .
            # p_(nz_idx+1) ... + q = y.pvc + qvc
            utils.start_timer('Transforming p and q')
            pvc = torch.cat( (p[ : nz_idx ], p[ nz_idx + 1 : ]), dim = 0 )
            pvc += vcc * p[ nz_idx ]
            qvc = q + vcb * p[ nz_idx ]
            utils.record_time('Transforming p and q')

            # Truncate lb and ub
            utils.start_timer('Calculate minimal value')
            lbvc = torch.cat( (lb[ : nz_idx ], lb[ nz_idx + 1 : ]), dim = 0 )
            ubvc = torch.cat( (ub[ : nz_idx ], ub[ nz_idx + 1 : ]), dim = 0 )

            # Finally, bound x.p + q via minimizing y.pvc + qvc
            y_opt = torch.where( pvc > 0, lbvc, ubvc )
            cur_val = torch.dot( y_opt, pvc ) + qvc
            utils.record_time('Calculate minimal value')

            # Check feasibility
            x_nz = torch.dot( y_opt, vcc ) + vcb
            feas = lb[ nz_idx ] <= x_nz and x_nz <= ub[ nz_idx ]

            # Set min_val to meaningful minimum
            if min_val is None or cur_val > min_val or feas:
                min_val = cur_val

    # Return by how much minimum val is below zero
    # pytype: disable=unsupported-operands
    return torch.nn.functional.relu( -min_val ) 
    # pytype: enable=unsupported-operands

def compare_planes_single_step_batched_nochk( ival, a, b, p, q, idx, bst_idx ):
    """
    Converts a quantified query comparing planes to a loss function form. If 
    the returned loss function is 0 then the following holds:

        forall lb <= x <= ub : x . a + b = 0 => x . p + q >= 0

    Further, the returned loss represents an upper bound to a value representing
    the distance this function is from holding.

    This problem can be solved iteratively, and this function solves one step at
    a time. The state that is shared from one iteration to the next are: the

    1.  The index which has given the best bound so far - `bst_idx`,
    2.  The index which has been checked in last iteration - `idx`.

    NOTE: 

    1.  This function is entirely differenciable, and should handle torch
        tensors seamlessly.
    2.  This is a batched version, pass stacks of vectors in a, p and vectors
        into b and q. bst_idx is also a stack of indices.
    3.  Components of a may appear in the denominator of the returned value.
        This can have consequences on the numerical stability, especially if a
        is being optimized with gradients from the output of this value.
    4.  In case both the current and previous best indices produce an unbounded
        LP, some large loss is returned. This will not produce gradients to
        a,b,p,q, but if multiplied by lambda may produce gradients for lambda
        and create confusion. A warning is in place to potentially stop this.
        TODO, use zero instead.
    5.  This version DOES NOT performs feasibility checks and returns something
        meaningful even if a,b is not feasible, or some other corner case
        happens.

    Regarding potential oscillation issues:

        When optimal value calculated by ignoring different constraints are very
        close, gradients can still be drastically different. Then, a situation
        can happen where these two points get shifted in alternate way. This may
        cause oscillations.

    Returns: 

    1.  The loss.
    2.  The new value of `bst_idx`,
    """
    # NOTE: NOTATAION: 
    # 1. foo_idx means indices where foo is true. These indices index into the
    #    whole stack.
    # 2. foo_in_bar_idx means indices where foo is true within part of the stack
    #    where bar is true. These indices index into the part of the stack where
    #    bar is true. In particular, bar_idx[ foo_in_bar_idx ] == foo_and_bar_idx
    # 3. foo_mask means bools marking foo is true. This is against whole stack.
    # 4. foo_in_bar_mask means bools marking foo is true within part of the stack
    #    where bar is true. This is against the part of the stack where
    #    bar is true. In particular, foo_and_bar_mask[ bar_mask ] == foo_in_bar_mask
    # 5. val_in_foo is a stack of values where foo is true. This is for values
    #    that are undefined for the whole stack, and only defined where foo is
    #    true.
    # 6. val is a value defined for the whole stack.


    lb = ival.lb
    ub = ival.ub
    
    # Assert correct lenghts
    if config.ASSERTS:
        assert len( a.size() ) == 2
        assert len( b.size() ) == 1
        assert len( p.size() ) == 2
        assert len( q.size() ) == 1
        assert len( lb.size() ) == 1
        assert len( ub.size() ) == 1
        assert len( bst_idx.size() ) == 1
        assert lb.size()[0] == ub.size()[0] 
        assert lb.size()[0] == a.size()[1] 
        assert lb.size()[0] == p.size()[1] 
        assert a.size()[0] == b.size()[0]
        assert a.size()[0] == p.size()[0]
        assert a.size()[0] == q.size()[0]
        assert a.size()[0] == bst_idx.size()[0] 

    # Set default return values to zero with bluffed grad info. 
    utils.start_timer('nc:Init ret')
    ret_vals = (a[:,0] + b + p[:,0] + q) * 0.
    ret_bst_idx = bst_idx
    utils.record_time('nc:Init ret')

    ### Calculate with idx, check feasibility

    # Get zero-ness. 
    utils.start_timer('nc:a nz idx')
    a_cur_z_mask = torch.isclose( 
            a[ torch.arange( a.shape[0] ), idx ], torch.tensor(0.) )
    a_cur_nz_mask = torch.logical_not( a_cur_z_mask )
    a_cur_nz_idx = torch.nonzero( a_cur_nz_mask, as_tuple = True)[0]
    utils.record_time('nc:a nz idx')

    # p / a at idx
    utils.start_timer('nc:Btc pba @ cur')
    pba_in_a_cur_nz = p[ a_cur_nz_idx, idx ] / a[ a_cur_nz_idx, idx ]
    utils.record_time('nc:Btc pba @ cur')

    #if config.DEBUG and idx == 34:
    #    pba_in_a_cur_nz.register_hook( utils.grad_log_hook( 
    #        'pba_in_a_cur_nz', [ 98, None ] ))

    # We do a variable change to y, x_0 = y_0 ... x_nz_idx = y . vcc +
    # vcb, x_(idx+1) = y_nz_idx ... so that for all y, x . a + b = 0
    # Now, x.p + q = y_0 p_0 + ... (y.vcc + vcb) . p_{idx} + y_{idx} .
    # p_{idx+1} ... + q = y.pvc_in_a_cur_nz + qvc_in_a_cur_nz
    # Also, instead of explicitly not operating on the idx comp, if we do the
    # same operations on this comp, in pvc_in_a_cur_nz that comp gets set to 0, and ignored.
    # So, we can just operate on the whole vects, ignoring idx comp.
    utils.start_timer('nc:Btc p,qvc @ cur')
    pvc_in_a_cur_nz = (
            p[ a_cur_nz_idx, : ] - a[ a_cur_nz_idx, : ] * 
            torch.unsqueeze( pba_in_a_cur_nz, dim = -1 ))
    qvc_in_a_cur_nz = q[ a_cur_nz_idx ] - b[ a_cur_nz_idx ] * pba_in_a_cur_nz
    utils.record_time('nc:Btc p,qvc @ cur')

    #if config.DEBUG:
    #    qvc_in_a_cur_nz.register_hook( utils.grad_log_hook( 
    #        'pba_in_a_cur_nz', [ 98, None ] ))

    #if config.DEBUG and idx == 27:
    #    l.debug("idx for pba: {}".format( idx ))
    #    l.debug("pba_in_a_cur_nz {}".format( pba_in_a_cur_nz )) 
    #    l.debug("pvc_in_a_cur_nz {}".format( pvc_in_a_cur_nz )) 
    #    l.debug("pvc_in_a_cur_nz[27] {}".format( pvc_in_a_cur_nz[0,27] ))
        

    # Finally, bound x.p + q via minimizing y.pvc_in_a_cur_nz + qvc_in_a_cur_nz
    # y_opt represents optimization where free vars are all columns except idx,
    # but in calculations the idx column is also computed. This doesn't matter
    # since that column is 0 in pvc, so eventually doesn't contribute.
    utils.start_timer('nc:Btc min v @ idx')
    y_opt_in_a_cur_nz = torch.where( pvc_in_a_cur_nz > 0, lb, ub )
    cur_bnd_in_a_cur_nz = (
            torch.sum( y_opt_in_a_cur_nz * pvc_in_a_cur_nz, dim = -1 ) +
            qvc_in_a_cur_nz
    )
    utils.record_time('nc:Btc min v @ idx')
    
    #if config.DEBUG:
    #    l.debug( "cur_bnd: {}".format( cur_bnd_in_a_cur_nz ))
    #    l.debug( "cur_y: {}".format( y_opt_in_a_cur_nz ))

    #if config.DEBUG and idx == 27:
    #    l.debug("cur_bnd_in_a_cur_nz {}".format( cur_bnd_in_a_cur_nz )) 
    #    l.debug("y_opt_in_a_cur_nz {}".format( y_opt_in_a_cur_nz )) 

    #if config.DEBUG:
    #    cur_bnd_in_a_cur_nz.register_hook( utils.grad_log_hook(
    #            "cur_bnd_in_a_cur_nz for idx {}".format( idx ), None ))

    # Check feasibility,
    utils.start_timer('nc:Feas @ cur')
    x_opt_in_a_cur_nz = (
        -(
            torch.sum( y_opt_in_a_cur_nz * a[ a_cur_nz_idx, : ], dim = -1) 
            + b[ a_cur_nz_idx ]
        ) / a[ a_cur_nz_idx, idx ] + y_opt_in_a_cur_nz[ :, idx ]
    ) 
    cur_feas = torch.zeros( (a.shape[0],), dtype = torch.bool )
    cur_feas[ a_cur_nz_idx ] = torch.logical_and( 
            x_opt_in_a_cur_nz >= lb[ idx ], x_opt_in_a_cur_nz <= ub[ idx ] )
    utils.record_time('nc:Feas @ cur')

    del pba_in_a_cur_nz, 
    del pvc_in_a_cur_nz, 
    del qvc_in_a_cur_nz, 
    del y_opt_in_a_cur_nz, 
    del x_opt_in_a_cur_nz

    ### Calculate with bst_idx

    # Get zero-ness.
    utils.start_timer('nc:a nz bst_idx')
    a_bst_z_mask = torch.isclose( 
            a[ torch.arange( a.shape[0] ), bst_idx ], torch.tensor(0.) )
    a_bst_nz_mask = torch.logical_not( a_bst_z_mask )
    a_bst_nz_idx = torch.nonzero( a_bst_nz_mask, as_tuple = True)[0]
    bidx_in_a_bst_nz = bst_idx[ a_bst_nz_idx ]
    utils.record_time('nc:a nz bst_idx')

    if config.ASSERTS:
        assert a_bst_nz_idx.shape[0] == bidx_in_a_bst_nz.shape[0]

    # p / a at bst idx
    utils.start_timer('nc:Btc pba @ bst')
    pba_in_a_bst_nz = ( 
            p[ a_bst_nz_idx, bidx_in_a_bst_nz ] / 
            a[ a_bst_nz_idx, bidx_in_a_bst_nz ] )
    utils.record_time('nc:Btc pba @ bst')

    # We do a variable change to y, x_0 = y_0 ... x_nz_idx = y . vcc +
    # vcb, x_(bidx_in_a_bst_nz+1) = y_nz_idx ... so that for all y, x . a + b =
    # 0. Now, x.p + q = y_0 p_0 + ... (y.vcc + vcb) . p_{bidx_in_a_bst_nz} +
    # y_{bidx_in_a_bst_nz}. p_{bidx_in_a_bst_nz+1} ... + q = y.pvc + qvc
    # Also, instead of explicitly not operating on the bidx_in_a_bst_nz comp, if
    # we do the same operations on this comp, in pvc that comp gets set to 0,
    # and ignored. So, we can just operate on the whole vects, ignoring
    # bidx_in_a_bst_nz comp.
    utils.start_timer('nc:Btc p,qvc @ bst')
    pvc_in_a_bst_nz = (
            p[ a_bst_nz_idx, : ] - a[ a_bst_nz_idx, : ] * 
            torch.unsqueeze( pba_in_a_bst_nz, dim = -1 ) )
    qvc_in_a_bst_nz = q[ a_bst_nz_idx ] - b[ a_bst_nz_idx ] * pba_in_a_bst_nz 
    utils.record_time('nc:Btc p,qvc @ bst')

    #if config.DEBUG:
    #    l.debug( "pvc_in_a_bst_nz: {}".format( pvc_in_a_bst_nz ))
    #    l.debug( "qvc_in_a_bst_nz: {}".format( qvc_in_a_bst_nz ))

    # Finally, bound x.p + q via minimizing y.pvc_in_a_bst_nz + qvc_in_a_bst_nz. Indexed alonga_bst_nz_idx 
    # y_opt represents optimization where free vars are all columns except idx,
    # but in calculations the idx column is also computed. This doesn't matter
    # since that column is 0 in pvc, so eventually doesn't contribute.
    utils.start_timer('nc:Btc min v @ bst')
    y_opt_bst_in_a_bst_nz = torch.where( pvc_in_a_bst_nz > 0, lb, ub )
    bst_bnd_in_a_bst_nz = torch.sum( 
            y_opt_bst_in_a_bst_nz * pvc_in_a_bst_nz, dim = -1 ) + qvc_in_a_bst_nz 
    utils.record_time('nc:Btc min v @ bst')

    #if config.DEBUG:
    #    bst_bnd_in_a_bst_nz.register_hook( utils.grad_log_hook( 
    #        "bst_bnd_in_a_bst_nz for bst_idx {}".format( bst_idx ), None ))

    #if config.DEBUG:
    #    l.debug( "bst_bnd: {}".format( bst_bnd_in_a_bst_nz ))
    #    l.debug( "bst_y: {}".format( y_opt_bst_in_a_bst_nz ))

    # Check feasibility
    utils.start_timer('nc:Feas @ bst')
    x_opt_in_a_bst_nz = (
        -(
            torch.sum( y_opt_bst_in_a_bst_nz * a[ a_bst_nz_idx, : ], dim = -1) 
            + b[ a_bst_nz_idx ]
        ) / a[ a_bst_nz_idx, bidx_in_a_bst_nz ] + 
        y_opt_bst_in_a_bst_nz[ 
            torch.arange( a_bst_nz_idx.shape[0] ), bidx_in_a_bst_nz ]
    ) 
    bst_feas = torch.zeros( (a.shape[0],), dtype = torch.bool )
    bst_feas[ a_bst_nz_idx ] = torch.logical_and( 
            x_opt_in_a_bst_nz >= lb[ bidx_in_a_bst_nz ], 
            x_opt_in_a_bst_nz <= ub[ bidx_in_a_bst_nz ] )
    utils.record_time('nc:Feas @ bst')

    del pba_in_a_bst_nz 
    del pvc_in_a_bst_nz
    del qvc_in_a_bst_nz 
    del y_opt_bst_in_a_bst_nz 
    del x_opt_in_a_bst_nz 
    del bidx_in_a_bst_nz


    ### Update indices and bounds

    # Conditions for updating indices and bound:
    # 1. If best index is already feasible, don't update       
    # 2. If current index is feasible, do update
    # 3. If best is unbounded, do update
    # 4. If we have found best yet, do update
    # This needs to be done because, even though maintaining best bound should
    # be enough, the gradients from such calculations don't represent a feasible
    # situation, so are wrong.
    utils.start_timer('nc:Update')
    both_nz_idx = torch.nonzero( 
            torch.logical_and( a_bst_nz_mask, a_cur_nz_mask ),
            as_tuple = True )[0]
    bst_nz_in_cur_nz_idx = torch.nonzero( 
            a_bst_nz_mask[ a_cur_nz_idx ],
            as_tuple = True )[0]
    cur_nz_in_bst_nz_idx = torch.nonzero( 
            a_cur_nz_mask[ a_bst_nz_idx ],
            as_tuple = True )[0]
    if config.ASSERTS:
        assert bst_nz_in_cur_nz_idx.shape[0] == cur_nz_in_bst_nz_idx.shape[0] 
    cur_gt_bst_in_both_nz_mask = (
            cur_bnd_in_a_cur_nz[ bst_nz_in_cur_nz_idx ] >  
            bst_bnd_in_a_bst_nz[ cur_nz_in_bst_nz_idx ] )
    cur_gt_bst_mask = torch.zeros( (a.shape[0],), dtype = torch.bool )
    cur_gt_bst_mask[ both_nz_idx ] = cur_gt_bst_in_both_nz_mask
    update_mask = torch.logical_and(
            torch.logical_not( bst_feas ),
            torch.logical_or( torch.logical_or(
                cur_feas, 
                a_bst_z_mask ), 
                cur_gt_bst_mask 
            )
    )
    bst_idx[ update_mask ] = idx
    utils.record_time('nc:Update')

    #if config.DEBUG:
    #    l.debug("idx: {}".format( idx ))
    #    l.debug("cur_bnd_in_a_cur_nz: {}".format( cur_bnd_in_a_cur_nz )) 
    #    l.debug("bst_bnd_in_a_bst_nz: {}".format( bst_bnd_in_a_bst_nz )) 
    #    l.debug( "update : {}".format( update_mask ))
    #    l.debug( "cur_feas : {}".format(   cur_feas )) 
    #    l.debug( "a_bst_z : {}".format(    a_bst_z_mask ))
    #    l.debug( "cur_gt_bst : {}".format( cur_gt_bst_mask ))
    #    l.debug( "both_nz_idx : {}".format( both_nz_idx ))
    #    l.debug( "cur_bnd_in_a_cur_nz[ bst_nz_in_cur_nz_idx ] : {}".format( 
    #        cur_bnd_in_a_cur_nz[ bst_nz_in_cur_nz_idx ] ))
    #    l.debug( "bst_bnd_in_a_bst_nz[ cur_nz_in_bst_nz_idx ] : {}".format( 
    #        bst_bnd_in_a_bst_nz[ cur_nz_in_bst_nz_idx ] ))

    # NOTE Logic:
    # if update:
    #   if cur bnd exists:
    #       return from cur bnd
    #   else:
    #       return unbounded
    # else:
    #   if bst bnd exists:
    #       return from bst bnd
    #   else:
    #       return unbounded
    #   
    # Equivalently: 
    #   update  & cur bnd exists -> ret cur bnd
    #   !update & bst bnd exists -> ret bst bnd
    #   otherwise -> ret ub

    # When update is true and a_cur is nz (ie cur bnd exists and is unbounded),
    # ret val comes from cur bnd 
    utils.start_timer('nc:rets <- cur')
    update_in_a_cur_nz_mask = update_mask[ a_cur_nz_idx ]
    ret_vals = ret_vals.index_put( 
            ( a_cur_nz_idx[ update_in_a_cur_nz_mask ], ),
            torch.nn.functional.relu( 
                -cur_bnd_in_a_cur_nz[ update_in_a_cur_nz_mask ]
            )
    )
    utils.record_time('nc:rets <- cur')

    # When update is not true and a_bst_nz, ret val comes from bst bnd.
    utils.start_timer('nc:rets <- bst')
    not_update_mask = torch.logical_not( update_mask )
    not_update_in_a_bst_nz_mask = not_update_mask[ a_bst_nz_idx ]
    not_update_and_a_bst_nz_mask = torch.logical_and(
            not_update_mask, a_bst_nz_mask)
    ret_vals = ret_vals.index_put(
            ( not_update_and_a_bst_nz_mask, ),
            torch.nn.functional.relu( 
                -bst_bnd_in_a_bst_nz[ not_update_in_a_bst_nz_mask ]
            )
    )
    utils.record_time('nc:rets <- bst')
    
    # When update is not true and a_bst_z, or when update is true and a_cur_z,
    # ret val is unbounded.
    utils.start_timer('nc:rets <- ub')
    not_update_and_a_bst_z_mask = torch.logical_and(
            not_update_mask, a_bst_z_mask )
    update_and_a_cur_z_mask = torch.logical_and(
            update_mask, a_cur_z_mask )
    ret_ub_mask = torch.logical_or( 
            not_update_and_a_bst_z_mask, update_and_a_cur_z_mask )
    ret_vals = ret_vals.index_put(
            (ret_ub_mask,), torch.tensor( config.LARGE_LOSS_VAL ))
    if config.WARN and torch.any( ret_ub_mask ):
        l.warning( "Compare plane returning unbounded loss ")
    utils.record_time('nc:rets <- ub')

    return ret_vals, bst_idx


"""
Currently being used versions of compare_planes
"""
compare_planes_single_step_batched = compare_planes_single_step_batched_nochk 


def soundness_inc_single_step_batched( 
        ival, w_cnc, b_cnc, w_abs, b_abs, 
        idx, l1_bst_idx, l2_bst_idx ):
    """
    Given a set of concrete and abstract weights for a single dec neuron,
    returns the soundness loss. Implemetation is fully differenciable.

    If the loss returned is 0, then the following holds:
    
        forall lb <= x <= ub, ReLU( x.w_abs + b_abs ) >= ReLU( x.w_cnc + b_cnc )

    This uses only a single step approximation of the loss, and is an optimized
    batched implementation.
    
    Arguments:
    
    ival                    -   Interval bounds for inputs to the linear layer
    w_cnc, b_cnc            -   Weights and bias for concrete nodes as torch
                                tensors
    w_abs, b_abs            -   Weights and bias for abstract nodes as torch
                                tensors
    idx                     -   The current index for calculating the l1 and l2
                                loss components
    l1_bst_idx, l2_bst_idx  -   The best index for calculating the l1 and l2
                                loss components

    Returns: 
        1.  Soundness loss
        4.  The new l1_bst_idx
        5.  The new l2_bst_idx
    """
    lb = ival.lb
    ub = ival.ub
    cen_b = ival.cen_b
    inc_dir = ival.inc_dir

    # Assert correct dimensions
    if config.ASSERTS:
        assert len( w_cnc.size() ) == 2
        assert len( b_cnc.size() ) == 1
        assert len( w_abs.size() ) == 2
        assert len( b_abs.size() ) == 1
        assert len( lb.size() ) == 1
        assert len( ub.size() ) == 1
        assert lb.size()[0] == ub.size()[0] 
        assert lb.size()[0] == w_cnc.size()[1] 
        assert lb.size()[0] == w_abs.size()[1] 
        assert w_cnc.size()[0] == w_abs.size()[0]
        assert w_cnc.size()[0] == b_cnc.size()[0]
        assert w_cnc.size()[0] == b_abs.size()[0]
        assert idx >= 0
        assert idx < lb.shape[0]
        assert torch.all( l1_bst_idx >= 0 )
        assert torch.all( l2_bst_idx >= 0 )
        assert torch.all( l1_bst_idx < lb.shape[0] )
        assert torch.all( l2_bst_idx < lb.shape[0] )

    # Mask, indices naming convention same as
    # compare_planes_single_step_batched(). When we say cnc greater than 0,
    # forall lb <= x <= ub is considered implicit

    # We keep collecting conditions representing whether the computation has
    # continued into `cont_mask`

    #if config.DEBUG:
    #    l.debug("w_cnc: {}".format( w_cnc ))
    #    l.debug("b_cnc: {}".format( b_cnc ))
    #    l.debug("w_abs: {}".format( w_abs ))
    #    l.debug("b_abs: {}".format( b_abs ))

    # Collect return values here
    utils.start_timer("Init ret_vals")
    ret_vals = (w_cnc[:,0] + w_abs[:,0] + b_cnc + b_abs) * 0 # 0 with gradient
    utils.record_time("Init ret_vals")

    # Pre-calc min & max cnc
    utils.start_timer("minmax cnc")
    cen_cnc = torch.sum( cen_b * w_cnc, dim = -1 )
    w_cnc_av = torch.abs( w_cnc )
    inc_cnc = torch.sum( inc_dir * w_cnc_av, dim = -1 )
    max_cnc = cen_cnc + inc_cnc + b_cnc
    min_cnc = cen_cnc - inc_cnc + b_cnc
    utils.record_time("minmax cnc")

    # Pre-calc min & max amc
    utils.start_timer("minmax amc")
    w_amc = w_abs - w_cnc
    b_amc = b_abs - b_cnc
    cen_amc = torch.sum( cen_b * w_amc, dim = -1 )
    w_amc_av = torch.abs( w_amc )
    inc_amc = torch.sum( inc_dir * w_amc_av, dim = -1 )
    max_amc = cen_amc + inc_amc + b_amc
    min_amc = cen_amc - inc_amc + b_amc
    utils.record_time("minmax amc")

    # Ad-hoc addition to loss to help differenciate certain situations
    if config.ADHOC_LS:
        utils.start_timer("adhoc inc")
        min_amc_point = torch.where( w_amc < 0, ub, lb )
        abs_vals_at_min_amc = torch.sum( min_amc_point * w_abs, dim = -1 ) + b_abs
        ret_amc_mask = torch.logical_and( abs_vals_at_min_amc > 0, min_amc < 0 )
        ret_vals[ ret_amc_mask ] = -min_amc[ ret_amc_mask ]
        cont_mask = torch.logical_not( ret_vals )
        utils.record_time("adhoc inc")

    #if config.DEBUG:
    #    l.debug("Min amc: {}".format( min_amc ))
    #    l.debug("Min amc point: {}".format( min_amc_point ))
    #    l.debug("abs at min amc: {}".format( abs_vals_at_min_amc ))
    #    l.debug("Adhoc: {}".format( -min_amc[ ret_amc_mask ] ))
    #    l.debug("Cont msk: {}".format( cont_mask ))

    # Find if cnc is always 0. Then, inc loss should be whatever has been set.
    # Mask and remember
    utils.start_timer("cnc < 0")
    if config.ADHOC_LS:
        cont_mask = torch.logical_and( cont_mask, max_cnc > 0 )
    else:
        cont_mask = max_cnc > 0 
    utils.record_time("cnc < 0")

    # If abs always >= cnc, safely return whatever already set. Mask and remember.
    utils.start_timer("abs > cnc")
    cont_mask = torch.logical_and( cont_mask, min_amc < 0 )
    utils.record_time("abs > cnc")

    # If cnc is always >= 0, abs must always be greater, ie abs-cnc always >= 0
    # Mask and rembember
    utils.start_timer("cnc > 0")
    cnc_gq_0_mask = min_cnc >= 0
    ret_vals = torch.where( 
            torch.logical_and( cnc_gq_0_mask, cont_mask),
            torch.nn.functional.relu( -min_amc ),
            ret_vals
    )
    cont_mask = torch.logical_and( cont_mask, torch.logical_not( cnc_gq_0_mask ))
    utils.record_time("cnc > 0")

    # At decision boundary of concrete, abstract should be positive. Slice and
    # calc
    utils.start_timer("calc l1")
    calc_l1_mask = cont_mask
    calc_l1_idx = torch.nonzero( calc_l1_mask, as_tuple = True )[0] 
    w_cnc_in_calc_l1 = w_cnc[ calc_l1_idx, : ]
    w_abs_in_calc_l1 = w_abs[ calc_l1_idx, : ]
    b_cnc_in_calc_l1 = b_cnc[ calc_l1_idx ]
    b_abs_in_calc_l1 = b_abs[ calc_l1_idx ]
    l1_in_calc_l1, l1_bst_idx_in_calc_l1 = (
        compare_planes_single_step_batched( 
            ival, 
            w_cnc_in_calc_l1, b_cnc_in_calc_l1, 
            w_abs_in_calc_l1, b_abs_in_calc_l1, 
            idx, l1_bst_idx[ calc_l1_idx ]
    ))
    l1_bst_idx[ calc_l1_idx ] = l1_bst_idx_in_calc_l1
    utils.record_time("calc l1")

    #if config.DEBUG:
    #    l.debug("l1, calc_l1: {}, {}".format( l1_in_calc_l1, calc_l1_mask ))
    #    l.debug("Size of l1: {}".format( l1_in_calc_l1.shape ))

    # NOTE: In above, it can happen that idx skips over some indices for some
    # entries where calc_l1 is true in false in exactly one iter. 
    # TODO can this create an issue?

    if config.CALC_L2_LS:

        # If abs always <= cnc set return to l1
        utils.start_timer("abs < cnc")
        max_amc_in_calc_l1 = max_amc[ calc_l1_idx ]
        ret_l1_mask_in_calc_l1 = max_amc_in_calc_l1 <= 0
        ret_vals = ret_vals.index_put( 
                (calc_l1_idx[ ret_l1_mask_in_calc_l1 ],), 
                l1_in_calc_l1[ ret_l1_mask_in_calc_l1 ] )
        cont_mask_in_calc_l1 = torch.logical_not( ret_l1_mask_in_calc_l1 )
        utils.record_time("abs < cnc")

        # NOTE: Cont mask is false outside calc l1

        # Points where abs and conc match pre-relu, pre-relu value of abs (and
        # conc) should be negative.
        utils.start_timer("calc l2")
        calc_l2_mask_in_calc_l1 = cont_mask_in_calc_l1
        calc_l2_idx = calc_l1_idx[ calc_l2_mask_in_calc_l1 ]
        w_cnc_in_calc_l2 = w_cnc[ calc_l2_idx, : ]
        w_abs_in_calc_l2 = w_abs[ calc_l2_idx, : ]
        b_cnc_in_calc_l2 = b_cnc[ calc_l2_idx ]
        b_abs_in_calc_l2 = b_abs[ calc_l2_idx ]
        l2_in_calc_l2, l2_bst_idx_in_calc_l2 = (
            compare_planes_single_step_batched( 
                ival, 
                w_cnc_in_calc_l2 - w_abs_in_calc_l2, 
                b_cnc_in_calc_l2 - b_abs_in_calc_l2, 
                -w_abs_in_calc_l2, -b_abs_in_calc_l2, 
                idx, l2_bst_idx[ calc_l2_idx ]
        ))
        l2_bst_idx[ calc_l2_idx ] = l2_bst_idx_in_calc_l2
        cont_idx = calc_l2_idx
        utils.record_time("calc l2")

        #if config.DEBUG:
        #    l.debug("l2, calc_l2_in_calc_l1: {}, {}".format( 
        #        l2_in_calc_l2, calc_l2_mask_in_calc_l1 ))

        # Set retval to l1+l2 if l2 is calced. Both should be zero for soundness
        utils.start_timer("final asm")
        l1_in_calc_l2 = l1_in_calc_l1[ calc_l2_mask_in_calc_l1 ]
        ret_vals = ret_vals.index_put( (cont_idx,) , 
                l1_in_calc_l2 + l2_in_calc_l2 )
        utils.record_time("final asm")

    else:
        ret_vals = ret_vals.index_put( (calc_l1_idx,) , l1_in_calc_l1 )

    return ret_vals, l1_bst_idx, l2_bst_idx


def soundness_dec_single_step_batched( 
        ival, w_cnc, b_cnc, w_abs, b_abs, 
        idx, l1_bst_idx, l2_bst_idx ):
    """
    Given a set of concrete and abstract weights for a single dec neuron,
    returns the soundness loss. Implemetation is fully differenciable.

    If the loss returned is 0, then the following holds:
    
        forall lb <= x <= ub, ReLU( x.w_abs + b_abs ) >= ReLU( x.w_cnc + b_cnc )

    This uses only a single step approximation of the loss, and is an optimized
    batched implementation.
    
    Arguments:
    
    ival                    -   Interval bounds for inputs to the linear layer
    w_cnc, b_cnc            -   Weights and bias for concrete nodes as torch
                                tensors
    w_abs, b_abs            -   Weights and bias for abstract nodes as torch
                                tensors
    idx                     -   The current index for calculating the l1 and l2
                                loss components
    l1_bst_idx, l2_bst_idx  -   The best index for calculating the l1 and l2
                                loss components

    Returns: 
        1.  Soundness loss
        4.  The new l1_bst_idx
        5.  The new l2_bst_idx
    """
    lb = ival.lb
    ub = ival.ub
    cen_b = ival.cen_b
    inc_dir = ival.inc_dir

    # Assert correct dimensions
    if config.ASSERTS:
        assert len( w_cnc.size() ) == 2
        assert len( b_cnc.size() ) == 1
        assert len( w_abs.size() ) == 2
        assert len( b_abs.size() ) == 1
        assert len( lb.size() ) == 1
        assert len( ub.size() ) == 1
        assert lb.size()[0] == ub.size()[0] 
        assert lb.size()[0] == w_cnc.size()[1] 
        assert lb.size()[0] == w_abs.size()[1] 
        assert w_cnc.size()[0] == w_abs.size()[0]
        assert w_cnc.size()[0] == b_cnc.size()[0]
        assert w_cnc.size()[0] == b_abs.size()[0]
        assert idx >= 0
        assert idx < lb.shape[0]
        assert torch.all( l1_bst_idx >= 0 )
        assert torch.all( l2_bst_idx >= 0 )
        assert torch.all( l1_bst_idx < lb.shape[0] )
        assert torch.all( l2_bst_idx < lb.shape[0] )

    # Mask, indices naming convention same as
    # compare_planes_single_step_batched(). When we say cnc greater than 0,
    # forall lb <= x <= ub is considered implicit

    # We keep collecting conditions representing whether the computation has
    # continued into `cont_mask`

    # Collect return values here
    utils.start_timer("Init ret_vals")
    ret_vals = (w_cnc[:,0] + w_abs[:,0] + b_cnc + b_abs) * 0 # 0 with gradient
    utils.record_time("Init ret_vals")

    # Pre-calc min & max cnc
    utils.start_timer("minmax cnc")
    cen_cnc = torch.sum( cen_b * w_cnc, dim = -1 )
    w_cnc_av = torch.abs( w_cnc )
    inc_cnc = torch.sum( inc_dir * w_cnc_av, dim = -1 )
    max_cnc = cen_cnc + inc_cnc + b_cnc
    min_cnc = cen_cnc - inc_cnc + b_cnc
    utils.record_time("minmax cnc")

    # Pre-calc min & max amc
    utils.start_timer("minmax amc")
    w_amc = w_abs - w_cnc
    b_amc = b_abs - b_cnc
    cen_amc = torch.sum( cen_b * w_amc, dim = -1 )
    w_amc_av = torch.abs( w_amc )
    inc_amc = torch.sum( inc_dir * w_amc_av, dim = -1 )
    max_amc = cen_amc + inc_amc + b_amc
    min_amc = cen_amc - inc_amc + b_amc
    utils.record_time("minmax amc")

    # Ad-hoc addition to loss to help differenciate certain situations
    if config.ADHOC_LS:
        utils.start_timer("adhoc dec")
        max_amc_point = torch.where( w_amc > 0, ub, lb )
        cnc_vals = torch.sum( max_amc_point * w_cnc, dim = -1 ) + b_cnc
        ret_amc_mask = torch.logical_and( cnc_vals > 0, max_amc > 0 )
        ret_vals[ ret_amc_mask ] = max_amc[ ret_amc_mask ]
        cont_mask = torch.logical_not( ret_vals )
        utils.record_time("adhoc dec")

    # Find if cnc is always 0. Then, abs must also be 0. Loss is relu(max_abs).
    utils.start_timer("cnc < 0")
    if config.ADHOC_LS:
        cnc_0_mask = torch.logical_and( cont_mask, max_cnc <= 0 )
    else:
        cnc_0_mask = max_cnc <= 0
    cnc_0_idx = torch.nonzero( cnc_0_mask, as_tuple = True )[0]
    w_abs_in_cnc_0 = w_abs[ cnc_0_idx ]
    b_abs_in_cnc_0 = b_abs[ cnc_0_idx ]
    w_cnc_in_cnc_0 = w_cnc[ cnc_0_idx ]
    b_cnc_in_cnc_0 = b_cnc[ cnc_0_idx ]
    max_abs_x_in_cnc_0 = torch.where( w_abs_in_cnc_0 > 0, ub, lb )
    max_abs_in_cnc_0 = (
            torch.sum( max_abs_x_in_cnc_0 * w_abs_in_cnc_0, dim = -1  ) + 
            b_abs_in_cnc_0 
    )
    #ret_vals = ret_vals.index_put( (cnc_0_idx,), 
    #        torch.nn.functional.relu( max_abs_in_cnc_0 ) +  # Artificially add 
    #        (w_cnc_in_cnc_0[:,0] + b_cnc_in_cnc_0 ) * 0 )   # grad dep
    ret_vals = ret_vals.index_put( (cnc_0_idx,), 
            torch.nn.functional.relu( max_abs_in_cnc_0 ))
    if config.ADHOC_LS:
        cont_mask = torch.logical_and( 
                cont_mask, torch.logical_not( cnc_0_mask ) )
    else: 
        cont_mask = torch.logical_not( cnc_0_mask ) 
    utils.record_time("cnc < 0")

    # If abs always <= cnc, safely return 0. Mask and remember.
    utils.start_timer("abs < cnc")
    cont_mask = torch.logical_and( cont_mask, max_amc > 0 )
    utils.record_time("abs < cnc")

    # If cnc is always >= 0, then R(abs) < R(cnc) => abs-cnc <= 0, return it's
    # max. Mask and rembember
    utils.start_timer("cnc > 0")
    cnc_gq_0_mask = min_cnc >= 0
    ret_vals = torch.where( 
            torch.logical_and( cnc_gq_0_mask, cont_mask),
            torch.nn.functional.relu( max_amc ),
            ret_vals
    )
    cont_mask = torch.logical_and( cont_mask, torch.logical_not( cnc_gq_0_mask ))
    utils.record_time("cnc > 0")

    # At decision boundary of concrete, abstract should be positive. Slice and
    # calc
    utils.start_timer("calc l1")
    calc_l1_mask = cont_mask
    calc_l1_idx = torch.nonzero( calc_l1_mask, as_tuple = True )[0] 
    w_cnc_in_calc_l1 = w_cnc[ calc_l1_idx, : ]
    w_abs_in_calc_l1 = w_abs[ calc_l1_idx, : ]
    b_cnc_in_calc_l1 = b_cnc[ calc_l1_idx ]
    b_abs_in_calc_l1 = b_abs[ calc_l1_idx ]
    l1_in_calc_l1, l1_bst_idx_in_calc_l1 = (
        compare_planes_single_step_batched( 
            ival, 
            w_cnc_in_calc_l1, b_cnc_in_calc_l1, 
            -w_abs_in_calc_l1, -b_abs_in_calc_l1, 
            idx, l1_bst_idx[ calc_l1_idx ]
    ))
    l1_bst_idx[ calc_l1_idx ] = l1_bst_idx_in_calc_l1
    utils.record_time("calc l1")

    # NOTE: In above, it can happen that idx skips over some indices for some
    # entries where calc_l1 is true in false in exactly one iter. 
    # TODO can this create an issue?

    # If abs always >= cnc, ie min abs-cnc >= 0 set return to l1.
    utils.start_timer("abs > cnc")
    min_amc_in_calc_l1 = min_amc[ calc_l1_idx ]
    ret_l1_mask_in_calc_l1 = min_amc_in_calc_l1 >= 0
    ret_vals = ret_vals.index_put( 
            (calc_l1_idx[ ret_l1_mask_in_calc_l1 ],), 
            l1_in_calc_l1[ ret_l1_mask_in_calc_l1 ] )
    cont_mask_in_calc_l1 = torch.logical_not( ret_l1_mask_in_calc_l1 )
    utils.record_time("abs > cnc")

    # NOTE: Cont mask is false outside calc l1

    # conc should be negative.
    if config.CALC_L2_LS:
        utils.start_timer("calc l2")
        calc_l2_mask_in_calc_l1 = cont_mask_in_calc_l1
        calc_l2_idx = calc_l1_idx[ calc_l2_mask_in_calc_l1 ]
        w_cnc_in_calc_l2 = w_cnc[ calc_l2_idx, : ]
        w_abs_in_calc_l2 = w_abs[ calc_l2_idx, : ]
        b_cnc_in_calc_l2 = b_cnc[ calc_l2_idx ]
        b_abs_in_calc_l2 = b_abs[ calc_l2_idx ]
        l2_in_calc_l2, l2_bst_idx_in_calc_l2 = (
            compare_planes_single_step_batched( 
                ival, 
                w_cnc_in_calc_l2 - w_abs_in_calc_l2, 
                b_cnc_in_calc_l2 - b_abs_in_calc_l2, 
                -w_abs_in_calc_l2, -b_abs_in_calc_l2, 
                idx, l2_bst_idx[ calc_l2_idx ]
        ))
        l2_bst_idx[ calc_l2_idx ] = l2_bst_idx_in_calc_l2
        cont_idx = calc_l2_idx
        utils.record_time("calc l2")

        # Set retval to l1+l2 if l2 is calced. Both should be zero for soundness
        utils.start_timer("final asm")
        l1_in_calc_l2 = l1_in_calc_l1[ calc_l2_mask_in_calc_l1 ]
        ret_vals = ret_vals.index_put( (cont_idx,) , 
                l1_in_calc_l2 + l2_in_calc_l2 )
        utils.record_time("final asm")

    return ret_vals, l1_bst_idx, l2_bst_idx
    

def soundness_loss_layer_ssb( no_of_abs_nodes, w_cnc, b_cnc, inc_dec_rep,
        prev_lyr_hidden, w_abs, b_abs, lb, ub, idx,
        inc_l1_bst_idx, inc_l2_bst_idx, dec_l1_bst_idx, dec_l2_bst_idx,
        ):
    """
    Calculates and returns the soundness losses for an entire layer. All vectors
    should have same dim for comparision (which mostly means it should be from
    N'').

    Arguments:

    no_of_abs_nodes -   Number of abstract nodes per layer
    w_cnc, b_cnc    -   Weights and bias of concrete nodes
    inc_dec_rep     -   A vector with +1 for inc and -1 for dec nodes 
    prev_lyr_hidden -   If true, calculate as if the previous layer is a hidden
                        layer, else consider it to be the input layer
    w_abs, b_abs    -   The proposed abstract weights and biases
    lb, ub          -   lb and ub of concrete net values in prev layer
    idx             -   Current index to calculate for
    inc_l1/2_bst_idx-   Bst idx for l1 and l2 calculation for inc nodes, one for
                        each inc node
    dec_l1/2_bst_idx-   Bst idx for l1 and l2 calculation for dec nodes, one for
                        each dec node

    Returns: 
    1.  A vector with the corresponding soundness loss value for each neuron in
        N'' in that layer.
    2.  The new inc_l1_bst_idx
    3.  The new inc_l2_bst_idx
    4.  The new dec_l1_bst_idx
    5.  The new dec_l2_bst_idx
    """
    if config.DEBUG:
        l.debug( "Shapes of w_cnc {}, b_abs {}, ub {}, lb {}".format( 
                w_cnc.shape, w_abs.shape, ub.shape, lb.shape ))
    if config.ASSERTS:
        assert w_cnc.shape == w_abs.shape
        assert b_cnc.shape == b_abs.shape
        assert ub.shape[0] == lb.shape[0]
        assert ub.shape[0] == w_cnc.shape[1]
        assert inc_dec_rep.shape[0] == w_abs.shape[0]

    # Find max and min of org neuron's vals for scaling
    with torch.no_grad():
        cen_b = (ub + lb) / 2
        inc_dir = (ub - lb) / 2 
        cen_org = w_cnc @ cen_b
        w_org_av = torch.abs( w_cnc )
        inc_org = w_org_av @ inc_dir
        max_org = cen_org + inc_org + b_cnc
        min_org = cen_org - inc_org + b_cnc
        scale = torch.maximum( torch.abs( max_org ), torch.abs( min_org ))

    # Get layer sizes
    npp_curr_lyr_size = w_cnc.shape[0]
    npp_prev_lyr_size = w_cnc.shape[1]

    # Get starting loss
    sound_losses = torch.empty( npp_curr_lyr_size, 
            dtype = config.FLOAT_TYPE_PYTORCH )
    
    # Caclulate ivals
    ival = get_ival_bounds( lb, ub )

    # Split into inc and dec
    utils.start_timer('indec split')
    is_inc_rep = (inc_dec_rep > 0)
    is_inc_rep_idx = torch.nonzero( is_inc_rep, as_tuple=True )[0]
    w_cnc_inc = w_cnc[ is_inc_rep_idx, : ]
    b_cnc_inc = b_cnc[ is_inc_rep_idx ]
    w_abs_inc = w_abs[ is_inc_rep_idx, : ]
    b_abs_inc = b_abs[ is_inc_rep_idx ]
    is_dec_rep_idx = torch.nonzero( torch.logical_not( is_inc_rep ), 
            as_tuple=True )[0]
    w_cnc_dec = w_cnc[ is_dec_rep_idx, : ]
    b_cnc_dec = b_cnc[ is_dec_rep_idx ]
    w_abs_dec = w_abs[ is_dec_rep_idx, : ]
    b_abs_dec = b_abs[ is_dec_rep_idx ]
    utils.record_time('indec split')

    # Check sizes
    if config.ASSERTS:
        assert is_inc_rep_idx.shape[0] == inc_l1_bst_idx.shape[0]
        assert is_inc_rep_idx.shape[0] == inc_l2_bst_idx.shape[0]
        assert is_dec_rep_idx.shape[0] == dec_l1_bst_idx.shape[0]
        assert is_dec_rep_idx.shape[0] == dec_l2_bst_idx.shape[0]

    #if config.DEBUG:
    #    #l.debug("Comparing inc w,b: {},{} vs {},{}".format(
    #    #    w_cnc_inc, b_cnc_inc, w_abs_inc, b_abs_inc ))

    # Call inc
    utils.start_timer('inc call')
    soundness_inc, inc_l1_bst_idx, inc_l2_bst_idx = (
        soundness_inc_single_step_batched(
            ival, w_cnc_inc, b_cnc_inc, w_abs_inc, b_abs_inc, 
            idx, inc_l1_bst_idx, inc_l2_bst_idx 
    ))
    sound_losses = sound_losses.index_put( (is_inc_rep_idx,), soundness_inc )
    utils.record_time('inc call')

    # Call dec
    utils.start_timer('dec call')
    soundness_dec, dec_l1_bst_idx, dec_l2_bst_idx = (
        soundness_dec_single_step_batched(
            ival, w_cnc_dec, b_cnc_dec, w_abs_dec, b_abs_dec, 
            idx, dec_l1_bst_idx, dec_l2_bst_idx 
    ))
    sound_losses = sound_losses.index_put( (is_dec_rep_idx,), soundness_dec )
    utils.record_time('dec call')

    # Scale
    if config.SCALE_LS:
        # TODO The following assert actually fails, investigate. It should only
        # fail if we can show some orig node to be always zero
        assert( not torch.isclose( torch.min( scale ), torch.tensor(0.) ))
        sound_losses = sound_losses / scale

    return (
        sound_losses, 
        inc_l1_bst_idx, inc_l2_bst_idx, 
        dec_l1_bst_idx, dec_l2_bst_idx,
    )
                    

class SoundnessLossCalculator:
    """
    A class to calculate soundness loss. This encapsulates some state that is
    kept between two gd steps.
    
    Members:

    npp_prev_lyr_size   -   Size of the previous layer in N''
    no_of_abs_nodes     -   Number of abstract nodes 
    inc_dec_vect        -   A vector with +1 for inc and -1 for dec nodes in the
                            concrete network (that is, N)
    prev_lyr_hidden     -   If true, calculate as if the previous layer is a
                            hidden layer, else consider it to be the input layer
    idx                 -   Next index LS calculation should be started from
    inc_l1_bidx         -   
    inc_l2_bidx         -   Best indices for l1 and l2 calculations of inc nodes
    dec_l1_bidx         -   
    dec_l2_bidx         -   Best indices for l1 and l2 calculations of dec nodes
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
        # Get count of inc and dec
        inc_count = torch.count_nonzero( inc_dec_vect > 0 )
        dec_count = inc_dec_vect.shape[0] - inc_count

        # Sizes
        self.no_of_abs_nodes = no_of_abs_nodes  
        self.npp_prev_lyr_size = npp_prev_lyr_size

        #if config.DEBUG:
        #    l.debug("Making ls calc with {} inc, {} dec, {} N'' nodes".format(
        #        inc_count, dec_count, self.npp_prev_lyr_size ))

        self.prev_lyr_hidden = prev_lyr_hidden
        self.inc_dec_vect = inc_dec_vect

        # Indices
        self.idx = 0
        self.inc_l1_bidx = torch.zeros( 
                (inc_count * no_of_abs_nodes,),
                dtype = config.INDEX_TYPE_PYTORCH )
        self.inc_l2_bidx = torch.zeros( 
                (inc_count * no_of_abs_nodes,),
                dtype = config.INDEX_TYPE_PYTORCH )
        self.dec_l1_bidx = torch.zeros( 
                (dec_count * no_of_abs_nodes,),
                dtype = config.INDEX_TYPE_PYTORCH )
        self.dec_l2_bidx = torch.zeros( 
                (dec_count * no_of_abs_nodes,),
                dtype = config.INDEX_TYPE_PYTORCH )

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
        num_iters       -   Number of iterations to do. Set to -1 to do all
                            iterations.

        Returns: 
        1.  A vector with the corresponding soundness loss value for each neuron in
            N'' in that layer.
        """
        # TODO Move num_iters to constructor

        # Repeat inc dec vects
        idv = self.inc_dec_vect.repeat( self.no_of_abs_nodes )

        # Do slicing
        if sub_idxs is not None:

            if config.ASSERTS:
                assert sub_idxs.shape[0] == w_cnc.shape[0]
                assert sub_idxs.shape[0] == b_cnc.shape[0]
                assert sub_idxs.shape[0] == w_abs.shape[0]
                assert sub_idxs.shape[0] == b_abs.shape[0]

            idv = idv[ sub_idxs] 
            inc_l1_bidx = self.inc_l1_bidx[ sub_idxs ]
            inc_l2_bidx = self.inc_l2_bidx[ sub_idxs ]
            dec_l1_bidx = self.dec_l1_bidx[ sub_idxs ]
            dec_l2_bidx = self.dec_l2_bidx[ sub_idxs ]

            # Match inc-dec vects
            if config.ASSERTS and check_idv is not None and (
                    not torch.allclose( check_idv, idv ) ):
                l.error( "Inc dec vect slices don't match")
                l.error( "Sub idx: {}".format( sub_idxs ))
                l.error( "idv: {}".format( idv ))
                l.error( "check_idv: {}".format( check_idv ))
                assert False

        else:
            inc_l1_bidx = self.inc_l1_bidx
            inc_l2_bidx = self.inc_l2_bidx
            dec_l1_bidx = self.dec_l1_bidx
            dec_l2_bidx = self.dec_l2_bidx

        # Loop correct number of times
        iter_no = 0
        idx = self.idx
        has_wrapped = False
        while iter_no != num_iters and not (has_wrapped and idx == self.idx):

            # Main call
            (
                    losses,
                    inc_l1_bidx,
                    inc_l2_bidx,
                    dec_l1_bidx,
                    dec_l2_bidx,
            ) = soundness_loss_layer_ssb(
                    self.no_of_abs_nodes, 
                    w_org, b_org, 
                    idv, 
                    self.prev_lyr_hidden,
                    w_abs, b_abs,
                    lb, ub,
                    idx, 
                    inc_l1_bidx, inc_l2_bidx, 
                    dec_l1_bidx, dec_l2_bidx,
            )

            # Update and wrap indices
            iter_no += 1
            idx += 1
            if idx >= self.npp_prev_lyr_size:
                has_wrapped = True
                idx = 0

        # Update self.idx
        self.idx = idx

        # Place back 
        if sub_idxs is not None:
            self.inc_l1_bidx[ sub_idxs ] = inc_l1_bidx
            self.inc_l2_bidx[ sub_idxs ] = inc_l2_bidx
            self.dec_l1_bidx[ sub_idxs ] = dec_l1_bidx
            self.dec_l2_bidx[ sub_idxs ] = dec_l2_bidx

        else:
            self.inc_l1_bidx = inc_l1_bidx
            self.inc_l2_bidx = inc_l2_bidx
            self.dec_l1_bidx = dec_l1_bidx
            self.dec_l2_bidx = dec_l2_bidx

        return losses



if __name__ == "__main__":
    
    import sys
    import utils
    
    utils.init_logging()

    tst_no = sys.argv[1]
    
    if tst_no == 'basic_0':
        """
        Some basic situation where loss is completely violated
        """
        
        lb = torch.tensor( [0., 0., 0.] ) 
        ub = torch.tensor( [1., 1., 1.] )
        
        # Constraint
        a = torch.tensor( [1., 1., 1.] )
        b = torch.tensor( -1. ) 
        
        # Line just touching
        p = torch.tensor( [0., 1., 1.] )
        q = torch.tensor( -1. ) 

        ivals = get_ival_bounds(lb, ub)
        loss = compare_planes_ref_impl(ivals, a, b, p, q)
        l.info( "loss: {}".format( loss))
        
        assert( loss > 0 )

    elif tst_no == 'basic_1':
        """
        Situation where loss is just satisfied, but touching at an edge corner
        """
        
        lb = torch.tensor( [0., 0., 0.] ) 
        ub = torch.tensor( [1., 1., 1.] )
        
        # Constraint
        a = torch.tensor( [1., 1., 1.] )
        b = torch.tensor( -1. ) 
        
        # Line just touching
        p = torch.tensor( [0., -1., -1.] )
        q = torch.tensor( 1. ) 

        ivals = get_ival_bounds(lb, ub)
        loss = compare_planes_ref_impl(ivals, a, b, p, q)
        l.info( "loss: {}".format( loss))
        
        assert( loss <= 0 )

    elif tst_no == 'basic_2':
        """
        Situation where loss is just unsatisfied, crossing an edge at a corner.
        Takes a look at the derivatives of losss
        """
        
        lb = torch.tensor( [0., 0., 0.] ) 
        ub = torch.tensor( [1., 1., 1.] )
        
        # Constraint
        a = torch.tensor( [1., 1., 1.] )
        b = torch.tensor( -1. ) 
        
        # Line just touching
        p = torch.tensor( [0, -1., -1], requires_grad = True )
        q = torch.tensor( 0.9, requires_grad = True ) 

        ivals = get_ival_bounds(lb, ub)
        loss = compare_planes_ref_impl(ivals, a, b, p, q)
        l.info( "loss: {}".format( loss))
        
        loss.backward()
        
        l.info( "p,q grad: {}, {}".format( p.grad, q.grad ))

        # Here, the optimization will focus on [0,1,0] or [0,0,1]. Changing x
        # coord will not change value at these points. So, grad should be zero
        # at x
        assert torch.allclose(p.grad[0], torch.tensor(0.))

        # Also, to drive loss toward zero, one must increase y,z intercept,
        # which is -q/p[1,2]. To do this, since p[1] and p[2] are negative,
        # value of p[1] and p[2] must increase. Therefore grad at these should
        # be negative, corresponding to increasing loss.
        assert p.grad[1] <= 0 
        assert p.grad[2] <= 0 

        # Again to improve loss, all intercepts should increase. So, q should
        # decrease. So, gradient should be negative corresponding to increasing
        # loss
        assert q.grad <= 0

    elif tst_no == 'basic_3':
        """
        Situation where bound is just unsatisfied, crossing an edge at a corner.
        Takes a look at the derivatives of losss
        """
        
        lb = torch.tensor( [0., 0., 0.] ) 
        ub = torch.tensor( [1., 1., 1.] )
        
        # Constraint
        a = torch.tensor( [1., 1., 1.], requires_grad = True )
        b = torch.tensor( -1., requires_grad = True  ) 
        
        # Line just touching
        p = torch.tensor( [0, -1., -1.], requires_grad = True )
        q = torch.tensor( 0.9, requires_grad = True ) 

        ivals = get_ival_bounds(lb, ub)
        loss = compare_planes_ref_impl(ivals, a, b, p, q)
        l.info( "loss: {}".format( loss))
        
        loss.backward()
        
        l.info( "a,b grad: {}, {}".format( a.grad, b.grad ))
        l.info( "p,q grad: {}, {}".format( p.grad, q.grad ))

        # Here, the optimization will focus on [0,1,0] or [0,0,1]. Changing x
        # coord will not change value at these points. So, grad should be zero
        # at x
        assert torch.allclose(a.grad[0], torch.tensor(0.))
        assert torch.allclose(p.grad[0], torch.tensor(0.))

        # Also, to drive loss toward zero, one must increase y,z intercept,
        # which is -q/p[1,2]. To do this, since p[1] and p[2] are -ve and q +ve,
        # value of p[1] and p[2] must increase. Therefore grad at these should
        # be negative, corresponding to increasing loss.
        assert p.grad[1] <= 0 
        assert p.grad[2] <= 0 

        # Again to improve loss, all intercepts should increase. Since p[1,2] is
        # negative q should increase. So, gradient should be negative
        # corresponding to increasing loss
        assert q.grad <= 0

        # Similarly, y,z intercept of a,b plane should reduce. Since a[1,2] is
        # positive, this means a[1,2] should increase. So, corresponding to
        # increasing loss, grad should be negative
        assert a.grad[1] <= 0 
        assert a.grad[2] <= 0 

        # And, to drive loss to zero, all intercepts should decrease. Since
        # a[1,2] is positive, b should increase. Thus, corresponding to
        # increasing loss, grad should be negative
        assert b.grad <= 0

    elif tst_no == 'gd_from_corner':
        """
        Situation where bound is just unsatisfied, crossing an edge at a corner.
        Sets up a simple GD loop to learn a better p,q
        """
        from torch.optim import Adam
        
        lb = torch.tensor( [0., 0., 0.] ) 
        ub = torch.tensor( [1., 1., 1.] )
        ivals = get_ival_bounds(lb, ub)
        
        # Constraint
        a = torch.tensor( [1., 1., 1.], requires_grad = False )
        b = torch.tensor( -1., requires_grad = False  ) 
        
        # Line just touching
        p = torch.tensor( [0, -1., -1.1], requires_grad = True )
        q = torch.tensor( 0.9, requires_grad = True ) 

        loss = compare_planes_ref_impl(ivals, a, b, p, q)
        l.info( "loss: {}".format( loss))
        opt = Adam( [p, q] )

        i = 0
        while torch.any( loss > 0 ):
            l.info( "\n GD Iter no: {}".format( i ))
        
            # Get gradients
            loss.backward()
            l.info( "p,q grad: {}, {}".format( p.grad, q.grad ))

            # Optimize
            opt.step()
            l.info( "p,q: {}, {}".format( p, q )) 
            
            # Get new losss
            loss = compare_planes_ref_impl(ivals, a, b, p, q)
            l.info( "loss: {}".format( loss))

            i += 1
            if i > 100000:
                break

        l.info("p,q at end: {}, {}".format( p, q ))
        l.info("Took {} iterations".format( i ))

        # p,q at end should have intercepts higher than a,b
        assert -p[1] / q >= -a[1] / q
        assert -p[2] / q >= -a[2] / q

    elif tst_no == 'binsearch_infeas':
        """
        Sitchuation where a,b plane slowly becomes infeasible. We search for the
        b where it just becomes infeasible, and check that it is close enough to
        just touching bounds via binsearch
        """
        
        lb = torch.tensor( [0., 0., 0.] ) 
        ub = torch.tensor( [1., 1., 1.] )
        ivals = get_ival_bounds(lb, ub)
        
        # Constraints
        a = torch.tensor( [1., 1., 1.] )
        bl = torch.tensor( -4. ) 
        bu = torch.tensor( -1. ) 
        
        # p,q plane always zero in bounds
        p = torch.tensor( [-1., -1., -1.] )
        q = torch.tensor( -1. ) 

        # Epsilon
        eps = 0.001

        i = 0
        while bu - bl > eps:
            l.info("Iter: {}, bl,bu: {}, {}".format(i, bl, bu))

            b = (bl + bu) / 2
            loss = compare_planes_ref_impl(ivals, a, b, p, q)
            l.info( "loss: {}".format( loss))

            # If infeasible, abs val of b too high, b is too low
            if loss <= 0:
                bl = b

            # Else, abs val of b too low, b is too high
            else:
                bu = b

            i += 1

        # Exact point at -3. Assert that
        l.info("Final bl,bu after {} iterations: {},{}".format( i, bl, bu ))
        assert bl >= -3 - eps
        assert bu <= -3 + eps

    elif tst_no == 'ref_impl_time':
        """
        Runs compare_planes for a realistic size situation corresponding to the
        first loss function (checking decision boundary order) and times it.
        """
        import numpy as np
        import time
        import tqdm

        import utils

        n_conc = 100    # Number of concrete nodes
        n_abs = 15      # Number of final abstract nodes per layer

        lb = torch.zeros( (n_conc * n_abs,), dtype = config.FLOAT_TYPE_PYTORCH )
        ub = torch.ones( (n_conc * n_abs,), dtype = config.FLOAT_TYPE_PYTORCH ) * 1000
        ivals = get_ival_bounds(lb, ub)

        # Rng
        rng = np.random.default_rng(seed = config.RNG_SEED)

        # Avg over runs
        num_runs = 50
        total_time = 0
        for _ in tqdm.tqdm( range( num_runs )):
        
            # Generate random concrete weights
            w_cnc = rng.uniform( low = -1, high = 1, size = n_conc )
            b_cnc = rng.uniform( low = -1, high = 1, size = 1 )

            # Convert to torch
            w_cnc = torch.from_numpy( config.FLOAT_TYPE_NP( w_cnc ))
            b_cnc = torch.from_numpy( config.FLOAT_TYPE_NP( b_cnc ))[0]
            
            # Repeat w_cnc
            w_cnc = w_cnc.repeat( n_abs )

            # Generate random abstract weights
            w_abs = rng.uniform( low = -1, high = 1, size = n_conc * n_abs )
            b_abs = rng.uniform( low = -1, high = 1, size = 1 )

            # Convert to torch
            w_abs = torch.from_numpy( config.FLOAT_TYPE_NP( w_abs ))
            b_abs = torch.from_numpy( config.FLOAT_TYPE_NP( b_abs ))[0]

            # Turn on gradients
            w_abs.requires_grad_( True ) 
            w_cnc.requires_grad_( True )
            b_abs.requires_grad_( True )
            b_cnc.requires_grad_( True )
            
            #l.info( "Starting calculation" )
            
            utils.start_timer( 'Total' )
            loss = compare_planes_ref_impl( ivals, w_cnc, b_cnc, w_abs, b_abs )
            utils.record_time( 'Total' )

            
            #l.info( "Loss: {}".format( loss ))
            #l.info( "Grads: {}, {}, {}, {}".format( w_cnc.grad, b_cnc.grad,
            #    w_abs.grad, b_abs.grad ))
            # TODO Grads coming out none, fix / test with non-none grads

        #l.info( "Avg time: {}".format( total_time / num_runs ))
        utils.log_times()

    elif tst_no == 'single_step_batched_nochk_impr':
        """
        Fuzz compare_planes_single_step_batched_nochk and check that
        bound returned only improves.
        """
        # TODO This test is known to fail, see notes at head of file.
        from tqdm import tqdm
        
        # Rng
        rng = torch.Generator()
        rng = rng.manual_seed( 1 )

        # Scales of values in bounds and plane
        bounds_scale = 1000
        plane_scale = 100

        # Size of the planes to compare
        size = 1500
        batch_size = 100

        # Number of tries
        num = 100
        for fuzz_no in tqdm( range( num )):
            
            # Generate random lb and ub
            x = ( torch.rand( (size,), generator = rng, requires_grad = False ) 
                * 2 - 1) * bounds_scale
            y = ( torch.rand( (size,), generator = rng, requires_grad = False ) 
                * 2 - 1) * bounds_scale
            lb = torch.where( y > x, x, y )
            ub = torch.where( y > x, y, x )
            ival = get_ival_bounds( lb, ub )
            
            # Generate random a and b, make 2 copies
            a1_leaf = torch.rand( (batch_size, size), generator = rng, requires_grad = True) 
            b1_leaf = torch.rand( (batch_size,), generator = rng, requires_grad = True) 
            a1 = (a1_leaf * 2 - 1) * plane_scale
            b1 = ((b1_leaf * 2 - 1) * plane_scale)
            #l.debug("{},{}".format( a1.shape, b1.shape ))
            #l.debug("{}".format( b1.shape ))
            
            # Generate random p and q, make 2 copies
            p1_leaf = torch.rand( (batch_size,size), generator = rng, requires_grad = True) 
            q1_leaf = torch.rand( (batch_size,), generator = rng, requires_grad = True) 
            p1 = (p1_leaf * 2 - 1) * plane_scale
            q1 = ((q1_leaf * 2 - 1) * plane_scale)

            # Skip to bad test
            #if config.DEBUG and fuzz_no < 3400:
            #    continue

            #l.debug("lb {}, ub {}".format( lb, ub ))
            #l.debug("a1 {},\n b1 {},\n p1 {},\n q1 {}".format( a1, b1, p1, q1 ))

            #l.debug("New impl")

            # Call new impl
            utils.start_timer( 'New Impl' )
            bst_idx = torch.zeros( (batch_size,),
                    dtype=config.INDEX_TYPE_PYTORCH )
            prev_loss = None
            for idx in tqdm(range(size)):
                #if config.DEBUG:
                #    l.debug("prev bst_idx: {}".format( bst_idx ))
                loss, bst_idx = compare_planes_single_step_batched_nochk( 
                        ival, a1, b1, p1, q1, idx, bst_idx)
                cond = (
                        prev_loss is None or 
                        torch.all( torch.logical_or( 
                            loss <= prev_loss ,
                            torch.isclose( loss, prev_loss )
                        ))
                )
                if config.DEBUG and not cond:
                    l.debug("Non monotonicity, {}->{}, improvement {}".format(
                        prev_loss, loss, prev_loss - loss ))
                    l.debug("Is close: {}".format( 
                        torch.isclose( loss, prev_loss )))
                    l.debug("Has improved: {}".format( 
                         loss <= prev_loss ))
                    l.debug("New bst_idx: {}".format( bst_idx ))
                assert cond
                prev_loss = loss
            #l.debug("Best idx: {}".format( bst_idx )) 
            utils.record_time( 'New Impl' )

        # Print out times
        utils.log_times()


    elif tst_no == 'match_compare_planes_ssb_nochk':
        """
        Fuzz and match compare_planes_single_step_batched_nochk with reference
        impl.
        """
        import random

        from tqdm import tqdm
        
        # Rng
        rng = torch.Generator()
        rng = rng.manual_seed( config.RNG_SEED )
        random.seed( config.RNG_SEED )

        # Scales of values in bounds and plane
        bounds_scale = 1000
        plane_scale = 100

        # Size of the planes to compare
        size = 100

        # Probability of getting zeros in a,b,p,q
        sparsity = 0.5

        # Size of batch
        batch_size = 100

        # Number of iterations
        num_fuzz = 100

        for fuzz_no in tqdm( range( num_fuzz )):
            
            # Generate random lb and ub
            x = ( torch.rand( (size, ), generator = rng, requires_grad = False ) 
                * 2 - 1) * bounds_scale
            y = ( torch.rand( (size, ), generator = rng, requires_grad = False ) 
                * 2 - 1) * bounds_scale
            lb = torch.where( y > x, x, y )
            ub = torch.where( y > x, y, x )
            ival = get_ival_bounds( lb, ub )
                
            #l.info("Generating a,b")

            # Generate random a and b with certain sparsity
            a1_leaf = torch.rand( (batch_size, size), generator = rng, 
                    requires_grad = True ) 
            a2_leaf = a1_leaf.detach()
            a2_leaf.requires_grad = True
            a_zmsk = torch.where( 
                torch.rand( (batch_size, size), generator = rng ) > sparsity, 
                1., 0. )
            b1_leaf = torch.rand( 
                    (batch_size,), generator = rng, requires_grad = True) 
            b2_leaf = b1_leaf.detach()
            b2_leaf.requires_grad = True
            b_zmsk = torch.where( 
                torch.rand( (batch_size,), generator = rng ) > sparsity, 1., 0. )
            a1 = (a1_leaf * a_zmsk * 2 - 1) * plane_scale
            a2 = (a2_leaf * a_zmsk * 2 - 1) * plane_scale
            b1 = ((b1_leaf * b_zmsk * 2 - 1) * plane_scale)
            b2 = ((b2_leaf * b_zmsk * 2 - 1) * plane_scale)
                
            #l.info("Generating p,q")

            # Generate random a and b with certain sparsity
            p1_leaf = torch.rand( (batch_size, size), generator = rng, 
                    requires_grad = True ) 
            p2_leaf = p1_leaf.detach()
            p2_leaf.requires_grad = True
            p_zmsk = torch.where( 
                torch.rand( (batch_size, size), generator = rng ) > sparsity, 
                1., 0. )
            q1_leaf = torch.rand( 
                    (batch_size,), generator = rng, requires_grad = True) 
            q2_leaf = q1_leaf.detach()
            q2_leaf.requires_grad = True
            q_zmsk = torch.where( 
                torch.rand( (batch_size,), generator = rng ) > sparsity, 1., 0. )
            p1 = (p1_leaf * p_zmsk * 2 - 1) * plane_scale
            p2 = (p2_leaf * p_zmsk * 2 - 1) * plane_scale
            q1 = ((q1_leaf * q_zmsk * 2 - 1) * plane_scale)
            q2 = ((q2_leaf * q_zmsk * 2 - 1) * plane_scale)

            # Check that a,b is feasible and a is not all zero. Else mask out.
            max_a = torch.sum( a1 * torch.where( a1 > 0, ub, lb ), dim = -1 ) + b1
            min_a = torch.sum( a1 * torch.where( a1 > 0, lb, ub ), dim = -1 ) + b1
            is_feas = torch.logical_and( min_a <= 0, 0 <= max_a )
            is_nonz = torch.logical_not( torch.all( 
                torch.isclose( a1, torch.tensor(0.) ), dim=-1 ))
            mask = torch.logical_and( is_feas, is_nonz )
            a1 = a1[ mask, : ]
            p1 = p1[ mask, : ]
            b1 = b1[ mask ]
            q1 = q1[ mask ]
            a2 = a2[ mask, : ]
            p2 = p2[ mask, : ]
            b2 = b2[ mask ]
            q2 = q2[ mask ]
            cur_batch_size = a1.shape[0]

            # Call Reference
            loss1 = torch.empty( (cur_batch_size,), dtype =
                    config.FLOAT_TYPE_PYTORCH )
            utils.start_timer( 'Orig Impl' )
            for sidx in range( cur_batch_size ):
                #if config.DEBUG:
                #    l.debug("sidx: {}".format( sidx ))
                loss1[sidx] = compare_planes_ref_impl(
                        ival, a1[ sidx ], b1[ sidx ], p1[ sidx ], q1[ sidx ] )
            utils.record_time( 'Orig Impl' )
            
            # Call batched impl
            #l.info("batched impl")
            bst_idx = torch.zeros( (cur_batch_size,), 
                    dtype = config.INDEX_TYPE_PYTORCH )
            utils.start_timer( 'Batched Impl' )
            for idx in range( size ):
                loss2, bst_idx = compare_planes_single_step_batched_nochk(
                        ival, a2, b2, p2, q2, idx, bst_idx )
            utils.record_time( 'Batched Impl' )

            # Backprop
            utils.start_timer( 'Back Orig' )
            torch.sum( loss1 ).backward()
            utils.record_time( 'Back Orig' )
            utils.start_timer( 'Back batched' )
            torch.sum( loss2 ).backward()
            utils.record_time( 'Back batched' )

            #if config.DEBUG:
            #    l.debug("Loss 1: {}".format( loss1 ))
            #    l.debug("Loss 2: {}".format( loss2 ))
            #    l.debug("diff: {}".format( loss1 - loss2 ))
            #    bad_idx = torch.nonzero( torch.logical_not( torch.isclose(
            #        loss1, loss2 )))

            # Compare and assert loss same
            assert torch.allclose( loss1, loss2 )

            # Match gradients relatively closely
            assert torch.allclose( a1_leaf.grad, a2_leaf.grad, rtol = 5e-2)
            assert torch.allclose( b1_leaf.grad, b2_leaf.grad, rtol = 5e-2)
            assert torch.allclose( p1_leaf.grad, p2_leaf.grad, rtol = 5e-2)
            assert torch.allclose( q1_leaf.grad, q2_leaf.grad, rtol = 5e-2)

        # Print out times
        utils.log_times()

        # Compare times
        l.info( "Improvement in mean times: {}".format(
            utils.timers[ 'Orig Impl' ].mean / 
            utils.timers[ 'Batched Impl' ].mean
        ))
        l.info( "Improvement in max times: {}".format(
            utils.timers[ 'Orig Impl' ].max / 
            utils.timers[ 'Batched Impl' ].max
        ))

    elif tst_no == 'soundness_inc_ssb_cases':
        """
        Check cases for soundness_loss_single_step_version in a batched
        way. The corner cases checked correspond to previous tests of
        non-batched version. They, in order, are:

        0.  Look at soundness loss for a situation where concrete is always 0,

        1.  Look at soundness loss for a situation where concrete is always > 0

        2.  Look at soundness loss for a situation where abs <= cnc, but cnc is
            not fixed. That is, abs is entirely lower than conc even before
            ReLU.

        3.  Look at soundness loss for a situation where there is only decision
            boundary violation

        4.  Look at soundness loss for a situation where there is only
            intersection violation

        For details, see original tests
        """
        
        # Compare planes is old ref_impl
        compare_planes = compare_planes_ref_impl

        lb = torch.tensor( [0., 0., 0.] ) 
        ub = torch.tensor( [1., 1., 1.] )
        ivals = get_ival_bounds(lb, ub)
        
        # Cnc <= 0
        w_cnc = torch.tensor( [
                [1., 1., 1.], [1., 1., 1.], [2., 2., 2.],[-1., -1., -1.],
                [-2., -2., -2.],
            ], requires_grad = True )
        b_cnc = torch.tensor( [
                -3., 2.,-2., 1.,2.,
            ], requires_grad = True  ) 
        w_cnc_c = w_cnc.detach()
        w_cnc_c.requires_grad = True
        b_cnc_c = b_cnc.detach()
        b_cnc_c.requires_grad = True
        
        # Abs can be whatever
        w_abs = torch.tensor( [
                [-1., 2., -3.],[2., 2., 2.],[1., 1., 1.],[0, -3.99, -4.],
                [-1., -1., -1.],
            ], requires_grad = True )
        b_abs = torch.tensor( [
                4., -2., -2.,3.6,1.,
            ], requires_grad = True ) 
        w_abs_c = w_abs.detach()
        w_abs_c.requires_grad = True
        b_abs_c = b_abs.detach()
        b_abs_c.requires_grad = True

        # Call
        l1_bst_idx = torch.zeros( (w_cnc.shape[0],), dtype = config.INDEX_TYPE_PYTORCH )
        l2_bst_idx = torch.zeros( (w_cnc.shape[0],), dtype = config.INDEX_TYPE_PYTORCH )
        for idx in range( 3 ):
            loss, l1_bst_idx, l2_bst_idx = soundness_inc_single_step_batched( 
                    ivals, w_cnc, b_cnc, w_abs, b_abs,
                    idx, l1_bst_idx, l2_bst_idx)
        l.info( "loss: {}".format( loss ))

        # Call l1 for correctness check
        l1_bst_idx_c = torch.zeros( (w_cnc.shape[0],), 
                dtype = config.INDEX_TYPE_PYTORCH )
        for idx in range( 3 ):
            l1, l1_bst_idx_c = compare_planes_single_step_batched( 
                    ivals, w_cnc_c, b_cnc_c, w_abs_c, b_abs_c, 
                    idx, l1_bst_idx_c)

        assert torch.allclose( loss[0], torch.tensor(0.) )
        assert torch.allclose( loss[1], b_cnc[1] - b_abs[1] )
        # Check that l1 was triggered
        assert torch.allclose( loss[2], l1[2] )

        torch.sum( loss ).backward()
        torch.sum( l1 ).backward()
        
        l.info( "abs grad: {}, {}".format( w_abs.grad, b_abs.grad ))
        l.info( "cnc grad: {}, {}".format( w_cnc.grad, b_cnc.grad ))

        # Nomatter what abstract is, loss shouldn't change
        assert torch.allclose( w_abs.grad[0,:], torch.tensor(0.) )
        assert torch.allclose( b_abs.grad[0], torch.tensor(0.) )

        # Loss changes inverse linearly with b_abs
        assert torch.allclose( w_abs.grad[1,:], torch.tensor(0.) )
        assert torch.allclose( b_abs.grad[1], torch.tensor(-1.) )

        # Loss changes linearly with b_cnc
        assert torch.allclose( w_cnc.grad[1,:], torch.tensor(0.) )
        assert torch.allclose( b_cnc.grad[1], torch.tensor(1.) )

        # Check grads match
        assert torch.allclose( w_cnc.grad[2,:], w_cnc_c.grad[2,:] )
        assert torch.allclose( w_abs.grad[2,:], w_abs_c.grad[2,:] )
        assert torch.allclose( b_cnc.grad[2],   b_cnc_c.grad[2] )
        assert torch.allclose( b_abs.grad[2],   b_abs_c.grad[2] )

        ## 3.

        # Here, the optimization will focus on [0,1,0] or [0,0,1]. Changing x
        # coord will not change value at these points. So, grad should be zero
        # at x
        assert torch.allclose(w_cnc.grad[3,0], torch.tensor(0.))
        assert torch.allclose(w_abs.grad[3,0], torch.tensor(0.))

        # Also, to drive loss toward zero, one must increase y,z intercept, of
        # abs which is -b/w[1,2]. To do this, since w[1] and w[2] are negative,
        # and b is +ve, value of w[1] and w[2] must increase. Therefore grad at
        # these should be negative, corresponding to increasing loss.
        assert w_abs.grad[3,1] <= 0 
        assert w_abs.grad[3,2] <= 0 

        # Again to improve loss, all intercepts should increase. Since w[1,2] is
        # negative b should increase. So, gradient should be negative
        # corresponding to increasing loss
        assert b_abs.grad[3] <= 0

        # Similarly, y,z intercept of cnc plane should reduce. Since w[1,2] is
        # -ve, b +ve, this means a[1,2] should decrease. So, corresponding to
        # increasing loss, grad should be positive
        assert w_cnc.grad[3,1] >= 0 
        assert w_cnc.grad[3,2] >= 0 

        # And, to drive loss to zero, all intercepts should decrease. Since
        # w[1,2] is negative, b should decrease. Thus, corresponding to
        # increasing loss, grad should be positive
        assert b_cnc.grad[3] >= 0

        ## 4.

        # The rate of increase of the concrete neuron must be reduced. That is,
        # the absolute values of w,b_cnc must be scaled down equally. So,
        # increasing loss corresponds to reducing w_cnc and increasing b_cnc
        # equally
        assert w_cnc.grad[4,0] <= 0
        assert w_cnc.grad[4,1] <= 0
        assert w_cnc.grad[4,2] <= 0
        assert torch.allclose( w_cnc.grad[4,0], w_cnc.grad[4,1] )
        assert torch.allclose( w_cnc.grad[4,0], w_cnc.grad[4,2] )
        assert b_cnc.grad[4] >= 0 
        assert torch.allclose( w_cnc.grad[4,0], b_cnc.grad[4] )

        # The rate of increase of the abstract neuron must be increased. That is,
        # the absolute values of w,b_cnc must be scaled up equally. So,
        # increasing loss corresponds to increasing w_cnc and decreasing b_cnc
        # equally
        assert w_abs.grad[4,0] >= 0
        assert w_abs.grad[4,1] >= 0
        assert w_abs.grad[4,2] >= 0
        assert torch.allclose( w_abs.grad[4,0], w_abs.grad[4,1] )
        assert torch.allclose( w_abs.grad[4,0], w_abs.grad[4,2] )
        assert b_abs.grad[4] <= 0 
        assert torch.allclose( w_abs.grad[4,0], b_abs.grad[4] )

    elif tst_no == 'soundness_dec_ssb_cases':
        """
        Multiple tests batched, see orig test for details. They, in order, are:

        0.  Look at soundness loss dec for a situation where concrete is always
            0
        1.  Look at soundness dec loss for a situation where concrete is always
            > 0
        2.  Look at soundness dec loss for a situation where abs >= cnc, but cnc
            is not fixed.
        """
        
        # Compare planes is old ref_impl
        compare_planes = compare_planes_ref_impl

        lb = torch.tensor( [0., 0., 0.] ) 
        ub = torch.tensor( [1., 1., 1.] )
        ivals = get_ival_bounds(lb, ub)
        
        # Cnc <= 0, Cnc >= 0, Cnc not fixed
        w_cnc = torch.tensor( [ 
                [1., 1., 1.], [1., 1., 1.],[2., 2., 2.],
            ], requires_grad = True )
        b_cnc = torch.tensor(  [
                -3., 2.,-2.,
            ], requires_grad = True )
        w_cnc_c = torch.tensor(  [
                [1., 1., 1.], [1., 1., 1.],[2., 2., 2.],
            ], requires_grad = True )
        b_cnc_c = torch.tensor(  [
                -3., 2.,-2.,
            ], requires_grad = True )
        
        # Abs can be whatever, Abs most above cnc at 0,Abs > cnc
        w_abs = torch.tensor( [
                [1., 1., 1.], [-2., -2., -2.],[3., 3., 3.],
            ], requires_grad = True )
        b_abs = torch.tensor(  [
                -2., 4.,-2.,
            ], requires_grad = True )
        w_abs_c = torch.tensor(  [
                [1., 1., 1.], [-2., -2., -2.],[3., 3., 3.],
            ], requires_grad = True )
        b_abs_c = torch.tensor(  [
                -2., 4.,-2.,
            ], requires_grad = True )

        # Call
        l1_bst_idx = torch.zeros( (3,), dtype = config.INDEX_TYPE_PYTORCH )
        l2_bst_idx = torch.zeros( (3,), dtype = config.INDEX_TYPE_PYTORCH )
        for idx in range( 3 ):
            loss, l1_bst_idx, l2_bst_idx = soundness_dec_single_step_batched( 
                    ivals, w_cnc, b_cnc, w_abs, b_abs,
                    idx, l1_bst_idx, l2_bst_idx)
        l.info( "loss: {}".format( loss ))

        # Call l1 for correctness check
        l1_bst_idx_c = torch.zeros( (3,), dtype = config.INDEX_TYPE_PYTORCH )
        for idx in range( 3 ):
            l1, l1_bst_idx_c = compare_planes_single_step_batched( 
                    ivals, w_cnc_c, b_cnc_c, -w_abs_c, -b_abs_c, 
                    idx, l1_bst_idx_c)
        # Loss is abs at 1,1,1

        assert torch.allclose( loss[0], torch.sum(w_abs[0,:]) + b_abs[0] )

        # In this case, loss is diff at 0
        assert torch.allclose( loss[1], b_abs[1] - b_cnc[1] )

        # Check that l1 was triggered
        assert torch.allclose( loss[2], l1[2] )

        torch.sum( loss ).backward()
        torch.sum( l1 ).backward()
        
        l.info( "abs grad: {}, {}".format( w_abs.grad, b_abs.grad ))
        l.info( "cnc grad: {}, {}".format( w_cnc.grad, b_cnc.grad ))

        # Nomatter what cnc is, loss shouldn't change
        assert torch.allclose( w_cnc.grad[0,:], torch.tensor(0.) )
        assert torch.allclose( b_cnc.grad[0],   torch.tensor(0.) )

        # Reducing growth rate of abs or constant of abs reduces loss
        assert torch.all( w_abs.grad[0,:] > 0 )
        assert torch.all( b_abs.grad[0]   > 0 )

        # Loss changes linearly with b_abs
        assert torch.allclose( w_abs.grad[1,:], torch.tensor(0.) )
        assert torch.allclose( b_abs.grad[1],   torch.tensor(1.) )

        # Loss changes neg linearly with b_cnc
        assert torch.allclose( w_cnc.grad[1,:], torch.tensor(0.) )
        assert torch.allclose( b_cnc.grad[1],   torch.tensor(-1.) )
        
        # Check grads match
        assert torch.allclose( w_cnc.grad[2,:], w_cnc_c.grad[2,:] )
        assert torch.allclose( b_cnc.grad[2],   b_cnc_c.grad[2]   )
        assert torch.allclose( w_abs.grad[2,:], w_abs_c.grad[2,:] )
        assert torch.allclose( b_abs.grad[2],   b_abs_c.grad[2]   )

    elif tst_no == 'time_soundness_inc_ssb_nochk':
        """
        Time soundness_inc_single_step_batched with random data.
        """
        import random

        from tqdm import tqdm
        
        # Rng
        rng = torch.Generator()
        rng = rng.manual_seed( config.RNG_SEED )
        random.seed( config.RNG_SEED )

        # Scales of values in bounds and plane
        bounds_scale = 1000
        plane_scale = 100

        # Size of the planes to compare
        size = 1500

        # Probability of getting zeros in a,b,p,q. Try with different vals
        sparsity = 0.9

        # Size of batch
        batch_size = 1500

        # Number of iterations
        num_fuzz = 25

        compare_planes = compare_planes_ref_impl

        for fuzz_no in tqdm( range( num_fuzz )):
            
            # Generate random lb and ub
            x = ( torch.rand( (size, ), generator = rng, requires_grad = False ) 
                * 2 - 1) * bounds_scale
            y = ( torch.rand( (size, ), generator = rng, requires_grad = False ) 
                * 2 - 1) * bounds_scale
            lb = torch.where( y > x, x, y )
            ub = torch.where( y > x, y, x )
            ival = get_ival_bounds( lb, ub )
                
            # Generate random cnc with certain sparsity
            w_cnc_1_leaf = torch.rand( (batch_size, size), generator = rng, 
                    requires_grad = True ) 
            a_zmsk = torch.where( 
                torch.rand( (batch_size, size), generator = rng ) > sparsity, 
                1., 0. )
            b_cnc_1_leaf = torch.rand( 
                    (batch_size,), generator = rng, requires_grad = True) 
            b_zmsk = torch.where( 
                torch.rand( (batch_size,), generator = rng ) > sparsity, 1., 0. )
            w_cnc_1 = (w_cnc_1_leaf * a_zmsk * 2 - 1) * plane_scale
            b_cnc_1 = ((b_cnc_1_leaf * b_zmsk * 2 - 1) * plane_scale)
                
            # Generate random abs with certain sparsity
            w_abs_1_leaf = torch.rand( (batch_size, size), generator = rng, 
                    requires_grad = True ) 
            a_zmsk = torch.where( 
                torch.rand( (batch_size, size), generator = rng ) > sparsity, 
                1., 0. )
            b_abs_1_leaf = torch.rand( 
                    (batch_size,), generator = rng, requires_grad = True) 
            b_zmsk = torch.where( 
                torch.rand( (batch_size,), generator = rng ) > sparsity, 1., 0. )
            w_abs_1 = (w_abs_1_leaf * a_zmsk * 2 - 1) * plane_scale
            b_abs_1 = ((b_abs_1_leaf * b_zmsk * 2 - 1) * plane_scale)

            # Generate random indices
            l1_bst_idx = torch.randint( size, (batch_size,), 
                    dtype = config.INDEX_TYPE_PYTORCH, generator = rng )
            l2_bst_idx = torch.randint( size, (batch_size,), 
                    dtype = config.INDEX_TYPE_PYTORCH, generator = rng )
            idx = torch.randint( size, (1,), 
                    dtype = config.INDEX_TYPE_PYTORCH, generator = rng )[0]

            # Call
            utils.start_timer( 'New Impl' )
            loss2, l1_bst_idx, l2_bst_idx = (
                soundness_inc_single_step_batched( 
                    ival, w_cnc_1, b_cnc_1, w_abs_1, b_abs_1, 
                    idx, l1_bst_idx, l2_bst_idx )
            )
            utils.record_time( 'New Impl' )

        # Print out times
        utils.log_times()

    elif tst_no == 'degen_ref_impl':
        """
        Attempts to trigger degeneracy based feasibility issue in compare planes
        ref impl
        """

        a = torch.tensor([1., 1., 1.])
        b = torch.tensor(0)
        p = torch.tensor([1., 1., 0.])
        q = torch.tensor(0)
        lb = torch.tensor([-1., -1., -1.])
        ub = torch.tensor([1., 1., 1.])

        ival = get_ival_bounds( lb, ub )

        loss1 = compare_planes_ref_impl( ival, a,b,p,q )
        l.debug("loss1 {}".format( loss1 ))

        assert torch.allclose( loss1, torch.tensor(1.) )

    elif tst_no == 'compare_planes_ssb_nochk_grad_match_bug':
        """
        A particular situation where gradients of compare planes don't match
        """ 
        
        a1 = torch.tensor(
        [   0.0000,    0.0000,  174.9995,   41.1868,   86.3190,    0.0000,
            0.0000,  -68.7189,    0.0000,    0.0000,  -74.6114,    0.0000,
          149.8939,   55.3513, -116.1648,   31.4545, -131.2135,    0.0000,
         -105.1739,   87.7899,  153.6305,  103.8862,  -81.4627,    0.0000,
         -130.5870,  -41.2787,   81.2943, -118.5283,  179.6814,   15.0078,
         -190.7245,  -41.4169,   39.5401, -123.4384,  143.9840,    9.0300,
          -87.0230,  161.8840,   73.6701, -125.1429,  -78.5720,    0.0000,
            0.0000,   42.0793,    0.0000,  -47.1069,   48.5323,  -41.1412,
            0.0000,  -65.4377,   99.4609,   74.7185,   88.1574, -112.3071,
          102.8012,   80.6909,  154.7846,  189.9745,   24.0770,  -42.8361,
           -2.8851,   -3.7080,    0.0000,   31.9246, -151.8804,  128.2135,
         -110.4714,   -6.5054,   36.0287,   44.2938,    0.0000,    0.0000,
            0.0000,  -94.3275,   71.5057,  125.5988,    0.0000,  169.8092,
            0.0000,    0.0000,   28.7343,  -74.6369,    0.0000,    0.0000,
           97.1642,    1.5826,  -47.5474,   92.0097,  147.9630,    0.0000,
            0.0000,   34.2456,   60.6452,   42.1435,  105.0054,   62.8084,
          -84.0010, -192.7882,  -67.9104, -106.9861], requires_grad = True)
        b1 = torch.tensor( 129.61154174804688, requires_grad = True )
        p1 = torch.tensor(
        [100.0000, 100.0000,  77.8153, 100.0000, 100.0000, 100.0000, 100.0000,
         -42.4753, 100.0000, 100.0000,  25.3886, 100.0000, 100.0000, 100.0000,
         -47.4522, 100.0000, -31.2135, 100.0000, -74.6606,  96.4738, 100.0000,
         100.0000,  18.5373, 100.0000, -30.5870,  58.7213,  -6.3800, -82.3214,
         100.0000,  97.1969, -90.7245,  58.5831,  16.9229, -23.4384, 100.0000,
         100.0000,  12.9770, 100.0000, -19.9624, -77.5348,  21.4280, 100.0000,
         100.0000, 100.0000, 100.0000,  -0.5355, 100.0000, -98.3084, 100.0000,
          34.5623, 100.0000, 100.0000, 100.0000, -12.3071, 100.0000,  39.0507,
         100.0000, 100.0000, 100.0000,  57.1639,  70.1578,  96.2920, 100.0000,
          63.5432, -96.7674,  45.9855, -10.4714,  93.4946, 100.0000, 100.0000,
         100.0000, 100.0000, 100.0000,   5.6725,  99.3953,  87.8432, 100.0000,
         100.0000, 100.0000, 100.0000, -64.7011,  25.3631, 100.0000, 100.0000,
         100.0000, 100.0000,  52.4526, 100.0000, 100.0000, 100.0000, 100.0000,
         100.0000, -30.9863, 100.0000, 100.0000, 100.0000,  15.9990, -92.7882,
          32.0896, -77.2962], requires_grad = True )
        q1 = torch.tensor( 36.652992248535156, requires_grad = True )

        a2 = torch.tensor(
        [[   0.0000,    0.0000,  174.9995,   41.1868,   86.3190,    0.0000,
            0.0000,  -68.7189,    0.0000,    0.0000,  -74.6114,    0.0000,
          149.8939,   55.3513, -116.1648,   31.4545, -131.2135,    0.0000,
         -105.1739,   87.7899,  153.6305,  103.8862,  -81.4627,    0.0000,
         -130.5870,  -41.2787,   81.2943, -118.5283,  179.6814,   15.0078,
         -190.7245,  -41.4169,   39.5401, -123.4384,  143.9840,    9.0300,
          -87.0230,  161.8840,   73.6701, -125.1429,  -78.5720,    0.0000,
            0.0000,   42.0793,    0.0000,  -47.1069,   48.5323,  -41.1412,
            0.0000,  -65.4377,   99.4609,   74.7185,   88.1574, -112.3071,
          102.8012,   80.6909,  154.7846,  189.9745,   24.0770,  -42.8361,
           -2.8851,   -3.7080,    0.0000,   31.9246, -151.8804,  128.2135,
         -110.4714,   -6.5054,   36.0287,   44.2938,    0.0000,    0.0000,
            0.0000,  -94.3275,   71.5057,  125.5988,    0.0000,  169.8092,
            0.0000,    0.0000,   28.7343,  -74.6369,    0.0000,    0.0000,
           97.1642,    1.5826,  -47.5474,   92.0097,  147.9630,    0.0000,
            0.0000,   34.2456,   60.6452,   42.1435,  105.0054,   62.8084,
          -84.0010, -192.7882,  -67.9104, -106.9861]], requires_grad = True)
        b2 = torch.tensor( [129.61154174804688], requires_grad = True )
        p2 = torch.tensor(
        [[100.0000, 100.0000,  77.8153, 100.0000, 100.0000, 100.0000, 100.0000,
         -42.4753, 100.0000, 100.0000,  25.3886, 100.0000, 100.0000, 100.0000,
         -47.4522, 100.0000, -31.2135, 100.0000, -74.6606,  96.4738, 100.0000,
         100.0000,  18.5373, 100.0000, -30.5870,  58.7213,  -6.3800, -82.3214,
         100.0000,  97.1969, -90.7245,  58.5831,  16.9229, -23.4384, 100.0000,
         100.0000,  12.9770, 100.0000, -19.9624, -77.5348,  21.4280, 100.0000,
         100.0000, 100.0000, 100.0000,  -0.5355, 100.0000, -98.3084, 100.0000,
          34.5623, 100.0000, 100.0000, 100.0000, -12.3071, 100.0000,  39.0507,
         100.0000, 100.0000, 100.0000,  57.1639,  70.1578,  96.2920, 100.0000,
          63.5432, -96.7674,  45.9855, -10.4714,  93.4946, 100.0000, 100.0000,
         100.0000, 100.0000, 100.0000,   5.6725,  99.3953,  87.8432, 100.0000,
         100.0000, 100.0000, 100.0000, -64.7011,  25.3631, 100.0000, 100.0000,
         100.0000, 100.0000,  52.4526, 100.0000, 100.0000, 100.0000, 100.0000,
         100.0000, -30.9863, 100.0000, 100.0000, 100.0000,  15.9990, -92.7882,
          32.0896, -77.2962]], requires_grad = True )
        q2 = torch.tensor( [36.652992248535156], requires_grad = True )

        lb = torch.tensor(
        [-630.4538, -311.9029, -915.5430, -923.7111, -283.9785, -814.9078,
          542.7217, -815.3558, -928.1142,   53.7384, -635.9372,  244.2340,
          187.7701, -715.3638, -248.4001, -698.2053, -511.4697, -794.7787,
         -279.8553,  629.6706,  109.7727, -442.0673, -614.0283, -852.0827,
         -339.2437, -476.4526, -190.7673, -744.5529, -596.7115,  249.8647,
         -894.5000, -955.0999, -234.4296,  428.7772, -893.6027, -557.2546,
          144.1535, -319.8801, -516.2176,   60.6775,  -72.6786,  377.9011,
         -917.3248,   26.9009,  -10.2823, -717.6666, -968.7993, -547.8624,
         -739.6598, -356.0244, -884.8413, -706.2584, -721.2842,  340.4914,
         -943.9958, -425.8958, -749.4748,  650.1739, -473.1685, -564.5031,
           37.4850,  404.2306, -378.3518,  255.9770, -712.7164, -856.4229,
          440.0650, -109.0003, -279.1681, -688.5646, -718.1943, -600.5937,
         -930.3246,  630.1700, -532.3682,   -7.5213, -112.1467,  -36.2142,
         -924.1811, -835.5907,  250.9146, -521.0925, -672.0462,  -80.9766,
         -897.9653, -745.3836, -508.1107, -843.3566, -376.1570, -944.8810,
          -60.4872, -678.4510, -929.0786, -668.1984, -931.2743, -815.5593,
         -418.1591, -870.1086, -913.2919,  -48.5067])
        ub = torch.tensor(
        [-428.2061,  600.0750, -402.7960, -146.1644,  847.0099, -534.7214,
          586.0020,  168.7547,  663.5129,  603.1548,  -27.3916,  874.3725,
          627.6913,  780.6598,  680.7971,  413.8262,  470.0679,  598.5939,
          918.3214,  653.8962,  480.1999,  593.3720, -320.9872, -110.0540,
          259.2248,  891.7325,  772.4837,  360.8465, -140.6773,  415.3082,
          535.4542,  961.6769,   87.6273,  727.4036, -503.3455,  421.0464,
          859.6221,  541.0211,  503.4505,   76.4011,   57.6618,  780.4953,
          554.5784,  680.9061,  523.4414,  699.8580,   74.5777, -236.8134,
          731.2468,  766.2579,  781.8328,  468.8420,  887.3094,  978.6633,
          309.1457,  383.7104, -482.9529,  700.8832, -353.4496, -297.4655,
          207.2267,  622.7257,  312.4365,  645.0193, -324.4910, -629.4447,
          925.6891,  319.4693,  119.5703, -551.8786, -503.1670,  734.8261,
         -461.6534,  905.9171, -403.3560,  839.0414,  870.9134,   77.9480,
          -56.3118, -553.4122,  440.4838,  444.7928,  467.9191,  376.9878,
          889.4470, -187.7697,  555.1326,  150.7078,  223.4834, -855.8154,
           86.0807,  775.9684,  126.5603, -566.0527,  522.9814,   86.2159,
          899.8055,  796.7770,  131.8770,  311.6992])
        ival = get_ival_bounds( lb, ub )
        
        l1 = compare_planes_ref_impl( ival, a1, b1, p1, q1 )

        bst_idx = torch.zeros( (1,), dtype = config.INDEX_TYPE_PYTORCH )
        for idx in range(100):
            l2, bst_idx = compare_planes_single_step_batched_nochk( 
                    ival, a2, b2, p2, q2, idx, bst_idx )

        l.info("Outs: {} {}".format( l1, l2 ))
        
        l1.backward()
        l2.backward()
        
        l.info("a grads: {} {}".format( a1.grad[27], a2.grad[0, 27] ))
        l.info("p grads: {} {}".format( p1.grad[27], p2.grad[0, 27] ))
        
    elif tst_no == 'loss_not_balanced':
        """
        A situation where the losses are not balanced.
        """
        w_prm = torch.tensor([
            [  5.1167, 148.6717, 122.3130, 131.8254, 117.2804],
            [  3.7424, 149.9143, 122.3501, 131.4426, 117.1603]])
        b_prm = torch.tensor([0.6387, 0.5247, 0.6826, 0.3051, 0.4635])

        w_org = torch.tensor([  
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
            ]).T
        b_org = torch.zeros( (10,), dtype = config.FLOAT_TYPE_PYTORCH )

        w_abs = torch.repeat_interleave( w_prm, 10, dim = 1 )
        b_abs = torch.repeat_interleave( b_prm, 10, dim = 0 )

        lb = torch.tensor([-1., -1.])
        ub = torch.tensor([ 1.,  1.])

        inc_l1_bidx = torch.zeros( (50,), dtype = config.INDEX_TYPE_PYTORCH )
        inc_l2_bidx = torch.zeros( (50,), dtype = config.INDEX_TYPE_PYTORCH )
        dec_l1_bidx = torch.zeros( (0,), dtype = config.INDEX_TYPE_PYTORCH )
        dec_l2_bidx = torch.zeros( (0,), dtype = config.INDEX_TYPE_PYTORCH )
        for idx in range(2):
            (
                    losses,
                    inc_l1_bidx,
                    inc_l2_bidx,
                    dec_l1_bidx,
                    dec_l2_bidx,
            ) = soundness_loss_layer_ssb(
                    5, 
                    w_org, b_org, 
                    torch.ones( (10,), dtype = config.FLOAT_TYPE_PYTORCH ), 
                    False,
                    w_abs, b_abs,
                    lb, ub,
                    idx, 
                    inc_l1_bidx, inc_l2_bidx, 
                    dec_l1_bidx, dec_l2_bidx,
            )
            l.info( "losses: {}".format( losses ))
            l.info( "l1_bidx: {}".format( inc_l1_bidx ))
            l.info( "l2_bidx: {}".format( inc_l2_bidx ))


    elif tst_no == 'l1_bad_vals':
        """
        l1 produces bad values

        Expected value at the end is 1.9831. This may initially seem like a bad
        value, but isnt a bad value.
        """
        lb = torch.tensor([-1., -1.])
        ub = torch.tensor([ 1.,  1.])
        ival = get_ival_bounds( lb, ub )
        
        w_cnc = torch.tensor([ [ 002.    , 003.    ] ])
        b_cnc = torch.tensor([ 0. ])
        w_abs = torch.tensor([ [ 5.1167  , 3.7424  ] ])
        b_abs = torch.tensor([ 0.6387 ])

        l1_bst_idx = torch.zeros( (1,), dtype = config.INDEX_TYPE_PYTORCH )
        for idx in range(2):
            l.info("idx {}, bst_idx {}".format( idx, l1_bst_idx ))
            l1, l1_bst_idx = compare_planes_single_step_batched( 
                ival, 
                w_cnc, b_cnc, 
                w_abs, b_abs, 
                idx, l1_bst_idx
            )
            l.info("l1: {}".format( l1 ))

        #l.debug( torch.isclose( l1, torch.tensor( 1.9831 ), rtol=1e-3))
        #l.debug( l1 - 1.9831 )

        assert torch.allclose( l1, torch.tensor( 1.9831 ), rtol=1e-3)

    elif tst_no == 'loss_spike':
        """
        Investigating a loss spike that is happenning, w,b cnc and abs:
         [403.000,402.000] 0.000 [401.333,401.938] 0.000
        """
        lb = torch.tensor([-1., -1.])
        ub = torch.tensor([ 1.,  1.])
        ival = get_ival_bounds( lb, ub )

        w_cnc = torch.tensor([ [403.000,402.000] ])
        b_cnc = torch.tensor([ 0. ])
        w_abs = torch.tensor([ [401.3682, 402.0064] ])
        b_abs = torch.tensor([ 0.000 ])

        inc_l1_bidx = torch.zeros( (1,), dtype = config.INDEX_TYPE_PYTORCH )
        inc_l2_bidx = torch.zeros( (1,), dtype = config.INDEX_TYPE_PYTORCH )
        for idx in range(2):
            losses, inc_l1_bidx, inc_l2_bidx = (
                soundness_inc_single_step_batched(
                    ival, w_cnc, b_cnc, w_abs, b_abs, 
                    idx, inc_l1_bidx, inc_l2_bidx 
            ))
            l.info( "losses: {}".format( losses ))
            l.info( "l1_bidx: {}".format( inc_l1_bidx ))
            l.info( "l2_bidx: {}".format( inc_l2_bidx ))

    else:
        l.error("Unknown test no {}".format( tst_no ))
