"""
Encodes a neural network verification problem as a Marabou query and solves it.
"""



import sys
import math
import logging as l

import numpy as np

import config

# Load marabou

from maraboupy import MarabouCore
from maraboupy import Marabou



def marabou_query( net, inp_bnds, out_ub ):
    """
    Given a network with a single output, input-side properties and output side
    properties, this uses Marabou to query weather the given property is valid.
    
    Arguments:

    net      -   The network. Must have exactly one output.
    inp_bnds -   The input side bounds. This is a list of tuples, each tuple
                 giving the upper and lower bounds of the corresponding input. If
                 a bound is given as None, that bound is left unbounded.
    out_ub   -   The upper bound on the output.

    Returns a counterexample as an input vector if it exists, none otherwise.
    """
    # if config.DEBUG:
    #     #l.debug(net.weights)
    #     l.debug(net.biases)

    # Variables are given by inputs, then the pre_relu and post_relu for each
    # layer, ending with the outputs
    num_vars = net.in_size
    num_vars += sum( net.layer_sizes[ 1 : -1 ] * 2 )
    num_vars += net.out_size * 2 if net.end_relu else net.out_size
    
    # Create input query
    inputQuery = MarabouCore.InputQuery()
    inputQuery.setNumberOfVariables( num_vars )

    # Set up bounds on inputs
    for v_idx, (lb, ub) in enumerate(inp_bnds):
        if lb is not None:
            inputQuery.setLowerBound( v_idx, lb )
        if ub is not None:
            inputQuery.setUpperBound( v_idx, ub )

    temp = [0.639928936958313,0.0, 0.0, 0.4749999940395355,-0.4749999940395355]

    # for v_idx in range(len(inp_bnds)):
    #     eqn = MarabouCore.Equation()
    #     eqn.addAddend(1, v_idx)
    #     eqn.setScalar(temp[v_idx])
    #     inputQuery.addEquation( eqn )

    # Encode network layer by layer
    pre_var_base = 0
    pst_var_base = net.in_size
    for w, b in zip( net.weights[:-1], net.biases[:-1] ):

        # Loop over equations and add
        for dv_idx, (col, bias) in enumerate( zip( w.T, b )):
            eqn = MarabouCore.Equation()
            eqn.addAddend( -1, pst_var_base + dv_idx )
            for sv_idx, term in enumerate( col ):
                eqn.addAddend( term, pre_var_base + sv_idx )
            eqn.setScalar( -bias )
            inputQuery.addEquation( eqn )

        # Shift variable bases
        pre_var_base = pst_var_base
        pst_var_base += w.shape[1]

        # Relu Constraints
        for v_idx in range( w.shape[1] ):
            MarabouCore.addReluConstraint( 
                inputQuery, pre_var_base + v_idx, pst_var_base + v_idx )

        # Shift variable bases
        pre_var_base = pst_var_base
        pst_var_base += w.shape[1]

    # Encode weights and bias of last layer
    w, b = net.weights[-1], net.biases[-1]
    for dv_idx, (col, bias) in enumerate( zip( w.T, b )):
        eqn = MarabouCore.Equation()
        eqn.addAddend( -1, pst_var_base + dv_idx )
        for sv_idx, term in enumerate( col ):
            eqn.addAddend( term, pre_var_base + sv_idx )
        eqn.setScalar( -bias )
        inputQuery.addEquation( eqn )
        if out_ub is not None:
            inputQuery.setLowerBound( pst_var_base + dv_idx, out_ub)

    # Encode relu at end if needed
    #assert not(net.end_relu)

    # Check that pst_var_base is correct
    print(out_ub)

    if config.DEBUG:
       MarabouCore.saveQuery( inputQuery, '/tmp/query2' )

    # Run marabou
    options = Marabou.createOptions()
    exit_code, model, stats = MarabouCore.solve(inputQuery, options, "")

    # Return input vector for cex
    if config.DEBUG:
        l.debug("Marabou exit code: {}".format( exit_code ))
    if exit_code == 'sat':

        # Check for a weird situation where model can have nans
        if config.ASSERTS:
            if any([ math.isnan( model[ v_idx ]) for v_idx in range( num_vars ) ]):
                if config.DEBUG:
                    l.debug("Network producing nans: {}".formatT( net ))
                    l.debug("Values: {}".format( model ))
                    l.debug("Exit_code: {}".format( exit_code ))
                    l.debug("stats: {}".format( stats ))
                assert False
        
        cex = np.array([ model[ v_idx ] for v_idx in range( net.in_size ) ])
        if config.DEBUG:
            l.debug("Net: {}".format( net ))
            l.debug("Cex: {}".format( cex ))
            l.debug("Eval: {}".format( net.eval( cex )))
            l.debug("Model: {}".format( model))
            l.debug("out_ub: {}".format( out_ub))
            #l.debug( net.eval( cex )[0][0] >= out_ub )
        # if config.ASSERTS:
        #     assert net.eval( cex )[0][0] + config.FLOAT_TOL >= out_ub
        return cex
    elif exit_code == 'unsat':
        return None
    else:
        raise RuntimeError("Unknown Marabou exit code: {}".format( exit_code ))
