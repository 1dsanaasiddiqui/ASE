"""
Loads properties and converts it to a form that the rest of the code can handle.
Adds extra layers to the network, etc.

"""

import logging as l

import numpy as np

from network import Network
import config



def encode_property( net, prop ):
    """
    Encodes the given property into the given network by adding a single layer
    to the network where each neuron captures wheather a corresponding equation
    in the output side of the property holds.

    The input property is of form out_lb <= out_val @ eqn <= out_ub. This is
    converted into the equisatisfiable property that all componenets of out_val'
    >= 0. In fact, a point satisfies out_lb <= out_val @ eqn <= out_ub for the
    orignial network iff all components of out_val' >= 0 for the encoded
    network.

    Properties are a conjunction of bounds on the input, and a conjunction of
    linear inequalities on the output. They are given by dicts with two things:
        
    input   -   The bounds on the input. This is a list of tuples of the index
                of the input variable, and it's bound.
    output  -   A list of tuples of a linear expression, and bound on the value
                of the expression.

    Bounds are specified by a dict that can contain two keys, "Lower" and
    "Upper", for the lower and upper bound respectively.

    Linear expressions in variables are specifiedby a list of tuples of form
    (value, variable_index), with each such tuple specifying a term of the form
    value * variable_index.

    The output property has intervals on the input side, and the output side
    property is simply that all outputs should be >= 0.

    NOTE: In the Verification performed, the check is to see if the property is
    SAT or UNSAT. Thus, if a cex exists, the property will be SAT, all the
    equations will hold, and all the output values will be > 0.

    Arguments:
    
    net     -   The network to encode the property into.
    prop    -   The property to encode.
    
    Returns:

    1.  The network with the property encoded.
    2.  The input bounds. This is a list of tuples, each tuple giving the upper
        and lower bounds of the corresponding input. If a bound is given as
        None, that bound is left unbounded.
    """
    # Check property has correct form
    if config.ASSERTS:
        assert len(prop) == 2
        assert 'input' in prop
        assert 'output' in prop
        for idx, bound in prop[ 'input' ]:
            assert idx >= 0 and idx < net.in_size
            for btype in bound.keys():
                assert btype == 'Lower' or btype == 'Upper'
        for lexpr, bound in prop[ 'output' ]:
            assert len(lexpr) >= 1
            for _, vi in lexpr:
                assert vi >= 0 and vi < net.out_size
            for btype in bound.keys():
                assert btype == 'Lower' or btype == 'Upper'

    # Copy network
    encoded_net = Network( 
        [ w for w in net.weights], 
        [ b for b in net.biases], 
        net.end_relu 
    )
    
    # Encode the input bounds
    inp_bnds = [ [None, None] for _ in range( net.in_size )]
    for idx, bound in prop[ 'input' ]:
        for btype, bval in bound.items():
            inp_bnds[ idx ][ 0 if btype == 'Lower' else 1 ] = bval
    
    # Gather linear expressions in the form out @ exprs >= lbs
    exprs = []
    lbs = []
    for lexpr, bound in prop[ 'output' ]:
        for btype, bval in bound.items():
            sign = -1 if btype == 'Upper' else 1
            row = [ 0 for _ in range( net.out_size )]
            for cf, vi in lexpr:
                row[vi] = sign * cf
            exprs.append( row )
            lbs.append( sign * bval )
    exprs = np.array( exprs )
    lbs = np.array( lbs )

    # Append this as a new layer to existing network. All values here > 0 iff
    # input satisfies property (is cex) for original network. Value of node > 0
    # iff equation holds
    encoded_net.append_layer( exprs.T, -lbs, False )

    return encoded_net, inp_bnds


if __name__ == "__main__":

    import sys
    
    tst_no = int( sys.argv[1] )
    
    l.basicConfig( level = l.DEBUG )

    if tst_no == 0:
        
        net = Network(
            weights = [
                np.array([  
                    [0., 0., 1., 0., 0.], 
                    [0., 1., 0., 1., 0.], 
                    [1., 0., 0., 0., 1.], 
                    [0., 1., 0., 1., 0.], 
                    [0., 0., 1., 0., 0.], 
                ]),
                np.array([  
                    [1., 0., 0., 0., 1.], 
                    [0., 1., 0., 1., 0.], 
                    [1., 1., 1., 0., 0.], 
                    [0., 0., 0., 1., 0.], 
                    [0., 0., 0., 0., 1.], 
                ]),
            ],
            biases = [
                np.array([ 0., 1., 2., 3., 4. ]),
                np.array([ 1., 2., 4., 5., 6. ]),
            ],
            end_relu = False
        )
        
        prop = {
            "input":
                [
                    (0, {"Lower": 0.6, "Upper": 0.6798577687}),
                    (1, {"Lower": -0.5, "Upper": 0.5}),
                    (2, {"Lower": -0.5, "Upper": 0.5}),
                    (3, {"Lower": 0.45, "Upper": 0.5}),
                    (4, {"Lower": -0.5, "Upper": -0.45}),
                ],
            "output":
                [
                    # representation of inequasion: +1*y1 -1*y0 <= 0
                    ([(1, 1), (-1, 0)], {"Upper": 0}),
                    # representation of inequasion: +1*y2 -1*y0 <= 5
                    ([(1, 2), (-1, 0)], {"Upper": 5}),
                    # representation of inequasion: -5 <= +1*y3 -1*y0 <= 0
                    ([(1, 3), (-1, 0)], {"Upper": 0, "Lower": -5}),
                    # representation of inequasion: +1*y4 -1*y0 >= 1
                    ([(1, 4), (-1, 0)], {"Lower": 1}),
                ]
        }
        
        encoded_net, inp_bnds = encode_property( net, prop )
        l.debug("Encoded net: ")
        l.debug(encoded_net)
        l.debug("inp_bnds: {}".format( inp_bnds ))

    elif tst_no == 1:
        """ TEST: Try loading all the properties """
        import os
        from network import load_nnet
        
        for net_name in os.listdir('../networks'):
            if 'ACASXU' in net_name and net_name.endswith('.nnet'):
                for prop_fname in os.listdir('../properties'):
                    if prop_fname.endswith('.prop'):
                        
                        net = load_nnet( '../networks/'+net_name )
                        with open('../properties/'+prop_fname, 'r') as f:
                            prop = eval( f.read() )
                        
                        encoded_net, inp_bnds = encode_property( net, prop )
    else:
        l.error( "No test numbered {}".format( tst_no ))




                   
    ## TEST: Encode and check something like sanity_SAT 
    #net = Network(
    #    weights = [
    #        np.array([  
    #            [0., 0., 1., 0., 0.], 
    #            [0., 1., 0., 1., 0.], 
    #            [1., 0., 0., 0., 1.], 
    #            [0., 1., 0., 1., 0.], 
    #            [0., 0., 1., 0., 0.], 
    #        ]),
    #        np.array([  
    #            [1., 0., 0., 0., 1.], 
    #            [0., 1., 0., 1., 0.], 
    #            [1., 1., 1., 0., 0.], 
    #            [0., 0., 0., 1., 0.], 
    #            [0., 0., 0., 0., 1.], 
    #        ]),
    #        np.array([  
    #            [1., 0., 0., 0., 1.], 
    #            [0., 1., 0., 1., 0.], 
    #            [1., 1., 1., 0., 2.], 
    #            [0., 0., 0., 1., 0.], 
    #            [0., 0., 2., 0., 1.], 
    #        ]),
    #    ],
    #    biases = [
    #        np.array([ 0., 1., 2., 3., 4. ]),
    #        np.array([ 1., 2., 4., 5., 6. ]),
    #        np.array([ 1., 2., 6., 5., 4. ]),
    #    ],
    #    end_relu = False
    #)

    #prop = {
    #    "input":
    #        [
    #            (0, {"Lower": 0.6, "Upper": 0.6798577687}),
    #            (1, {"Lower": -0.5, "Upper": 0.5}),
    #            (2, {"Lower": -0.5, "Upper": 0.5}),
    #            (3, {"Lower": 0.45, "Upper": 0.5}),
    #            (4, {"Lower": -0.5, "Upper": -0.45}),
    #        ],
    #    "output":
    #        [
    #    	([(1.0, 0)], {'Lower': -988888888888888.0}),
    #        ]
    #}
    #
    #encoded_net, inp_bnds, out_ub = encode_property( net, prop )
    #print("Encoded net: ")
    #print(encoded_net)
    #print("inp_bnds: {}".format( inp_bnds ))
    #print("out_ub: {}".format( out_ub ))
    #
    #from marabou_query import marabou_query
    #print("Querying")
    #rets = marabou_query( encoded_net, inp_bnds, out_ub )
    #print("Returned: ", rets)


    ## TEST: High degradation check with actual network

    # Load one dnn
    #from network import load_nnet
    #net = load_nnet('./networks/ACASXU_run2a_3_2_batch_2000.nnet')
    #
    #prop = {
    #    "input":
    #        [
    #            (0, {"Lower": 0.6, "Upper": 0.6798577687}),
    #            (1, {"Lower": -0.5, "Upper": 0.5}),
    #            (2, {"Lower": -0.5, "Upper": 0.5}),
    #            (3, {"Lower": 0.45, "Upper": 0.5}),
    #            (4, {"Lower": -0.5, "Upper": -0.45}),
    #        ],
    #    "output":
    #        [
    #    	([(1.0, 0)], {'Lower': -988888888888888.0}),
    #        ]
    #}
    #
    #encoded_net, inp_bnds, out_ub = encode_property( net, prop )
    #print("Encoded net: ")
    #print(encoded_net)
    #print("inp_bnds: {}".format( inp_bnds ))
    #print("out_ub: {}".format( out_ub ))
    #
    #from marabou_query import marabou_query
    #print("Querying")
    #rets = marabou_query( encoded_net, inp_bnds, out_ub )
    #print("Returned: ", rets)
