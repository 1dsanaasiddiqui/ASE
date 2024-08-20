"""
Loads properties and converts it to a form that the rest of the code can handle.
Adds extra layers to the network, etc.

"""



import numpy as np

from network import Network
import config



def encode_property( net, prop ):
    """
    Encodes the given property into the given network.

    Bounds are specified by a dict that can contain two keys, "Lower" and
    "Upper", for the lower and upper bound respectively.

    Linear expressions in variables are specifiedby a list of tuples of form
    (value, variable_index), with each such tuple specifying a term of the form
    value * variable_index.

    The returned network and property returned is VALID if the given property is
    UNSAT. That is, if any point satisfies the given input and output
    properties, then in the output network, that point will violate the given
    upper bound.

    Properties are a conjunction of bounds on the input, and a conjunction of
    linear inequalities on the output. They are given by dicts with two things:
        
    input   -   The bounds on the input. This is a list of tuples of the index
                of the input variable, and it's bound.
    output  -   A list of tuples of a linear expression, and bound on the value
                of the expression.

    Arguments:
    
    net     -   The network to encode the property into.
    prop    -   The property to encode.
    
    Returns:

    1.  The network with the property encoded.
    2.  The input bounds. This is a list of tuples, each tuple giving the upper
        and lower bounds of the corresponding input. If a bound is given as
        None, that bound is left unbounded.
    3.  The upper bound of the output node.
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
        # print(bound.items())
        for btype, bval in bound.items():
            inp_bnds[ idx ][ 0 if btype == 'Lower' else 1 ] = bval    
    # Gather linear expressions and upper bounds for output
    exprs = []
    ubs = []
    for lexpr, bound in prop[ 'output' ]:
        for btype, bval in bound.items():
            sign = 1 if btype == 'Upper' else -1
            row = [ 0 for _ in range( net.out_size )]
            for cf, vi in lexpr:
                row[vi] = sign * cf
            exprs.append( row )
            ubs.append( sign * bval )
    exprs = np.array( exprs )
    ubs = np.array( ubs )

    # print(type(exprs))
            
    # Append this as a new layer to existing network. Nodes of this layer
    # produce something > 0 iff an inequality is violated.
    encoded_net.append_layer( exprs.T, -ubs, True )

    # At this point, each output node should produce something > ub iff
    # correspoding inequality is violated.
    
    # Add a layer taking or over all the inequality violations of the last
    # layer. The output of this will be >0 iff any equation is violated.
    encoded_net.append_layer(
        weight = np.ones(( exprs.shape[0], 1 )),
        bias = np.zeros(( 1, )),
        relu = False
    )

    # Negate the output. If any equation is violated, this will be < 0. So, if
    # property is UNSAT, eqns are always violated, and 0 is an upper bound
    encoded_net.append_layer(
        weight = np.array([[ -1 ]]),
        bias = np.array([0]),
        relu = False
    )
    
    # The upper bound of output is 0
    out_ub = 0

    return encoded_net, inp_bnds, out_ub, exprs.T


