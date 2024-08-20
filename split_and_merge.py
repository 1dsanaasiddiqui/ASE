"""
Classifies neurons, splits them according to classification, and merges them
again.

NOTE: This code was originally written as a part of the re-write of the split
and merge framework. This has been heavily modified and used here.
"""


import logging as l

import numpy as np

import network 
import config



def split_net( net, out_lyr_idv = None ):
    """
    Splits a network according to classification of neurons. The network must
    have a single output neuron, which is assumed to have a pos+inc
    classification.
    
    Arguments:
    
    net             -   The Network to split
    out_lyr_idv     -   Inc-dec vects for output layer. If not given, the output
                        layer is assumed to have all incs.

    Returns: 

    1.  The network after splitting
    2.  A list of vectors for each layer storing the inc-dec classification.
        Each vector has one element for each neuron. Each element is either +1
        for inc, or -1 for dec. There is no vector for the input layer, instead
        there is None.
    """
    # Inital inc-dec-vects
    if out_lyr_idv is None:
        inc_dec_vects = [ np.array([ +1 for _ in range( net.out_size ) ]) ]
    else:
        inc_dec_vects = [ out_lyr_idv ]

    new_weights = [ w for w in net.weights ]
    new_biases = [ b for b in net.biases ]
    
    # Loop over layers in backward order
    for lyr_idx in range( net.num_layers-2, 0, -1 ):

        # Get weight and bias connected to current layer
        o_w, o_b = new_weights[ lyr_idx ], new_biases[ lyr_idx ]
        i_w, i_b = new_weights[ lyr_idx-1 ], new_biases[ lyr_idx-1 ]

        # Previous layer classifications
        prev_inc_dec = inc_dec_vects[-1]

        # New weights
        n_o_w = []
        n_i_w = []
        n_i_b = []

        # New classification
        new_inc_dec = []
        
        # Iterate over neurons
        for n_idx in range(0, net.layer_sizes[ lyr_idx ]):

            # A list of bool vects, each giving which dest neurons to connect to
            masks = []

            # Collects classifications for the split neurons 
            sn_inc_dec = []
            
            # Collect masks for out edges according to inc-dec.
            inc_mask = o_w[ n_idx ] * prev_inc_dec > 0
            if np.any( inc_mask ):
                masks.append( inc_mask )
                sn_inc_dec.append( +1 )
            if not np.all( inc_mask ):
                masks.append( np.logical_not( inc_mask ))
                sn_inc_dec.append( -1 )

            # Use the masks to get the weights after splitting
            new_inc_dec.extend( sn_inc_dec )
            for m in masks:
                n_o_w.append( np.where( m, o_w[ n_idx ], 0 ))
                n_i_w.append( i_w[ :, n_idx ] )
                n_i_b.append( i_b[ n_idx ] )

        # Set up changed weights, biases, and classification
        o_w, o_b = new_weights[ lyr_idx ], new_biases[ lyr_idx ]
        i_w, i_b = new_weights[ lyr_idx-1 ], new_biases[ lyr_idx-1 ]
        new_weights[ lyr_idx ] = np.stack( n_o_w, axis=0 )
        new_weights[ lyr_idx-1 ] = np.stack( n_i_w, axis=1 )
        new_biases[ lyr_idx-1 ] = np.array( n_i_b )
        inc_dec_vects.append( np.array( new_inc_dec ))
           
    # Return
    return (
        network.Network( new_weights, new_biases, end_relu = net.end_relu ),
        list(reversed(inc_dec_vects + [None])),
    )

    
def merge_net( net, merge_dir, inc_dec ):
    """
    Merges given network using given partitions. The merging is done in the
    order of the partitions given.

    Arguments:
    
    net         -   Network to merge
    merge_dir   -   The merge directive to merge using.
    inc_dec     -   A list of inc-dec vectors, one for each layer. Each element
                    of these vectors corresponds to one neuron, and is +1 if the
                    neuron is inc, -1 if dec.

    Returns:
    
    1.  The merged network
    """
    if config.ASSERTS:
        # Each layer index makes sense
        for lyr_idx, _ in merge_dir:
            assert 1 <= lyr_idx and lyr_idx < net.num_layers

        # Correct number of inc dec layers
        assert len(inc_dec) == net.num_layers

        # Each partition has only inc or only dec
        for lyr_idx, partition in merge_dir:
            for part in partition:
                assert (
                    np.all( inc_dec[ lyr_idx ][ part ] > 0 ) or
                    np.all( inc_dec[ lyr_idx ][ part ] < 0 )
                ) 

        # Each neuron is in exactly one partition
        for lyr_idx, partition in merge_dir:
            for n_idx in range( net.layer_sizes[ lyr_idx ] ):
                assert sum([ 1 for part in partition if n_idx in part ]) == 1

    # Copy of weights and biases
    weights = [ w for w in net.weights ]
    biases = [ b for b in net.biases ]

    # Loop over layers in partition, merge them
    for lyr_idx, partition in merge_dir:

        # Copy of old weights and biases
        o_w = weights[ lyr_idx ]
        i_w, i_b = weights[ lyr_idx-1 ], biases[ lyr_idx-1 ]

        # Allocate space for new weights and bias
        n_o_w = np.empty(( len(partition), o_w.shape[1] ), 
            dtype = config.FLOAT_TYPE_NP )
        n_i_w = np.empty(( i_w.shape[0], len(partition) ),
            dtype = config.FLOAT_TYPE_NP )
        n_i_b = np.empty(( len(partition), ), dtype = config.FLOAT_TYPE_NP )

        # Fill new weights and biases
        for i, part in enumerate( partition ):
            n_o_w[ i, : ] = np.sum( o_w[ part, : ], axis = 0 )
            merge_fn = np.amax if inc_dec[ lyr_idx ][ part[0] ] > 0 else np.amin
            n_i_w[ :, i ] = merge_fn( i_w[ :, part ], axis = 1 )
            n_i_b[ i ] = merge_fn( i_b[ part ] )

        # Set weights and biases
        weights[ lyr_idx ] = n_o_w
        weights[ lyr_idx-1 ] = n_i_w
        biases[ lyr_idx-1 ] = n_i_b 

    return network.Network( weights, biases, end_relu = net.end_relu )


    
if __name__ == "__main__":
    
    import random
    from tqdm import tqdm
    import sys
    l.basicConfig( level = l.DEBUG )

    tst_no = int( sys.argv[1] )

    if tst_no == 0:
        """
        Split and print example network
        """

        # Input network
        in_net = network.Network(
            weights = [
                np.array([  
                    [1., 2., 3.], 
                    [3., 1., 2.] 
                ]),
                np.array([  
                    [ 1., -1., -1.], 
                    [ 1.,  1., -1.],
                    [ 1., -1.,  1.],
                ]),
            ],
            biases = [
                np.array([ 0., 1., 0. ]),
                np.array([ 1., 2., 3. ]),
            ],
            end_relu = False
        )

        id_split_net, inc_dec_vects = split_net( in_net )

        l.info( "Split net: {}".format( id_split_net ))
        l.info( "Inc dec vects: {}".format( inc_dec_vects ))

    elif tst_no == 1:
        """
        Try with actual loaded networks
        """
        import property_encode
        
        net = network.load_nnet( "../networks/ACASXU_run2a_1_1_batch_2000.nnet")
        with open( "../properties/property_3.prop", 'r' ) as f:
            prop = eval( f.read() )
        
        enc_net, inp_bnds = property_encode.encode_property( net, prop )

        sp_net, id_vects = split_net( enc_net )

        l.info( "Split net: {}".format( sp_net ))
        l.info( "Inc dec vects: {}".format( id_vects ))
        l.info( "Property was: {}".format( prop ))

        
    else:
        l.error( "Unknown test no {}".format( tst_no ))
