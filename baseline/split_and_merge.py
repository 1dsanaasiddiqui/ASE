"""
Classifies neurons, splits them according to classification, and merges them
again.
"""


import numpy as np

from network import Network

import config
import utils



def split_net( net, out_lyr_idv = None ):
    """
    Splits a network according to classification of neurons. The network must
    have a single output neuron, which is assumed to have a pos+inc
    classification.
    
    Arguments:
    
    net             -   The Network to split
    out_lyr_idv   -   Inc-dec vects for output layer, optional

    Returns: 

    1.  The network after splitting
    2.  A list of vectors for each layer storing the inc-dec classification.
        Each vector has one element for each neuron. Each element is either +1
        for inc, or -1 for dec. There is no vector for the input layer, instead
        there is None.
    """
    # Inital inc-dec-vects
    if out_lyr_idv is None:
        if config.ASSERTS:
            assert net.out_size == 1
        inc_dec_vects = [ np.array([ +1 ]) ]
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
        Network( new_weights, new_biases, end_relu = net.end_relu ),
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
    #print("Merge Dir", merge_dir)

    if config.ASSERTS:
        # Each layer index makes sense
        for lyr_idx, _ in merge_dir:
            assert 1 <= lyr_idx and lyr_idx < net.num_layers

        # Correct number of inc dec layers
        assert len(inc_dec) == net.num_layers

        # Each partition has only inc or only dec
        # for lyr_idx, partition in merge_dir:
        #     for part in partition:
        #         assert (
        #             np.all( inc_dec[ lyr_idx ][ part ] > 0 ) or
        #             np.all( inc_dec[ lyr_idx ][ part ] < 0 )
        #         ) 

        # Each neuron is in exactly one partition
        for lyr_idx, partition in merge_dir:
            for n_idx in range( net.layer_sizes[ lyr_idx ] ):
                assert sum([ 1 for part in partition if n_idx in part ]) == 1

    # Copy of weights and biases
    weights = [w for w in net.weights]
    biases = [b for b in net.biases]

    # Loop over layers in partition, merge them
    for lyr_idx, partition in merge_dir:

        # Copy of old weights and biases
        o_w = weights[lyr_idx]
        i_w, i_b = weights[lyr_idx - 1], biases[lyr_idx - 1]

        # Allocate space for new weights and bias
        n_o_w = np.empty((len(partition), o_w.shape[1]), dtype=config.FLOAT_TYPE_NP)
        n_i_w = np.empty((i_w.shape[0], len(partition)), dtype=config.FLOAT_TYPE_NP)
        n_i_b = np.empty((len(partition),), dtype=config.FLOAT_TYPE_NP)

        # Fill new weights and biases
        for i, part in enumerate(partition):
            # print(part)
            n_o_w[i, :] = np.sum(o_w[part, :], axis=0)

            # Calculate maximum or minimum manually
            if inc_dec[lyr_idx][part[0]] > 0:
                merge_fn = max  # Use built-in Python max function
            else:
                merge_fn = min  # Use built-in Python min function

            # Manually calculate the merge_fn
            i_w_part = i_w[:, part]
            merge_result = np.empty((i_w_part.shape[0],))
            for j in range(i_w_part.shape[0]):
                merge_result[j] = merge_fn(i_w_part[j, :])
            n_i_w[:, i] = merge_result

            # Manually calculate the merge_fn for bias
            n_i_b[i] = merge_fn(i_b[part])

        # Set weights and biases
        weights[lyr_idx] = n_o_w
        weights[lyr_idx - 1] = n_i_w
        biases[lyr_idx - 1] = n_i_b



    return Network( weights, biases, end_relu = net.end_relu )


def merge_redundant( net ):
    """
    Merges redundant nodes together in a network
    """
    utils.start_timer('merge_redundant')
    weights = net.weights
    biases = net.biases
    
    for out_wb_idx in range( 1, len( weights ) ):

        in_wb_idx = out_wb_idx - 1
        i_w = weights[ in_wb_idx ]
        i_b = biases[ in_wb_idx ]
        o_w = weights[ out_wb_idx ]

        # Get new w,b
        n_i_w = []
        n_i_b = []
        n_o_w = []
        i_w_taken = [ False for _ in range(i_w.shape[1]) ]
        for i in range( i_w.shape[1] ):

            # If already explored, skip
            if i_w_taken[i]: continue

            # Take this as input weight
            n_i_w.append( i_w[ :, i ] )
            n_i_b.append( i_b[ i ] )

            # Find indices that are equal
            potentially_eq_idxs = np.nonzero( 
                    np.isclose( i_w[0,:], i_w[0,i] ))[0]
            b_eq_submask = np.isclose( i_b[ potentially_eq_idxs ], i_b[i] )
            potentially_eq_idxs = potentially_eq_idxs[ b_eq_submask ]
            w_eq_submask = np.all( 
                    np.isclose( i_w[ 1:, potentially_eq_idxs ], i_w[1:,[i]]), 
                    axis = 0)
            #if config.DEBUG:
            #    print( "potentially eq shape: ", potentially_eq_idxs.shape )
            #    print( "w eq shape: ", w_eq_submask.shape )
            #    print( "eq shape: ", 
            #            (i_w[ 1:, potentially_eq_idxs ] == i_w[1:,[i]]).shape )
            #    print( i_w[ 1:, potentially_eq_idxs ].shape, i_w[1:,[i]].shape )
            eq_idxs = potentially_eq_idxs[ w_eq_submask ]
            
            # Find output weight
            n_o_w.append( np.sum( o_w[ eq_idxs, : ], axis = 0 ) )

            # Mark taken array
            for j in eq_idxs: i_w_taken[ j ] = True

        # Copy back weights
        weights[ in_wb_idx ] = np.transpose( np.stack( n_i_w ) )
        biases[ in_wb_idx ] = np.stack( n_i_b )
        weights[ out_wb_idx ] = np.stack( n_o_w )

    # Construct and return network
    utils.record_time('merge_redundant')
    return Network( weights, biases, end_relu = net.end_relu )


    
if __name__ == "__main__":
    
    ## Input network
    #in_net = Network(
    #    weights = [
    #        np.array([  
    #            [1., 2., 3.], 
    #            [3., 1., 2.] 
    #        ]),
    #        np.array([  
    #            [ 1., -1., -1.], 
    #            [ 1.,  1., -1.],
    #            [ 1., -1.,  1.],
    #        ]),
    #        np.array([  
    #            [ 1.], 
    #            [-1.],
    #            [ 1.],
    #        ])
    #    ],
    #    biases = [
    #        np.array([ 0., 1., 0. ]),
    #        np.array([ 1., 2., 3. ]),
    #        np.array([ 0 ]),
    #    ],
    #    end_relu = False
    #)

    ## Get output network
    #split_net, inc_dec_vects, pos_neg_vects = split_net( in_net, True )

    ## Set up merge partitions on split network
    ##merge_dir_1 = [ (2, [ [0, 2], [1] ]) ]
    ##merge_dir_2 = [ (1, [ [0, 2, 4], [1, 3] ]) ]

    ##merge_net_1 = merge_net( split_net, merge_dir_1, inc_dec_vects )
    ##merge_net_2 = merge_net( split_net, merge_dir_2, inc_dec_vects )
    ##merge_net_12 = merge_net( merge_net_1, merge_dir_2, inc_dec_vects )
    ##merge_net_12c = merge_net( split_net, merge_dir_1 + merge_dir_2, inc_dec_vects )
    ##
    ##print("Input net")
    ##print(in_net)
    ##print("Split net") 
    ##print(split_net)
    ##print("Inc-dec classification")
    ##print(inc_dec_vects)
    ##print("Pos-neg classification")
    ##print(pos_neg_vects)
    ##print("Merge layer 2")
    ##print(merge_net_1) 
    ##print("Merge layer 1")
    ##print(merge_net_2) 
    ##print("Merge layer 1 then 2")
    ##print(merge_net_12) 
    ##print("Merge layer 1->2")
    ##print(merge_net_12c) 

    #print("Input net")
    #print(in_net)
    #print("Split net") 
    #print(split_net)

    #merge_dir = [(2, [[0], [2], [1]]), (1, [[0, 3, 6], [4], [1, 7], [2, 5]])]

    #merge_net_1 = merge_net( split_net, merge_dir[:1], inc_dec_vects )
    #print("Merge net 1")
    #print(merge_net_1) 

    #merge_net_2 = merge_net( merge_net_1, merge_dir[1:], inc_dec_vects )
    #print("Merge net 2")
    #print(merge_net_2) 

    #merge_net_3 = merge_net( split_net, merge_dir, inc_dec_vects )
    #print("Merge net 3")
    #print(merge_net_3) 

    # RANDOM TESTING

    from cegar import saturation_partition
    import random
    from tqdm import tqdm

    for _ in tqdm(range(1)):
    
        k = 100
        w1 = np.random.rand(k, k)
        b1 = np.random.rand(k)
        w2 = np.random.rand(k, k)
        b2 = np.random.rand(k)
        w3 = np.random.rand(k, 1)
        b3 = np.random.rand(1)
        net = Network( 
            weights = [w1, w2, w3], 
            biases = [b1, b2, b3], 
            end_relu = False
        )
        
        splitted_net, inc_dec_vects = split_net( net)
        
        sat_part = saturation_partition( inc_dec_vects )
        
        merge_dir = []
        for lyr_idx, partition in sat_part:
            new_partition = []
            for part in partition:
                n_splits = random.randint(1, len(part))
                splits = [ [] for _ in range(n_splits) ]
                for nidx in part:
                    splits[ random.randrange( n_splits ) ].append( nidx )
                new_partition.extend( [ s for s in splits if len(s) > 0 ] )
            merge_dir.append(( lyr_idx, new_partition ))
        
        #print("merge_dir", merge_dir)
        merged_net = merge_net( splitted_net, merge_dir, inc_dec_vects )
        
        # rand_inp = np.random.rand( 1000, k )
        # assert np.all( 
        #     merged_net.eval( rand_inp )[0][0] >= splitted_net.eval( rand_inp )[0][0]
        # ) 
