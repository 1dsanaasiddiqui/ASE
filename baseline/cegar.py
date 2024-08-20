"""
Implements the original merge upto saturation and counterexample guided
refinement
"""


import numpy as np

import config


import torch


class LinkageTree:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

def get_leaf_nodes_linkage_tree(root, leaf_node):

    if root.left is None and root.right is None:
        leaf_node[root.value] = [root.value]
        return [root.value]

    left_leaves = get_leaf_nodes_linkage_tree(root.left,leaf_node)
    right_leaves = get_leaf_nodes_linkage_tree(root.right,leaf_node)
    leaf_node[root.value] = left_leaves + right_leaves
    return left_leaves + right_leaves

def compute_gradient( net, inp ):
    """
    Evaluates the network on the given input `inp`. Input can also be a
    stack of vectors.
    
    Returns:
    
    1.  The vector of return values
    2.  A list with the values at each layer. At each layer, if the layer
        has a ReLU, the value is the output of the ReLU, or if the layer is
        an input layer, the value is the input, and otherwise the value is
        the output of the linear layer.
    """
    cval = torch.tensor(inp, requires_grad = True,dtype=torch.float32)
    #print("Shape",cval.shape)
    vals = [cval]

    relu = torch.nn.ReLU()

    #print('Cval: ', cval)
    #print('Cval req grad: ', cval.requires_grad)

    # Evaluate inner layers
    count = 0
    for w, b in zip(net.weights[:-1], net.biases[:-1]):
        count = count+1
        cval = relu(cval @ torch.from_numpy(w).float() + torch.from_numpy(b).float())
        vals.append(cval)

    # Evaluate last layer
    cval = cval @ torch.from_numpy(net.weights[-1]).float() + torch.from_numpy(net.biases[-1]).float()
    if net.end_relu:
        cval = relu(cval)
    vals.append(cval)

    vals[-1][0].backward(inputs = vals)

    grads = [ v.grad.numpy() for v in vals ]
    
    #print('grads: ', grads )

    return grads




def saturation_partition( inc_dec_vects, merge_first_lyr = False):
    """
    Given inc-dec and pos-neg classifications, generate partitions for merging
    upto saturation. 
    
    Arguments:

    inc_dec_vects   -   A list of vectors for each layer storing the inc-dec
                        classification.  Each vector has one element for each
                        neuron. Each element is either +1 for inc, or -1 for
                        dec. There is no vector for the input layer, instead
                        there is None.
    merge_first_lyr -   If true, the first layer is merged, otherwise it is not.

    Returns: A merge directive representing merging to saturation.
    """
    merge_dir = []
    
    # Loop over layers
    for lyr_idx in range( len(inc_dec_vects) - 2, 0, -1 ):

        # Collects masks for each partition
        masks = []

        # Masks from inc-dec
        inc_dec = inc_dec_vects[ lyr_idx ]
        masks.append( inc_dec > 0 )
        masks.append( inc_dec < 0 )
        
        # Add partition list
        merge_dir.append( ( lyr_idx,
            [ np.nonzero( m )[0] for m in masks if np.any( m ) ] ) )
    
    return merge_dir


def get_new_merge_dir(conc_net, culprit_neuron, merge_dir):

    split_set = set()
    split_set.add( culprit_neuron )
   
    new_merge_dir = []
    for lyr_idx, partition in merge_dir:
        if lyr_idx!= culprit_neuron[0]:
            new_partition = partition
        else:            
            new_partition = []
            for part in partition:
                rem_part = [ i for i in part if (lyr_idx,i) not in split_set ]
                if len(rem_part) > 0:
                    new_partition.append( rem_part )
                new_partition.extend([ 
                    [i] for i in part if (lyr_idx,i) in split_set ])
        new_merge_dir.append(( lyr_idx, new_partition ))
        
    return new_merge_dir


def get_cegar_culprit_neuron(conc_net, abs_net, counter_examples, visited_culprit, grads):
    # TODO Potentially redundant with identify culprit neurons, fix
    
    __, conc_vals = conc_net.eval(counter_examples)
    __, abs_vals  = conc_net.eval(counter_examples)
    
    culprit = None
    
    for i in range(1, len(conc_vals)-1):
        c_val, a_val, grad_lyr    = conc_vals[i], abs_vals[i], grads[i]
        
        val_diff_abs_conc_gradients = np.abs(np.multiply( 
            np.array(c_val)-np.array(a_val),grad_lyr))
        val_mean = np.mean(val_diff_abs_conc_gradients, axis=0)
        
        indices_not_visited         = np.where(
            np.array(visited_culprit[i-1])==0)[0].tolist()
       
        if len(indices_not_visited) == 0:
            continue 

        pos_max_elem = indices_not_visited[val_mean[indices_not_visited].argmax()]

        max_val_elem                = val_mean[pos_max_elem]

        if culprit is None and visited_culprit[i-1][pos_max_elem]==0:
            culprit = (i, pos_max_elem, max_val_elem)
        elif culprit is not None and (culprit[2] < max_val_elem) and visited_culprit[i-1][pos_max_elem]==0:
            culprit = (i, pos_max_elem, max_val_elem)
    
    if culprit is not None:
        visited_culprit[culprit[0]-1][culprit[1]] = 1
        return culprit[0:-1]
    else: 
        None




def cex_guided_refine( conc_net, abs_net, merge_dir, cex, 
        n_refine = config.NUM_REFINE ):
    """
    Uses a given cex to find a concrete neuron to split out of the abstract
    neuron and returns the resulting partitions

    Arguments:
    
    conc_net    -   The concrete network
    abs_net     -   The abstract network
    merge_dir   -   The merge directive to refine
    cex         -   The counterexample to use for splitting
    n_refine    -   The number of neurons to split out.

    Returns the merge_dir after splitting
    """

    if config.ASSERTS:
        assert conc_net.num_layers == abs_net.num_layers

    # Simulate abstract and concrete
    _, conc_vals = conc_net.eval( cex )
    a, abs_vals = abs_net.eval( cex )

    # Collect distance between abstract and concrete neurons
    neurs_list = []
    dists = []
    for lyr_idx, partition in merge_dir:
        for abs_idx, part in enumerate(partition):
            if len(part) > 1:
                neurs_list.extend([ (lyr_idx, ci) for ci in part ])
                dists.append( np.abs( 
                    conc_vals[ lyr_idx ][ part ] - abs_vals[ lyr_idx ][ abs_idx ]
                ))

    # Sort by distance
    split_idxs = np.argsort( np.concatenate( dists ))
    if split_idxs.shape[0] > n_refine:
        split_idxs = split_idxs[ -n_refine : ]

    # Collect indices into set
    split_set = set()
    for i in split_idxs:
        split_set.add( neurs_list[i] )

    # Create new partition
    new_merge_dir = []
    for lyr_idx, partition in merge_dir:
        new_partition = []
        for part in partition:
            rem_part = [ i for i in part if (lyr_idx,i) not in split_set ]
            if len(rem_part) > 0:
                new_partition.append( rem_part )
            new_partition.extend([ 
                [i] for i in part if (lyr_idx,i) in split_set ])
        new_merge_dir.append(( lyr_idx, new_partition ))
        
    return new_merge_dir



