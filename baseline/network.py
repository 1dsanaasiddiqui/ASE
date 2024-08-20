"""
Contains a network class and various related utility methods
"""



import numpy as np

import config

import sys
from maraboupy import MarabouNetworkNNet




class Network:
    """
    A class representing a DNN as a sequence or affine and relu transforms.

    Notation: Given any network, layer indices are always all-inclusive. That
    is, layer 0 is the input layer, and the last layer is the output layer.
    
    Members are:
    
    weights     -   List of weight matrices. Result of applying weight w on x is
                    given as x@w.
    biases      -   List of bias vectors
    end_relu    -   Does the last layer have ReLU or no
    num_layers  -   Holds the total number of layers in the network, including
                    input and output
    layer_sizes -   A list with sizes of each layer, including input and output
    out_size    -   The size of the output layer
    in_size     -   The size of the input layer
    """
    def __init__(self, weights, biases, end_relu = False):

        # Check for consistency in number of layers
        if config.ASSERTS:
            assert len(weights) == len(biases)
        self.num_layers = len(weights) + 1

        self.in_size = weights[0].shape[0]
        self.out_size = weights[-1].shape[1]

        # Check dimensions of weights and biases
        if config.ASSERTS:
            assert weights[0].shape[1] == biases[0].shape[0]
            for i in range( 1, self.num_layers-1 ):
                assert weights[i-1].shape[1] == weights[i].shape[0]
                assert weights[i].shape[1] == biases[i].shape[0]

        # Set up layer sizes
        self.layer_sizes = [ self.in_size ]
        for i in range( 1, self.num_layers-1 ):
            self.layer_sizes.append( weights[i].shape[0] )
        self.layer_sizes.append( self.out_size )
            
        # Set up weights and biases
        self.weights = weights
        self.biases = biases

        self.end_relu = end_relu

    def eval( self, inp ):
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
        vals = [inp]
        cval = inp

        # Evaluate inner layers
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            cval = np.where(cval @ w + b > 0, cval @ w + b, 0)
            vals.append(cval)

        # Evaluate last layer
        cval = cval @ self.weights[-1] + self.biases[-1]
        if self.end_relu:
            cval = np.where(cval > 0, cval, 0)
        vals.append(cval)

        return cval, vals


    def append_layer( self, weight, bias, relu = False,
                        fold_perms = True ):
        """
        Adds a new layer to the end of network. Collapses consecutive linear
        layers into a single layer.
        
        Arguments:
        
        weight      -   The weight matrix of the new layer
        bias        -   The bias vector for the new layer.
        relu        -   Whether the newly added layer contains a ReLU layer.
        fold_perms  -   If true, and if the new layer performs a positively
                        scaled permutation, it is folded into the last layer.
        """
        
        # Check sizes
        if config.ASSERTS:
            assert self.out_size == weight.shape[0]
            assert bias.shape[0] == weight.shape[1]

        # If there already was relu, create new layer
        if self.end_relu:
            self.weights.append( weight )
            self.biases.append( bias )
            self.num_layers += 1
            self.layer_sizes.append( bias.shape[0] )

            # Fold permutations
            # if fold_perms and 

        # Otherwise, collapse into last layer
        else:
            self.weights[-1] = self.weights[-1] @ weight
            self.biases[-1] = self.biases[-1] @ weight + bias
            self.layer_sizes[-1] = bias.shape[0]
            
        # Set new end_relu
        self.end_relu = relu
        
        # Set new output size
        self.out_size = bias.shape[0]

    def __str__( self ):
        """
        Simply return weights and biases, etc.
        """
        return '\n'.join(
            ["Network has {} layers with sizes {}".format( 
                self.num_layers, self.layer_sizes )] +
            ["Connected to output of layer {}: \nWeight: \n{} \nBias: \n{}".format(
                i, w, b) for i, (w,b) in enumerate( 
                    zip( self.weights, self.biases )) ] +
            ["The network {} with ReLU".format(
                "ends" if self.end_relu else "does not end" )]
        )

    def dump_npz( self, fname, extra_entries = {} ):
        """
        Saves network to given filename/path. Saves it as an .npz file with the
        following named entries:

        layer_sizes -   An array of layer sizes including input and output layer
        weight_i    -   Weight connected to output of layer i
        bias_i      -   Bias connected to output of layer i
        end_relu    -   A boolean value, if true, network ends with ReLU

        Entries in `extra_entries` are also added if given. Should be numbers or
        np arrays only.

        NOTE: a .npz extension is auto-added to filename if its not there
        """
        save_dict = {}
        for i, (w,b) in enumerate( zip( self.weights, self.biases )):
            save_dict['weight_{}'.format( i )] = np.float32(w)
            save_dict['bias_{}'.format( i )] = np.float32(b)
        save_dict['layer_sizes'] = np.array( self.layer_sizes ) 
        save_dict['end_relu'] = np.array([ self.end_relu ]) 
        
        for entry_name, entry_val in extra_entries.items():
            save_dict[ entry_name ] = entry_val

        if config.DEBUG:
            print("Layer sizes of network being dumped: ",
                save_dict['layer_sizes']
            )
            print("Total size: ", sum( save_dict['layer_sizes'] ))
            print("Dumping to: ", fname)

        np.savez( fname, **save_dict )


def load_npz( npz_data ):
    """
    Loads a Network from given npz data
    """
    data_dict = np.load(npz_data)
    layer_sizes = data_dict[ 'layer_sizes' ]
    end_relu = data_dict[ 'end_relu' ] 
    wb_num = len(layer_sizes) - 1
    weights = [ data_dict[ 'weight_{}'.format( wb_idx ) ] 
            for wb_idx in range( wb_num ) ]
    biases = [ data_dict[ 'bias_{}'.format( wb_idx ) ] 
            for wb_idx in range( wb_num ) ]
    return Network( weights, biases, end_relu )


def load_nnet( nnet_fname ):
    """
    Loads a network from an nnet file. Assumes that the network does not end
    with a ReLU.
    """

    # read nnetfile into Marabou Network
    marabou_net = MarabouNetworkNNet.MarabouNetworkNNet( filename=nnet_fname )

    net = Network(
        weights = [ np.array( w ).T for w in marabou_net.weights ],
        biases =  [ np.array( b )   for b in marabou_net.biases ],
        end_relu = False
    )

    return net

def load_nnet_from_tf(nnet_fname):
    with open(nnet_fname, 'r') as file:
        lines = file.readlines()

    weights = []
    biases = []

    for i in range(0, len(lines), 3):
    
        weight_str = lines[i + 1].strip()
        bias_str = lines[i + 2].strip()

        # Parse weight matrix
        weight = np.array(eval(weight_str)).T

        # Parse bias vector
        bias = np.array(eval(bias_str))

        weights.append(weight)
        biases.append(bias)

    net = Network(weights, biases, end_relu=False)
    return net

    # return activations, weights, biases

def load_onnx( onnx_filename ):
    """
    Given the name of the onnx example file, load it and convert it into a
    Network. Returns this Network.

    Arguments:
        onnx_filename   -   Name of the onnx file to load
    
    Returns: the network

    NOTE: This can only process .onnx files where:

    1. There are only Flatten, Constant, Add, Sub, Mul, Div and Genn operators.
    
    2. The input gets fed into exactly one Add/Sub, with a vector, followed by
       one Mul/Div with a scalar for normalization, then possibly Flattened, and
       then fed into the first linear layer.

    3. The linear layers driven by a Gemm operation.

    4. There should be exactly one input, and exactly one output.

    5. Each op has exactly one output (this holds for the ops in 1)

    6. The transA attribute of Genn is not supported.

    7. Attributes can only have float (typecode 1) or integer (typcode 2)
       constants

    NOTE: This code was originally written by Diganta M to extend the original
    implementation of CEGAR_NN to accept onnx files.
    """
    import onnx
    import onnx.numpy_helper

    # Collect weights and biases into here.
    weights = []
    biases = [] 
    
    # Load onnx
    model = onnx.load( onnx_filename )
    onnx.checker.check_model( model )
    graph = model.graph

    # Dict mapping link to node producing it as output
    link2outputter = { n.output[0] : n for n in graph.node}

    # Dict mapping link name to initializer
    link2initializer = { i.name : i for i in graph.initializer }

    # Map each link to the nodes taking that as input.
    # Check that there is upto one node for each input, ie, graph is linear
    link2inputter = {}
    for node in graph.node:
        for lnk in node.input:
            if lnk in link2inputter:    # One link is going to >1 inputters
                # l.error("Potential branch in onnx file")
                # l.error("link {} inputting into nodes {} and {}".format( lnk, node, link2inputter[lnk] ))
                raise RuntimeError("Potential branch in onnx file")

            link2inputter[ lnk ] = node 

    # Dict mapping initializer name to initializer
    name2init = { i.name : i for i in graph.initializer }

    # Helper function to get tensor values from initializer, node, or otherwise
    def _get_data( link ):
        if link in link2initializer:
            return onnx.numpy_helper.to_array( link2initializer[ link ] )
        elif link in link2outputter: 
            return onnx.numpy_helper.to_array( 
                link2outputter[ link ].attribute[0].t
            ) 
        else:
            # l.error("Link not found in outputters or initializers: {}".format(
            #     link ))
            # l.error("Outputters: {}, \n Initializers: {}".format(
            #     list(link2outputter.keys()), list(link2initializer.keys())
            # ))
            raise RuntimeError("Link not found in outputters or initializers")

    # Helper function to get the node's data(secondary) link given
    # input(primary) link
    def _get_secondary_input( node, primary_in):
        snd_inp_list = [
            i for i in node.input if i != primary_in 
        ]
        if len(snd_inp_list) != 1:
            # l.error("Not exactly one secondary input: {}".format( snd_inp_list))
            # l.error("Primary: {}".format( primary_in ))
            # l.error("Inputs: {}".format(  node.input ))
            raise RuntimeError("Not exactly one secondary input")
        if config.DEBUG:
            pass
            # l.debug("snd_inp_list: {}".format( snd_inp_list ))
        snd_lnk = snd_inp_list[0]

        return snd_lnk

    # Get the input names. Ignore initializers. Should be exactly one input
    init_names = [ i.name for i in graph.initializer ]
    inp_names = [ n.name for n in graph.input if n.name not in init_names ]
    if config.DEBUG:
        pass
        # l.debug( "Initializer names: {}".format( init_names )) 
        # l.debug( "Raw Input names: {}".format( [ n.name for n in graph.input ])) 
        # l.debug( "Input names: {}".format( inp_names )) 
    if len(inp_names) != 1:
        raise RuntimeError("Not exactly 1 input in onnx")
        # l.error("Onnx does not have exactly 1 inputs. They are {}".format(
        #     inp_names ))
        
    inp_name = inp_names[0]

    # Get the output name. Should be exactly one output.
    if len(graph.output) != 1:
        # l.error("Onnx does not have exactly 1 outputs. They are {}".format(
        #     [ n.name for n in graph.output ] ))
        raise RuntimeError("Not exactly 1 output in onnx")
    out_name = graph.output[0].name

    # Get the node input feeds into
    inp_node = link2inputter[ inp_name ]

    # There is some normalization happening
    if inp_node.op_type != 'Gemm':
        # l.debug("Potential normalization present")
        inp_add_node = inp_node

        # Get the bias added/subtracted from the input
        inp_add_lnk = next( i for i in inp_add_node.input if i != inp_name )
        inp_add = _get_data( inp_add_lnk )
        if inp_add_node.op_type == 'Sub':
            inp_add = - inp_add 
        elif inp_add_node.op_type != 'Add':
            #l.error("Bad node taking input: {}".format( inp_add_node))
            raise RuntimeError("Bad node taking input:")

        # Get the node the add feeds into
        inp_mul_node = link2inputter[ inp_add_node.output[0] ]

        # If 'Div', get the factor to div as 1/data, first node is next
        if inp_mul_node.op_type == 'Div':
            inp_mul_lnk = _get_secondary_input( 
                    inp_mul_node, inp_add_node.output[0] )
            inp_mul = _get_data( inp_mul_lnk ) 
            inp_mul = 1 / inp_mul 

            # Check that the second first (numerator) arg comes from input
            if not inp_add_node.output[0] == inp_mul_node.input[0]:
                #l.error("Input is in denominator of Div node")
                #l.error("Input via: {}, output: ".format( 
                    # inp_add_node, inp_add_node.output ))
                #l.error("Div node: {}, input: ".format( 
                    # inp_add_node, inp_add_node.input ))
                raise RuntimeError("Input is in denominator of Div node")

            # First node is next node, value produced by connector
            first_node = link2inputter[ inp_mul_node.output[0] ]
            norm_inp_name = inp_mul_node.output[0]

        # If 'Mul', just extract factor
        elif inp_mul_node.op_type == 'Mul':
            inp_mul_lnk = _get_secondary_input( 
                    inp_mul_node, inp_add_node.output[0] )
            inp_mul = _get_data( inp_mul_lnk ) 

            # First node is next node, value produced by connector
            first_node = link2inputter[ inp_mul_node.output[0] ]
            norm_inp_name = inp_mul_node.output[0]

        # If flatten, set factor to 1
        elif inp_mul_node.op_type == 'Flatten':
            inp_mul = np.array([1.]) 

            # Flatten node should have exactly 1 input
            if len(inp_mul_node.input) != 1:
                #l.error("Flatten node doesn't have exactly 1 input: {}".format(
                    # inp_mul_node.input ))
                raise RuntimeError("Flatten node doesn't have exactly 1 input")

            # First node is this node itself, value produced by input
            first_node = inp_mul_node
            norm_inp_name = first_node.input[0]

        else:
            #l.error("Bad node at multiply stage: {}".format( inp_mul_node))
            raise RuntimeError("Bad node at multiply stage")

        # If "first" node is a flatten node, flatten stuff, move to next node
        if first_node.op_type == 'Flatten':
            inp_add = inp_add.flatten()
            inp_mul = inp_mul.flatten()
            norm_inp_name = first_node.output[0]
            first_node = link2inputter[ first_node.output[0] ]

    # Else, input node is the first node
    else:
        first_node = inp_node
        norm_inp_name = inp_name
    
    # Node should now be Gemm, first linear layer
    if first_node.op_type != 'Gemm':
        #l.error("First non-normalization node is not Gemm: {}".format(
            # first_node.op_type ))
        raise RuntimeError("First non-normalization node is not Gemm")

    # Get which input of the Gemm is weight
    fwidx = 1 if first_node.input[0] == norm_inp_name else 0

    # Get first weight and bias from Gemm, folding in all attributes
    attrs = { 
        a.name : (a.f if a.type == 1 else a.i) for a in first_node.attribute 
    }
    first_weight = np.copy( 
        onnx.numpy_helper.to_array( name2init[ first_node.input[fwidx] ]) )
    first_weight *= attrs.get( 'alpha', 1 )
    if attrs.get( 'transB', 0 ) == 1:
        first_weight = np.transpose( first_weight )
    first_bias = np.copy( 
        onnx.numpy_helper.to_array( name2init[ first_node.input[2] ]))
    first_bias *= attrs.get( 'beta', 1 )

    # Transpose again if Genn multiplication convention puts vector to right
    if fwidx == 0:
        first_weight = np.transpose(first_weight)

    # Fold in inp_add and inp_mul if needed
    if inp_node.op_type != 'Gemm':
        first_weight = inp_mul * first_weight
        if inp_add.size == 1:
            inp_add = np.repeat( inp_add, first_weight.shape[0])
        first_bias += np.dot( inp_add, first_weight )

    # Size of input layer
    num_inputs = first_weight.shape[0]
    
    # Loop over layers
    weight = first_weight
    bias = first_bias
    layer_idx = 0
    node = first_node
    while True:
        
        # Get size of next layer
        current_hidden_lyr_size = weight.shape[1]
        
        # Add and update layer
        layer_idx += 1
        weights.append( weight )
        biases.append( bias )

        # Try to get next node
        if node.output[0] not in link2inputter:
            break
        link_name = node.output[0]
        node = link2inputter[ node.output[0] ]

        # Next node should be Relu
        if node.op_type != 'Relu':
            #l.error("Expected Relu node after linear layer, got: {}".format(
                # node.op_type ))
            raise RuntimeError("Expected Relu node after linear layer")

        # Try to get next node
        if node.output[0] not in link2inputter:
            break
        link_name = node.output[0]
        node = link2inputter[ node.output[0] ]

        # Next node should be Gemm
        if node.op_type != 'Gemm':
            #l.error("Expected Gemm node after relu layer, got: {}".format(
                # node.op_type ))
            raise RuntimeError("Expected Gemm node after relu layer")

        # Get which input of the Gemm is weight
        widx = 1 if node.input[0] == link_name else 0

        # Get weight and bias from Gemm, folding in all attributes
        attrs = { 
            a.name : (a.f if a.type == 1 else a.i) for a in node.attribute 
        }
        weight = np.copy( 
                onnx.numpy_helper.to_array( name2init[ node.input[widx] ]) )
        weight *= attrs.get( 'alpha', 1 )
        if attrs.get( 'transB', 0 ) == 1:
            weight = np.transpose( weight )
        bias = np.copy( onnx.numpy_helper.to_array( name2init[ node.input[2] ]))
        bias *= attrs.get( 'beta', 1 )

        # Transpose again if Genn multiplication convention puts vector to right
        if widx == 0:
            weight = np.transpose( weight )
    
    # If last node was a Relu, set a flag
    end_relu = node.op_type == 'Relu'
       
    # Set up and return network
    return Network( weights, biases, end_relu )
        

if __name__ == "__main__":
    
    fname = "networks/ACASXU_run2a_1_1_batch_2000.nnet"
    
    net = load_nnet( fname )
    
    print( net.layer_sizes )
    print( net.eval( np.array([ 1., 1., 1., 1., 1. ]) )) 
