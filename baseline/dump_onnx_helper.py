"""
Contains methods for dumping the network to an onnx file
"""
import onnx
import onnx.helper as oh
import onnx.checker
import numpy as np
import traceback

# import utils
# import config



def dump_weights_biases( G, dump_fname, batch_sizes=None ):
    """
    residual connections.
    
    Arguments:
    G           -   The graph
    dump_fname  -   The file to dump onnx to

    Returns: Nothing

    NOTE: Assumes that the last layer does not have relu
    """

    # Layers and layer sizes
    #layers = utils.getLayers( G )
    layer_sizes = G.layer_sizes
    print(layer_sizes)
    ## ONNX ##
    
    # Create the input tensor descriptor, store it in list
    if batch_sizes==None:
        onnx_inputs = [
            oh.make_tensor_value_info(
                "input", onnx.TensorProto.FLOAT, (layer_sizes[0], ) 
            )
        ]
    else:
        onnx_inputs = [
        oh.make_tensor_value_info(
            "input", onnx.TensorProto.FLOAT, (batch_sizes, layer_sizes[0]) 
        )
    ]
    print(onnx_inputs)

    # Keeps track of all nodes, initializers. Nodes must be topologically sorted.
    onnx_nodes = []
    onnx_initializers = []

    # Keep track of various links
    # lyr_idx -> link from relu. 'input' for input, None for output
    post_relu_lnks = { 0: "input", len(layer_sizes) - 1: None }

    # Link with output so far
    out_lnk = None
    weights = G.weights
    biases  = G.biases

    # Loop over each layer
    for i, lyr_size in enumerate( layer_sizes[1:] ):
        lyr_idx = i+1
        ## Add MatMuls for edges
        # Get stack of weights
        w_stack = weights[i]
        #print(w_stack)
        dst_lyr = lyr_idx

        # Extract first weight
        src_lyr, weight_vct = i, w_stack

        # Create an ONNX tensor for the weight matrix
        weight_init_name = "Weight_{}_{}".format( src_lyr, dst_lyr ) 
        onnx_initializers.append( 
            oh.make_tensor(
                name = weight_init_name,
                data_type = onnx.TensorProto.FLOAT,
                dims = weight_vct.shape,  
                vals = weight_vct.flatten().tolist(),  
            )
        )
            
        # Create matmul node
        out_lnk = "Link_Post_MatMul_{}_{}".format( src_lyr, dst_lyr )
        onnx_nodes.append( 
            oh.make_node( "MatMul", 
                inputs = [ 
                    post_relu_lnks[ src_lyr ],
                    weight_init_name 
                ],
                outputs = [ out_lnk ],
                name = "Node_MatMul_{}_{}".format( src_lyr, dst_lyr ) 
        ))
        ## Bias

        bias_vct = biases[ i ]

        # Create an ONNX tensor for the bias vector
        bias_init_name = "Bias_{}".format( lyr_idx ) 
        onnx_initializers.append( 
            oh.make_tensor(
                name = bias_init_name,
                data_type = onnx.TensorProto.FLOAT,
                dims = bias_vct.shape,  
                vals = bias_vct.tolist(),  
            )
        )

        # Make the bias node
        new_out_lnk = "Link_Post_Bias_{}".format( lyr_idx ) 
        onnx_nodes.append( 
            oh.make_node( "Add", 
                inputs = [ out_lnk, bias_init_name ],
                outputs = [ new_out_lnk ],
                name = "Node_Add_{}".format( lyr_idx ) 
        ))
        out_lnk = new_out_lnk 

        ## ReLU

        # If not last layer, add ReLU node
        if lyr_idx < len(layer_sizes) - 1:

            # Create node
            new_out_lnk = "Link_Post_Relu_{}".format( lyr_idx )
            onnx_nodes.append( 
                oh.make_node( "Relu", 
                    inputs = [ out_lnk ],
                    outputs = [ new_out_lnk ],
                    name = "Node_Relu_{}".format( lyr_idx ) 
            ))
            out_lnk = new_out_lnk 

            # Record links
            post_relu_lnks[ lyr_idx ] = out_lnk

    # Create the output tensor descriptor, store it in list
    onnx_outputs = [
        oh.make_tensor_value_info(
            out_lnk, onnx.TensorProto.FLOAT, (layer_sizes[-1], ) 
        )
    ]

    # Create model
    onnx_model = oh.make_model( oh.make_graph(
        nodes = onnx_nodes,
        name = "Dumped_Onnx",
        inputs = onnx_inputs,
        outputs = onnx_outputs,
        initializer = onnx_initializers,
    ))
    # if config.ASSERTS:
    #     try:
    #         onnx.checker.check_model( onnx_model, full_check = True )
    #     except Exception as e:
    #         utils.log( "Onnx model check failed" )
    #         utils.log( traceback.format_exc() )
    #         assert False

    # Serialize the ONNX model to a file
    onnx.save(onnx_model, dump_fname)
def dump_onnx( G, dump_fname, batch_sizes ):
    """
    Given a graph, converts it to an onnx model. All edges between two layers
    are packed into the same MatMul node. Edges that skip layers are treated as
    residual connections.
    
    Arguments:
    G           -   The graph
    dump_fname  -   The file to dump onnx to

    Returns: Nothing

    NOTE: Assumes that the last layer does not have relu
    """

    # Layers and layer sizes
    #layers = utils.getLayers( G )
    layer_sizes = G.layer_sizes
    print(layer_sizes)
    ## ONNX ##
    
    # Create the input tensor descriptor, store it in list
    if batch_sizes==None:
        onnx_inputs = [
            oh.make_tensor_value_info(
                "input", onnx.TensorProto.FLOAT, (layer_sizes[0], ) 
            )
        ]
    else:
        onnx_inputs = [
        oh.make_tensor_value_info(
            "input", onnx.TensorProto.FLOAT, (batch_sizes, layer_sizes[0]) 
        )
    ]
    print(onnx_inputs)

    # Keeps track of all nodes, initializers. Nodes must be topologically sorted.
    onnx_nodes = []
    onnx_initializers = []

    # Keep track of various links
    # lyr_idx -> link from relu. 'input' for input, None for output
    post_relu_lnks = { 0: "input", len(layer_sizes) - 1: None }

    # Link with output so far
    out_lnk = None
    weights = G.weights
    biases  = G.biases

    # Loop over each layer
    for i, lyr_size in enumerate( layer_sizes[1:] ):
        lyr_idx = i+1
        ## Add MatMuls for edges
        # Get stack of weights
        w_stack = weights[i]
        #print(w_stack)
        dst_lyr = lyr_idx

        # Extract first weight
        src_lyr, weight_vct = i, w_stack

        # Create an ONNX tensor for the weight matrix
        weight_init_name = "Weight_{}_{}".format( src_lyr, dst_lyr ) 
        onnx_initializers.append( 
            oh.make_tensor(
                name = weight_init_name,
                data_type = onnx.TensorProto.FLOAT,
                dims = weight_vct.shape,  
                vals = weight_vct.flatten().tolist(),  
            )
        )
            
        # Create matmul node
        out_lnk = "Link_Post_MatMul_{}_{}".format( src_lyr, dst_lyr )
        onnx_nodes.append( 
            oh.make_node( "MatMul", 
                inputs = [ 
                    post_relu_lnks[ src_lyr ],
                    weight_init_name 
                ],
                outputs = [ out_lnk ],
                name = "Node_MatMul_{}_{}".format( src_lyr, dst_lyr ) 
        ))
        ## Bias

        bias_vct = biases[ i ]

        # Create an ONNX tensor for the bias vector
        bias_init_name = "Bias_{}".format( lyr_idx ) 
        onnx_initializers.append( 
            oh.make_tensor(
                name = bias_init_name,
                data_type = onnx.TensorProto.FLOAT,
                dims = bias_vct.shape,  
                vals = bias_vct.tolist(),  
            )
        )

        # Make the bias node
        new_out_lnk = "Link_Post_Bias_{}".format( lyr_idx ) 
        onnx_nodes.append( 
            oh.make_node( "Add", 
                inputs = [ out_lnk, bias_init_name ],
                outputs = [ new_out_lnk ],
                name = "Node_Add_{}".format( lyr_idx ) 
        ))
        out_lnk = new_out_lnk 

        ## ReLU

        # If not last layer, add ReLU node
        if lyr_idx < len(layer_sizes) - 1:

            # Create node
            new_out_lnk = "Link_Post_Relu_{}".format( lyr_idx )
            onnx_nodes.append( 
                oh.make_node( "Relu", 
                    inputs = [ out_lnk ],
                    outputs = [ new_out_lnk ],
                    name = "Node_Relu_{}".format( lyr_idx ) 
            ))
            out_lnk = new_out_lnk 

            # Record links
            post_relu_lnks[ lyr_idx ] = out_lnk

    # Create the output tensor descriptor, store it in list
    onnx_outputs = [
        oh.make_tensor_value_info(
            out_lnk, onnx.TensorProto.FLOAT, (layer_sizes[-1], ) 
        )
    ]

    # Create model
    onnx_model = oh.make_model( oh.make_graph(
        nodes = onnx_nodes,
        name = "Dumped_Onnx",
        inputs = onnx_inputs,
        outputs = onnx_outputs,
        initializer = onnx_initializers,
    ))
    # if config.ASSERTS:
    #     try:
    #         onnx.checker.check_model( onnx_model, full_check = True )
    #     except Exception as e:
    #         utils.log( "Onnx model check failed" )
    #         utils.log( traceback.format_exc() )
    #         assert False

    # Serialize the ONNX model to a file
    onnx.save(onnx_model, dump_fname)



if __name__ == "__main__":
    import sys
    
    tst_no = int( sys.argv[1] )
    
    if tst_no == 0:
        """
        Load the network and dump onnx
        """
        import reader
        
        in_fname = "../networks/ACASXU_run2a_1_6_batch_2000.onnx"
        out_fname = "/tmp/net.onnx"
        net = reader.network_data(in_fname, False)
        
        dump_onnx( net, out_fname )
