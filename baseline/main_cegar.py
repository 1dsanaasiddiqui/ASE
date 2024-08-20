"""
The main file
"""


import time
from itertools import count
import argparse
import os.path
from datetime import datetime
import traceback
import copy


import numpy as np
from split_and_merge import split_net, merge_net
from cegar import saturation_partition, cex_guided_refine
from property_encode import encode_property
from network import load_nnet, load_nnet_from_tf ,Network, load_onnx
from utils import start_timer, record_time, log_times
from extract_cex import extract_cex_from_file 
from dump_custom_vnnlib import dump_vnnlib
import dump_onnx_helper
import config

import subprocess



def combine_inc_dec(merge_dir, conc_net):
    
    
    refined_merge_dir1 = []

    for merge in merge_dir:
        layer_no = merge[0]
        merge_dir_2 = copy.deepcopy(merge[1])
        w = np.array(conc_net.weights[layer_no-1]).T
        merge_dir1 = []
        i = 0
        while( i < len(merge_dir_2)):
            arr = merge_dir_2[i]
            flag = False
            j = i+1
            while( j < len(merge_dir_2) and len(merge_dir_2) != 0):
                arr1 = merge_dir_2[j]
                if (len(arr)==len(arr1)==1):
                    t = np.array_equal(w[arr[0]], w[arr1[0]])
                    if t == True:
                        flag = True
                        x = np.array([arr[0],arr1[0]])
                        merge_dir1.append(x)
                        merge_dir_2.pop(j)
                        break
                    else: 
                        j = j+1
                else: 
                    j = j+1
            if not flag:
                merge_dir1.append(arr)
            i = i + 1
        refined_merge_dir1.append((layer_no, merge_dir1))
    return refined_merge_dir1



def save_stats( stats, stats_fname, res, t0, size ):
    """
    A helper function to save the statistics dict to given file

    Arguments:
    
    stats       -   The existing stats dict, gets updated.
    stats_fname -   The file to save to
    res         -   The result so far
    t0          -   The time when timer was started
    size        -   The size of the network so far
    """
    if stats_fname is None:
        return

    stats['result'] = res
    stats['time'] = time.perf_counter() - t0 
    stats['net_size'] = size

    if config.DEBUG:
        print("Stats: ", stats )
    
    with open( stats_fname, 'w' ) as f:
        f.write( str( stats ))


def abs_refine_loop( budget, net, inp_bnds, out_ub,
        abstraction = 'none', 
        stats = {}, stats_fname = None):
    """
    Tries to verify the given property on the given network using an
    abstraction-refinement loop.

    Supports the following abstraction methods:
    
    'none'      -   No abstraction, the network is directly passed to Marabou
    'saturation'-   Merge to saturation.

    This function can be passed a stats_dict, and a stats_file. It will
    periodically update the stats dict with the below statistics, and save the
    resulting dict into the given stats file. Any entries already present in the
    dict will also be saved. The statistics tracked are:
    
    'result'    -   'Sat' if a cex was found, 'Unsat' if cex proven to not
                    exist, 'Unknown' if the process is still running, and
                    'Error' if an error has occurred
    'time'      -   The total time taken
    'net_size'  -   The size of the network
    
    Arguments:

    net         -   The network. Must have exactly one output.
    inp_bnds    -   The input bounds. This is a list of tuples, each tuple
                    giving the upper and lower bounds of the corresponding
                    input. If a bound is given as None, that bound is left
                    unbounded.
    out_ub      -   The upper bound on the output of the network 
    abstraction -   The abstraction method to use.
    refinement  -   The refinement method to use
    stats       -   A dict with existing entries to be saved to the stats file
    stats_fname -   Where to save the stats dict

    Returns:
    
    1.  Cex if it is found, else None.
    2.  The statistics dict.
    """
    conc_net = net
    
    # Start timer
    t0 = time.perf_counter()

    try:
        
        # If no abstraction, do nothing
        if abstraction == 'none':
            abs_net = net
            conc_net = net

        # Else, split the net, classify, do saturation merging.
        elif abstraction == 'saturation':
            conc_net, inc_dec_vects = split_net( net )
            merge_dir = saturation_partition( inc_dec_vects )
            abs_net = merge_net( conc_net, merge_dir, inc_dec_vects )
        
        save_stats( stats, stats_fname, 'Unknown', t0, sum( abs_net.layer_sizes ))

        # CEGAR loop

        refined_net = abs_net
        net_sizes_trail = []

        if abstraction != 'none':
            refined_merge_dir = merge_dir
        
        for num in count():
	    
            if sum(refined_net.layer_sizes[1:-1]) > budget:
                print(refined_net.layer_sizes[1:-1])
                break
            # refined_merge_dir = combine_inc_dec(refined_merge_dir, refined_net)
            chopped_refined_net = Network(refined_net.weights[:-1],refined_net.biases[:-1])
            chopped_refined_net.dump_npz("chopped_refined_net.npz")
            

            print("\n\n------- Refine Loop {} -------\n\n".format( num ))
            print("Network layer sizes: {}".format( chopped_refined_net.layer_sizes ))
            
            # Save stats
            net_sizes_trail.append( [ n for n in chopped_refined_net.layer_sizes ])
            stats['net_sizes'] = net_sizes_trail
            save_stats( 
                stats, stats_fname, 'Unknown', t0, sum( chopped_refined_net.layer_sizes
            ))

            if config.ASSERTS:
                rand_inps = np.random.rand( 100, refined_net.in_size )
                ref_out = refined_net.eval( rand_inps )[0][0]
                conc_out = conc_net.eval(rand_inps)[0][0]
                if config.DEBUG:
                    if np.any( np.isnan( ref_out )):
                        print("ref_out: ", ref_out)
                    if np.any( np.isnan( conc_out )):
                        print("ref_out: ", conc_out)
                    if not np.all( ref_out >= conc_out ):
                        where_bad = np.nonzero( ref_out < conc_out )[0]
                        print("Bad ref_out elems: ", ref_out[ where_bad ])
                        print("Bad conc_out elems: ", conc_out[ where_bad ])
                        print("Bad diff: ", (ref_out - conc_out)[ where_bad ])
                assert not np.any( np.isnan( ref_out ))
                assert not np.any( np.isnan( conc_out ))
                assert np.all( ref_out + config.FLOAT_TOL >= conc_out )



            # Do a query, exit if return is positive.
            os.system("rm res.txt")

            dump_vnnlib('custom_vnnlib.vnnlib',inp_bnds, 
                    chopped_refined_net.layer_sizes)
            
            dump_onnx_helper.dump_onnx(chopped_refined_net, "chopped_dump_net.onnx", 1)

            command = "python neuralsat/neuralsat-pt201/main.py --net {} --spec {} --result_file {} --export_cex ".format( 'chopped_dump_net.onnx', 'custom_vnnlib.vnnlib', 'res.txt')

            time_before_call = time.perf_counter()

            completed_process = subprocess.run(command, shell=True)

            time_after_call = time.perf_counter()

            #times_for_solver_calls.append(time_after_call-time_before_call)

            if completed_process.returncode == 0:
                # Command completed successfully, proceed with extracting cex
                cex = extract_cex_from_file('res.txt')
                print("Extracted cex: ", type(cex), cex)
                os.system("rm custom_vnnlib.vnnlib* ")
                #os.system("rm chopped_refined_net.npz")
            else:
                print("The command did not complete successfully.")
                raise RuntimeError()
            
            #os.system("rm chopped_dump_net.onnx")
            os.system("rm res.txt")


            if config.ASSERTS:
                if cex is not None and len(cex)>0:     # Sanity check
                    assert (
                        refined_net.eval( cex )[0][0] + config.FLOAT_TOL >= 
                        conc_net.eval( cex)[0][0]
                    )

                    
            if cex is None or len(cex)==0:
                
                print("No cex, property proved")
                break
            
            elif conc_net.eval( cex )[0][0] >= out_ub:
                
                print("Found cex, property disproved")
                break               

            else:
                # This should never happen if abstraction method is 'none'
                assert not abstraction == 'none'
                if config.DEBUG:
                    if not refined_net.eval( cex )[0][0] >= out_ub:
                        print("Mismatch")
                        print(cex)
                        print(conc_net.eval( cex )[0])
                        print(refined_net.eval( cex )[0])
                        print(out_ub)
                        assert False

            
            # Refine network until cex is no longer spurious
            while refined_net.eval( cex )[0][0] >= out_ub:
                
                # Save stats
                save_stats( 
                    stats, stats_fname, 'Unknown', t0, sum( refined_net.layer_sizes ))

                print("Refining network with cegar")
                refined_merge_dir = cex_guided_refine( 
                    conc_net, 
                    refined_net,
                    refined_merge_dir,
                    cex
                )
                
                refined_net = merge_net( conc_net, refined_merge_dir, inc_dec_vects )
                if sum(refined_net.layer_sizes[1:-1]) > budget:
                    print(refined_net.layer_sizes[1:-1])
                    chopped_refined_net = Network(refined_net.weights[:-1],refined_net.biases[:-1])
                    chopped_refined_net.dump_npz("chopped_refined_net.npz")
                    break
                print("Refined network layer sizes: {}".format( refined_net.layer_sizes ))
                      

        # Save stats
        save_stats( 
            stats, 
            stats_fname, 
            'Safe' if cex is None or len(cex)==0 else 'Unsafe', 
            t0, 
            sum( refined_net.layer_sizes )
        )

        # Return cex
        return cex, stats

    except Exception as e:
        print("Exception: ", e)
        traceback.print_exc()
        save_stats( stats, stats_fname, 'Error', t0, -1 )
        raise e
    
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--network',
        dest='net_fname',
        required=True,
        type=str,
        help="The network file to check"
    )
    parser.add_argument('-p', '--property',
        dest='prop_fname',
        required=True,
        type=str,
        help="The file with the property to check"
    )
    parser.add_argument('-size', '--size',
        dest='net_size',
        required=True,
        type=str,
        help="The file with the property to check"
    )
    parser.add_argument('-a', '--abstraction', 
        dest='abs_method',
        default='none',
        choices=['saturation', 'none'],
        type=str,
        help="The abstraction method to use, none for direct Marabou"
    )
    parser.add_argument('--stats-file',
        dest='stats_fname',
        default=None,
        type=str,
        help="The file to which to store statistics"
    )
    args = parser.parse_args()

    # Load network
    net_fname = args.net_fname
    if net_fname.endswith( 'nnet' ):
        net = load_nnet( net_fname )
    elif net_fname.endswith('.tf'):
        net = load_nnet_from_tf(net_fname)
    elif net_fname.endswith('.onnx'):
        net = load_onnx(net_fname)
    else:
        raise RuntimeError("Unknown network filetype")

    budget = int(args.net_size)

    # Load property
    prop_fname = args.prop_fname
    # if prop_fname.endswith( 'prop' ):
    with open( prop_fname, 'r' ) as f:
        prop = eval(f.read())
    # else:
    #     raise RuntimeError("Unknown property filetype")
        
    # Encode property
    encoded_net, inp_bnds, out_ub, property = encode_property( net, prop )    

    stats = {}
    
    for arg_name, arg_val in vars( args ).items():
        stats[ arg_name ] = arg_val
    
    if args.stats_fname is None:
        stats_fname = os.path.join( 
            config.STATS_PATH, 
            str( datetime.today() ))
    else:
        stats_fname = args.stats_fname

    # Call loop
    cex, stats = abs_refine_loop( budget, encoded_net, inp_bnds, out_ub,
        abstraction = args.abs_method, 
        stats = stats,
        stats_fname = stats_fname,
    )

    print("Output: {}".format( stats ))
    if cex is not None and len(cex)>0:
        print("Counterexample found: ", cex)
        print("Network output on cex: ", net.eval( np.array( cex ))[0] )
