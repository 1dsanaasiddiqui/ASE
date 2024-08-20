"""
Generates .prop files from given csv file
"""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--csv-file', type = str, required = True,
        help = "Csv file to load from", )
parser.add_argument('--out-pfx', type = str, required = True,
        help = "Prefix of output files", )
parser.add_argument('--epsilons', type = str, 
        default = "0.004,0.008,0.016,0.031",
        help = "Epsilons to generate properites for, comma seperated", )
parser.add_argument('--scale', type = float, default = 1./256.,
        help = "Epsilons to generate properites for", )
args = parser.parse_args()

# Read csv data
with open( args.csv_file, 'rt' ) as f:
    csv_str = f.read()
csv_lines = [ s.split(',') for s in csv_str.splitlines() ]
tgts = [ int(line[0]) for line in csv_lines ]
inps = [ list( map( float, line[1:] )) for line in csv_lines ]

# Extract list of epsilons
eps_list = list( map( float, args.epsilons.split(',') ))
        
# Iterate over epsilons
for eps in eps_list:
    # Iterate over each instance
    for inst_no, (tgt, inp) in enumerate( zip( tgts, inps )):

        # Scale
        inp = [ x * args.scale for x in inp ]

        # Input conditions
        inp_conds = [ 
                (i, {'Lower': max(0., x - eps), 'Upper': min(1., x + eps) })
                for i, x in enumerate( inp ) ]

        # Iterate over all possible misclassifications
        for max_class in range(10):
            if max_class == tgt: continue
        
            # Output conditions
            out_conds = [ 
                ( [ (1.0, max_class), (-1.0, oth_class) ], {'Lower': 0.} )
                for oth_class in range(10) if not oth_class == max_class ]

            # Make and save
            prop_struct = {'input': inp_conds, 'output': out_conds}
            prop_fname = '{}_cls_{}_inst_{}_eps_{}'.format(
                    args.out_pfx, max_class, inst_no, eps, )
            with open( prop_fname, 'wt' ) as f:
                print( "Writing ", prop_fname )
                f.write( str( prop_struct ))
                
