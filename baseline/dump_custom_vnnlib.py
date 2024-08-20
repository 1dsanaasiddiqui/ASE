# Specify the value of n
# You can change this to your desired value

# Open a file for writing
def dump_vnnlib(file_name,inp_bnds,layer_sizes):
    with open(file_name, 'w') as file:
        # Loop through and generate the declarations for X_0 to X_n
        num_inps = layer_sizes[0]
        for i in range(num_inps):
            declaration = f'(declare-const X_{i} Real)\n'
            file.write(declaration)
        num_out = layer_sizes[-1]
        for i in range(num_out):
            declaration = f'(declare-const Y_{i} Real)\n'
            file.write(declaration)

        for i, (lower, upper) in enumerate(inp_bnds):
            file.write(f'(assert (<= X_{i} {upper}))\n')
            file.write(f'(assert (>= X_{i} {lower}))\n')

        for i in range(num_out):
            file.write(f'(assert (<= Y_{i} {0}))\n')

def dump_vnnlib_for_dataset(file_name, inc_class, layer_sizes):
    
    with open(file_name, 'w') as file:
        num_inps = layer_sizes[0]
        for i in range(num_inps):
            declaration = f'(declare-const X_{i} Real)\n'
            file.write(declaration)
        num_out = layer_sizes[-1]

        for i in range(num_out):
            declaration = f'(declare-const Y_{i} Real)\n'
            file.write(declaration)

        for i in range(num_inps):   # TODO Generalize beyond mnist
            file.write(f'(assert (<= X_{i} {1}))\n')
            file.write(f'(assert (>= X_{i} {0}))\n')

        for i in range(num_out):    # TODO Generalize beyond mnist
            if i!= inc_class:
                file.write(f'(assert (>= Y_{inc_class} Y_{i}))\n')

def dump_csv(model,custom_vnnlib):
    with open(file_name, 'w') as file:
        file.write(model)
        file.write(",")
        file.write(custom_vnnlib)
        file.write(",")
        file.write("116")


if __name__ == "__main__":
    layer_sizes = [ 5 ,50, 50, 50, 50, 50, 50,  1]
    inp_bnds = [[0.6, 0.6798577687], [-0.5, 0.5], [-0.5, 0.5], [0.45, 0.5], [-0.5, -0.45]]
    dump_vnnlib('custom_vnnlib.vnnlib',inp_bnds,layer_sizes)
    

