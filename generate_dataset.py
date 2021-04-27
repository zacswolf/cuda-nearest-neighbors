# Python script to generate datasets

import numpy as np

def main(args):
    rng = np.random.default_rng(args.seed)

    data = rng.normal(size=(args.num_points, args.dimension))
    if (args.label):
        labels = data[:, 0] > 0
        data = np.column_stack((data, labels))

    np.savetxt(args.output, data, delimiter=', ', fmt='%f')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('-n', 
                        '--num_points', 
                        type=int, 
                        help='the number of data points to generate', 
                        required=True)
    parser.add_argument('-d', 
                        '--dimension', 
                        type=int, 
                        help='the dimension of the data points', 
                        required=True)
    parser.add_argument('-o', 
                        '--output', 
                        type=str, 
                        help='the path to the output file', 
                        required=True)
    parser.add_argument('-s', 
                        '--seed', 
                        type=int, 
                        help='the random number generator\'s seed', 
                        default=0)
    parser.add_argument('-l', 
                        '--label', 
                        action='store_true',
                        help='generate labels in the last column')

    args = parser.parse_args()

    main(args)
