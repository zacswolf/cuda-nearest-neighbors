# Python script to generate datasets

import numpy as np

def main(args):
	rng = np.random.default_rng(args.seed)

	data = np.zeros((args.num_points, args.dimension))

	group0 = args.num_points//2
	group1 = args.num_points-group0
	
	data[:group0] = rng.normal(loc=0., scale=args.sd, size=(group0, args.dimension))
	data[group0:] = rng.normal(loc=args.loc, scale=args.sd, size=(group1, args.dimension))

	if (args.label):
		labels = np.arange(0, data.shape[0]) >= group0
		data = np.column_stack((data, labels))

	rng.shuffle(data)
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
	parser.add_argument('--loc',
						type=float,
						help='location of distribution 1',
						default=1.)
	parser.add_argument('--sd',
						type=float,
						help='standard deviation of distributions',
						default=1.)

	args = parser.parse_args()

	main(args)
