import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


import subprocess
import re
import pandas as pd
import argparse
import time
import math




def get_times(outputStr):
	# print(outputStr)
	output = outputStr.split('\n')[-2].split(",")
	# print(output)
	_time = float(output[1])
	accuracy = float(output[0])
	# print(time)
	return _time, accuracy


def run_code(mode, gpu, d_prime):
	timing_runs = 4
	
	if gpu:
		gpu_str = "-g"
	else:
		gpu_str = ""
	
	execution_str = "./runner -d trainData.csv -p testData.csv -m %i -n %i %s" % (
		mode, d_prime, gpu_str
	)

	sum_time = 0.
	sum_accuracy = 0.

	for _ in range(timing_runs):
		# tic = time.perf_counter()
		outputStr = subprocess.run(execution_str.split(), stdout=subprocess.PIPE).stdout.decode("utf-8")
		# toc = time.perf_counter()
		
		# time_cur = float(toc-tic)
		# print(time_cur)
		# sum_time += time_cur
		time_cur, accuracy_cur = get_times(outputStr)
		sum_time += time_cur
		sum_accuracy += accuracy_cur


	return [float(sum_time) / timing_runs, float(sum_accuracy) / timing_runs]



def main(run, pickle_path):
	# Build
	# subprocess.run(['make'])

	# num train points, num test points, dim
	dataset_files = [
		(100, 100, 100),
		(1000, 1000, 100),
		# (1000, 1000, 1000),
		# (1000, 1000, 10000),
		]


	exact_modes = [0]
	approx_modes = [1, 2, 3]
	
	approximate_dims_percent = [.25, .5, .75]

	# Run
	if run:
		record = []
		for dataset_file in dataset_files:
			#gen test files
			subprocess.run(['python3', 'generate_dataset.py', '-n', str(dataset_file[0]), '-d', str(dataset_file[2]), '-o', 'trainData.csv', '-l', '--loc', '100', '--sd', '4'])
			subprocess.run(['python3', 'generate_dataset.py', '-n', str(dataset_file[1]), '-d', str(dataset_file[2]), '-o', 'testData.csv', '-l', '-s', '1', '--loc', '100', '--sd', '4'])
			for gpu in [False]:
				for m in exact_modes:
					stats = run_code(m, gpu, dataset_file[2])
					output = [str(dataset_file), m, gpu, dataset_file[2]]+list(stats)
					print(output)
					record.append(output)
				for m in approx_modes:
					for d_prime_per in approximate_dims_percent:
						d_prime = math.ceil(d_prime_per*dataset_file[2])
						stats = run_code(m, gpu, d_prime)
						output = [str(dataset_file), m, gpu, d_prime]+list(stats)
						print(output)
						record.append(output)

		df = pd.DataFrame(record, columns=[
						  'File', 'mode', 'gpu', "d_prime", "time", "accuracy"])
		print("writing to disk")
		df.to_pickle(pickle_path)

	# Analyze
	df = pd.read_pickle(pickle_path)

	df["ModeGpuDprime"] = list(zip(df["mode"], df["gpu"], df["d_prime"]))

	df["ModeAndGpu"] = list(zip(df["mode"], df["gpu"]))

	analyze(df, "mode", "time", label_all_xaxis=True)

	analyze(df, "gpu", "time", label_all_xaxis=True)

	analyze(df, "ModeAndGpu", "time", label_all_xaxis=True)

	analyze(df, "ModeGpuDprime", "time", label_all_xaxis=True)

	#analyze(df[df["Mode"]!="SEQUENTIAL"], "Mode", "TimePerIter", label_all_xaxis=True)


def analyze(df, x, y, label_all_xaxis=False):
	df_ = df.groupby(['File', x], as_index=False)[y].mean()
	df_ = df_.pivot(index=x, columns='File', values=y)

	num_files = df['File'].nunique()
	fig, axes = plt.subplots(nrows=num_files, ncols=1, sharex=True)

	if label_all_xaxis:
		for axi in axes.flat:
			axi.xaxis.set_major_locator(plt.MaxNLocator(len(df_)))

	for (key, ax, style) in zip(df_.columns, axes.flatten(), ['bs-', 'ro-', 'y^-']):
		print(key)
		print(ax)
		df_.plot.bar(y=key, ax=ax, style=style)


	plt.xticks(rotation=30*int(label_all_xaxis))
	plt.autoscale()

	fig.suptitle('%s Plots' % x, fontsize=16)
	fig.text(0.04, 0.5, '%s (ms)' % y, va='center', rotation='vertical')
	#plt.ylabel('%s (ns)')
	fig.savefig('%s.png' % x, bbox_inches = "tight")


if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument('-r', action='store_true',
						help="Run go testing script")
	parser.add_argument('-p', '--path', type=str, help="path to pickle file")

	args = parser.parse_args()
	main(args.r, args.path)
