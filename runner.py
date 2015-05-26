#! /usr/bin/env python2.7
import math
import subprocess
import os, glob, shutil
import errno
import sys
from argparse import ArgumentParser

def check_dir(dir, force):
	old_files = [file for file in os.listdir(dir) if '.timing' in file and os.path.isfile(os.path.join(dir, file))]
	if len(old_files) and not force:
		raise Exception("Old data found in /{}/... stopping".format(dir))

def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

def run(name, diff=False, shuffle=False, force=False):
	maker = ['make', '-j24', 'DEBUG=FALSE', 'IGN=TRUE', 'PRINT=FALSE',  'LOG_OUTPUT=FALSE']
	if shuffle:
		maker.extend(["SAME_IC=FALSE", "SHUFFLE=TRUE"])
	elif diff:
		maker.extend(["SAME_IC=FALSE"])

	with open('src/launch_bounds.cuh') as file:
		lines = [line.strip() for line in file.readlines()]
		block_size = next(line for line in lines if 'TARGET_BLOCK_SIZE' in line)
		block_size = block_size[block_size.index('(') + 1 : block_size.index(')')]

	all_exes = []
	for file in glob.glob('*-int*'):
		all_exes.append(file)

	all_exes = sorted(all_exes, key = lambda x: 1000 if 'gpu' in x else 1)

	threads = [6, 12]
	powers = [0, 3, 7, 10, 14, 17, 20]
	repeats = 5
	l_t = [False, True]
	l_s = [False, True]
	exppowers = [int(pow(2, exponent)) for exponent in powers]

	#make sure we don't have old data
	check_dir('output', force)
	if not shuffle and not diff:
		desc = '_same_ic'
	elif shuffle:
		desc = '_shuffled_ic'
	elif diff:
		desc = '_diff_ic'
	else:
		raise Exception('Cannot turn on Shuffle and Diff at same time')

	make_sure_path_exists(name + desc)
	results_folder = os.path.abspath(name + desc)
	check_dir(results_folder, force)

	file_list = []
	for repeat in range(repeats):
		for low_tol in l_t:
			for large_step in l_s:
				flags = maker[:]
				if low_tol:
					flags += ['LOW_TOL=TRUE', 'FAST_MATH=TRUE']
				else:
					flags += ['FAST_MATH=FALSE']
				if large_step:
					flags += ['LARGE_STEP=TRUE']
				#force remake
				subprocess.call(flags)
				for exe in all_exes:
					the_threads = threads if not 'gpu' in exe else [block_size]
					for thread in the_threads:
						for i in range(len(powers)):
							if powers[i] > 14 and not diff and not shuffle:
								continue
							if 'exp' in exe: 
								continue
							filename = './output/' + exe + '_' + str(thread) +'_' + str(powers[i]) + ('_lt' if low_tol else '') \
										+ ('_ls' if large_step else '') +'.timing'
							if filename not in file_list:
								file_list.append(filename)
							mode = 'a' if repeat != 0 else 'w'
							file = open(filename, mode)
							if 'gpu' in filename:
								subprocess.call([os.path.join(os.getcwd(), exe), str(exppowers[i])], stdout=file)
							else:
								subprocess.call([os.path.join(os.getcwd(), exe), str(thread), str(exppowers[i])], stdout=file)
							file.flush()
							file.close()
	#finally move the files so we can run other cases
	for file in file_list:
		shutil.move(os.path.join('output', file), os.path.join(results_folder, file))

if __name__ == '__main__':
	parser = ArgumentParser(description='Runs timing runs for the various integrators')
	parser.add_argument('-n', '--name',
						type=str,
						required=True,
						help='the name of the mechanism')
	parser.add_argument('-f', '--force',
						required=False,
						default=False,
						action='store_true',
						help='Force reuse of past data files')
	args = parser.parse_args()

	run(args.name, diff=True, shuffle=False, force=args.force)
	args.force = False
	run(args.name, diff=False, shuffle=True, force=args.force)