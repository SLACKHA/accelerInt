#! /usr/bin/env python2.7
import math
import subprocess
import os, glob
import sys
from argparse import ArgumentParser

parser = ArgumentParser(description='Runs all integrators for the given mechanism / options')
parser.add_argument('-diff', '--different_ics',
					default=False,
					action = 'store_true',
					required=False,
					help = 'Use different initial conditions')
parser.add_argument('-sh', '--shuffle',
					default=False,
					action='store_true',
					required=False,
					help = 'Shuffle the ics, implies --different-ics')
args = parser.parse_args()
maker = ['make', '-j24', 'DEBUG=FALSE', 'IGN=TRUE', 'PRINT=FALSE',  'LOG_OUTPUT=FALSE']
if args.shuffle:
	maker.extend(["SAME_IC=FALSE", "SHUFFLE=TRUE"])
elif args.different_ics:
	maker.extend(["SAME_IC=FALSE"])

all_exes = []
for file in glob.glob('*-int*'):
	all_exes.append(file)

all_exes = sorted(all_exes, key = lambda x: 1000 if 'gpu' in x else 1)

threads = [12]
powers = [0, 3, 7, 10, 14, 17, 20]
l_t = [False, True]
l_s = [False, True]
exppowers = [int(pow(2, exponent)) for exponent in powers]

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
			for thread in threads:
				for i in range(len(powers)):
					if powers[i] > 14 and not args.different_ics and not args.shuffle:
						continue
					if ('exp' in exe and 'gpu' in exe) and not low_tol:
						continue
					elif ('exp' in exe and 'gpu' in exe) and powers[i] > 10:
						continue
					elif ('exp' in exe and not 'gpu' in exe) and powers[i] > 14:
						continue
					filename = './output/' + exe + '_' + str(thread) +'_'+str(powers[i]) + ('_lt' if low_tol else '') \
								+ ('_ls' if large_step else '') +'.txt'
					if (os.path.isfile(filename)):
						continue
					file = open(filename, "w")
					if 'gpu' in filename:
						subprocess.call([os.path.join(os.getcwd(), exe), str(exppowers[i])], stdout=file)
					else:
						subprocess.call([os.path.join(os.getcwd(), exe), str(thread), str(exppowers[i])], stdout=file)
					file.flush()
					file.close()