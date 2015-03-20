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
maker = ['make', '-j24', 'DEBUG=FALSE', 'FAST_MATH=FALSE', 'IGN=TRUE', 'PRINT=FALSE',  'LOG_OUTPUT=FALSE']
if args.shuffle:
	maker.extend(["SAME_IC=FALSE", "SHUFFLE=TRUE"])
elif args.different_ics:
	maker.extend(["SAME_IC=FALSE"])

#force remake
subprocess.call(maker)

all_exes = []
for file in glob.glob('*-int*'):
	all_exes.append(file)

all_exes = sorted(all_exes, key = lambda x: 1000 if 'gpu' in x else 1)

threads = [12]
powers = [0, 3, 7, 10, 14, 17]
exppowers = [int(pow(2, exponent)) for exponent in powers]

for exe in all_exes:
	for thread in threads:
		for i in range(len(powers)):
			filename = './output/' + exe + '_' + str(thread) +'_'+str(powers[i])+'.txt'
			if (os.path.isfile(filename)):
				continue
			file = open(filename, "w")
			if 'gpu' in filename:
				subprocess.call([os.path.join(os.getcwd(), exe), str(exppowers[i])], stdout=file)
			else:
				subprocess.call([os.path.join(os.getcwd(), exe), str(thread), str(exppowers[i])], stdout=file)
			file.flush()
			file.close()