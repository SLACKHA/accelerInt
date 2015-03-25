#! /usr/bin/env python2.7

#timing_plotter.py
#compares timing plots for the various integrators
from os import listdir, getcwd
from os.path import isfile, join
from math import pow
import re
import matplotlib.pyplot as plt
import sys

from argparse import ArgumentParser

parser = ArgumentParser(description='Runs all integrators for the given mechanism / options')
parser.add_argument('-n', '--name',
                    type=str,
                    required=True,
                    help = 'Name of the mechanism, used in title construction')
parser.add_argument('-diff', '--diff_ics',
                    default=False,
                    action='store_true',
                    required=False,
                    help = 'Data was generated using different initial condition')
parser.add_argument('-sh', '--shuffle',
                    default=False,
                    action='store_true',
                    required=False,
                    help = 'Data was generated using shuffled initial condition. Implies Different ICS')
parser.add_argument('-dir', '--directory',
                    default=getcwd(),
                    required=False,
                    help = 'The directory containing the files.')

args = parser.parse_args()
Title = args.name + ' Timing'
if args.shuffle:
    Title += ' - Shuffled ICs'
elif args.diff_ics:
    Title += ' - Different ICs'
else:
    Title += ' - Same ICs'
print args.directory

onlyfiles = [ f for f in listdir(args.directory) if isfile(join(args.directory, f)) and re.search("\d.txt$", f) ]

def parse_cpu(file):
    with open(file) as infile:
        lines = [line.strip() for line in infile]
    odes = int(lines[0][lines[0].index(":") + 1:])
    threads = int(lines[1][lines[1].index(":") + 1:])
    time = float(lines[2][lines[2].index(':') + 1:lines[2].index('sec')])
    return odes, threads, time

def parse_gpu(file):
    with open(file) as infile:
        lines = [line.strip() for line in infile]
    odes = int(lines[0][lines[0].index(":") + 1 : lines[0].index('block')])
    block = int(lines[0][lines[0].rindex(":") + 1:])
    time = float(lines[1][lines[1].index(':') + 1:lines[1].index('sec')])
    return odes, block, time

fig = plt.figure()
plot = fig.add_subplot(1,1,1)
data = {}
gpu_data = {}
for file in onlyfiles:
    descriptor = file[:file.index('-int')]
    if 'gpu' in file:
        descriptor += '-gpu'
        if not descriptor in gpu_data:
            gpu_data[descriptor] = {}
        odes, block, time = parse_gpu(join(args.directory, file))
        if not block in gpu_data[descriptor]:
            gpu_data[descriptor][block] = []
        gpu_data[descriptor][block].append((odes, time))
    else:
        if not descriptor in data:
            data[descriptor] = {}
        odes, threads, time = parse_cpu(join(args.directory, file))
        if not threads in data[descriptor]:
            data[descriptor][threads] = []
        data[descriptor][threads].append((odes, time))

for desc in data:
    for threads in data[desc]:
        data[desc][threads] = sorted(data[desc][threads], key = lambda val: val[0])

for desc in gpu_data:
    for block in gpu_data[desc]:
        gpu_data[desc][block] = sorted(gpu_data[desc][block], key = lambda val: val[0])
		
#plot
for desc in data:
    for threads in data[desc]:
        plt.loglog(*zip(*data[desc][threads]), label = desc + " - " + str(threads) + " threads", marker = "v", basex=2)
for desc in gpu_data:
    for block in gpu_data[desc]:
        plt.loglog(*zip(*gpu_data[desc][block]), label = desc + ' - blocksize: ' + str(block), marker = ">", basex=2)

plt.legend(loc = 0, fontsize=10).draggable(state = True)
plt.xlabel("ODEs")
plt.ylabel("Time (s)")
plt.title(Title)
plt.savefig(Title + '.png')
plt.close()
