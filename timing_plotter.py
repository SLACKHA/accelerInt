#! /usr/bin/env python2.7

#timing_plotter.py
#compares timing plots for the various integrators
from os import listdir, getcwd
from os.path import isfile, join
from math import pow
import re
import matplotlib.pyplot as plt
from matplotlib import cm
import sys
from argparse import ArgumentParser
from numpy import linspace

def parse_cpu(file):
    try:
        with open(file) as infile:
            lines = [line.strip() for line in infile]
        odes = int(lines[0][lines[0].index(":") + 1:])
        threads = int(lines[1][lines[1].index(":") + 1:])
        time = float(lines[2][lines[2].index(':') + 1:lines[2].index('sec')])
        return odes, threads, time
    except:
        return None

def parse_gpu(file):
    try:
        with open(file) as infile:
            lines = [line.strip() for line in infile]
        odes = int(lines[0][lines[0].index(":") + 1 : lines[0].index('block')])
        block = int(lines[0][lines[0].rindex(":") + 1:])
        time = float(lines[1][lines[1].index(':') + 1:lines[1].index('sec')])
        return odes, block, time
    except:
        return None

def time_plotter(Title, directory, out_dir='', plot_vs_blocksize=False):

    onlyfiles = [ f for f in listdir(directory) if isfile(join(directory, f)) and re.search(".*int.*\.txt$", f) ]

    if plot_vs_blocksize and any(x for x in onlyfiles if not 'gpu' in x):
        print 'Plot Vs Blocksize not supported for CPU data!'
        sys.exit(-1)

    filter_vals = ['', '_lt', '_ls', '_lt_ls']

    for filter_text in filter_vals:
        Title_out = Title
        if '_lt' in filter_text:
            Title_out += ' - Large Tol'
        if '_ls' in filter_text:
            Title_out += ' - Large Stepsize'

        the_files = [f for f in onlyfiles if re.search('\d' + filter_text + '\.txt', f)]
        print filter_text, the_files
        if not len(the_files):
            continue
        fig = plt.figure()
        plot = fig.add_subplot(1,1,1)
        data = {}
        gpu_data = {}
        for file in the_files:
            if plot_vs_blocksize:
                descriptor = file[file.index('_') + 1:file.rindex('_')]
            else:
                descriptor = file[:file.index('-int')]
            if 'gpu' in file:
                if not plot_vs_blocksize:
                    descriptor += '-gpu'
                if not descriptor in gpu_data:
                    gpu_data[descriptor] = {}
                tup = parse_gpu(join(directory, file))
                if tup is None:
                    continue
                odes, block, time = tup 
                if plot_vs_blocksize:
                    if not block in gpu_data[descriptor]:
                        gpu_data[descriptor][block] = time
                    else:
                        val = gpu_data[descriptor][block]
                        gpu_data[descriptor][block] = min(time, val)
                else:
                    if not block in gpu_data[descriptor]:
                        gpu_data[descriptor][block] = []
                    gpu_data[descriptor][block].append((odes, time))
            else:
                if not descriptor in data:
                    data[descriptor] = {}
                tup = parse_cpu(join(directory, file))
                if tup is None:
                    continue
                odes, threads, time = tup
                if not threads in data[descriptor]:
                    data[descriptor][threads] = []
                data[descriptor][threads].append((odes, time))

        for desc in data:
            for threads in data[desc]:
                data[desc][threads] = sorted(data[desc][threads], key = lambda val: val[0])

        if not plot_vs_blocksize:
            for desc in gpu_data:
                for block in gpu_data[desc]:
                    gpu_data[desc][block] = sorted(gpu_data[desc][block], key = lambda val: val[0])

        #plot
        for desc in data:
            for threads in data[desc]:
                plt.loglog(*zip(*data[desc][threads]), label = desc + " - " + str(threads) + " threads", marker = "v", basex=2)
        
        colorwheel = cm.jet(linspace(0,1,len(gpu_data)))
        index = 0
        for desc in gpu_data:
            if plot_vs_blocksize:
                data = []
                for block in gpu_data[desc]:
                    data.append((block, gpu_data[desc][block]))
                plt.loglog(*zip(*data), label = desc, marker = ">", basex=2, color=colorwheel[index])
                index += 1
            else:
                for block in gpu_data[desc]:
                    plt.loglog(*zip(*gpu_data[desc][block]), label = desc + ' - blocksize: ' + str(block), marker = ">", basex=2)

        plt.legend(loc = 0, fontsize=10).draggable(state = True)
        if plot_vs_blocksize:
            plt.xlabel("Blocksize (# threads)")
        else:
            plt.xlabel("ODEs")
        plt.ylabel("Time (s)")
        plt.title(Title_out)
        print out_dir, Title_out, '.png'
        plt.savefig(join(out_dir, Title_out + '.png'))
        plt.close()

if __name__ == '__main__':
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
    parser.add_argument('-odir', '--out-directory',
                        dest='out_dir',
                        default=getcwd(),
                        required=False,
                        help = 'The directory to output graphs to.')

    args = parser.parse_args()
    Title = args.name + ' Timing'
    if args.shuffle:
        Title += ' - Shuffled ICs'
    elif args.diff_ics:
        Title += ' - Different ICs'
    else:
        Title += ' - Same ICs'

    time_plotter(Title, args.directory, args.out_dir)