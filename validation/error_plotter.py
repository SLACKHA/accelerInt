#! /usr/bin/env python2.7
import matplotlib
import numpy as np
import os
import sys
sys.path.append('../performance/')
import plot_styles as ps
from data_parser import data_series
import matplotlib.pyplot as plt
import re
import os
from optionloop import OptionLoop

lines = []
files = [f for f in os.listdir(os.getcwd()) if f.endswith('logfile') and os.path.isfile(f)]
for f in files:
    with open(f, 'r') as file:
        lines.extend([l.strip() for l in file.readlines() if l.strip()])

series_list = []
PaSR = None
opt = None
smem = None
timestep = None
for line in lines:
    if not line.strip():
        continue
    if 'lang' in line:
        lang = line[line.index(':') + 1:]
        continue
    if 'PaSR ICs' in line:
        PaSR = True
        continue
    elif 'Same ICs' in line:
        PaSR = False
        continue
    match = re.search(r'cache_opt:\s*(\w+)', line)
    if match:
        opt = match.group(1) == 'True'
        continue
    match = re.search(r'shared_mem:\s*(\w+)', line)
    if match:
        smem = match.group(1) == 'True'
        continue
    match = re.search(r't_step=(\d+e(?:-)?\d+)', line)
    if match:
        timestep = float(match.group(1))
        continue
    match = re.search(r'log/([\w\d-]+)-log.bin', line)
    if match:
        solver = match.group(1)
        solver = solver[:solver.index('-int')]
        continue
    match = re.search(r'L2 \(max, mean\) = (nan, nan)', line)
    match2 = re.search(r'L2 \(max, mean\) = (\d+\.\d+e(?:[+-])?\d+)', line)
    if match or match2:
        yval = np.nan if (match and not match2) else float(match2.group(1))
        test = data_series(solver, gpu=lang=='cuda', cache_opt=opt, smem=smem, finite_difference=False)
        series = next((x for x in series_list if x == test), None)
        if series is None:
            series_list.append(test)
            series = test
        series.add_x_y(timestep, yval)
        continue
    if 'Linf' in line:
        continue
    raise Exception(line)

c_params = OptionLoop({'gpu' : False, 
            'opt' : [True, False],
            'same_ics' : [False]}, lambda: False)
cuda_params = OptionLoop({'gpu' : True, 
            'opt' : [True, False],
            'smem' : [True, False],
            'same_ics' : [False]}, lambda: False)
#create color dictionary
color_dict = {}
color_list = iter(ps.color_wheel)
for x in series_list:
    if not x.name in color_dict:
        color_dict[x.name] = color_list.next()

op = c_params + cuda_params
for state in op:
    gpu = state['gpu']
    opt = state['opt']
    smem = state['smem']

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_xscale("log", nonposx='clip')
    ax.set_yscale("log", nonposy='clip')

    data_list = [x for x in series_list if x.gpu == gpu and
                    x.cache_opt == opt and x.smem == smem]

    for s in sorted(data_list, key = lambda x: x.name):
        marker, hollow = ps.marker_dict[s.name]
        color = color_dict[s.name]
        s.set_clear_marker(marker=marker, color=color, **ps.clear_marker_style)
        s.plot(ax, ps.pretty_names)

    plt.xlabel(r'$\delta t(s)$')
    plt.ylabel(r'$\left\lvert\textbf{E}\right\rvert$')

    plt.legend(**ps.legend_style)
    ps.finalize()

    #figure out dummy slope
    exp4 = next(x for x in data_list if x.name == 'exp4')
    slope = (np.log(x.y[3]) - np.log(x.y[0])) / (np.log(x.x[3]) - np.log(x.x[0]))
    print slope

    plt.savefig('{}_{}_{}_error.pdf'.format(
        'c' if not gpu else 'cuda',
        'co' if opt else 'nco', 
        'smem' if smem else 'nosmem'))
    plt.close()