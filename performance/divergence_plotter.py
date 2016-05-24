#! /usr/bin/env python2.7
import os
import matplotlib
import numpy as np
import plot_styles as ps
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from glob import glob
from matplotlib.ticker import FuncFormatter
import plot_styles as ps

marker_list = ['o', 's']
color_list = ['r', 'b']

dt_dict = {}
mech_dict = {}
num_bins = 25

def to_percent(y, position):
    # Ignore the passed in position. This has the effect of scaling the default
    # tick locations.
    s = str(y) #str(100 * y)

    # The percent symbol needs escaping in latex
    if matplotlib.rcParams['text.usetex'] is True:
        return s + r'$\%$'
    else:
        return s + '%'

class data(object):
    def __init__(self, mech, solver, dt, div):
        self.mech = mech
        self.solver = solver
        self.dt = dt
        self.div = div

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return ('mech: {}\t'.format(self.mech) +
            'solver: {}\t'.format(self.solver) +
            'dt: {}\t'.format(self.dt))

data_list = []
solver_list = []
mech_list = []
dt_list = []
for f in glob('divergence/*/*/*.txt'):
    if not f.endswith('div.txt'):
        continue
    args = f.split('/')
    mech = args[1].strip()
    dt = float(args[2].strip())
    solver = args[3][:args[3].index('-int')]

    div = []
    with open(f) as file:
        div = [float(d.strip()) for d in file.readlines()]

    data_list.append(data(mech, solver, dt, div))

    if not solver in solver_list:
        solver_list.append(solver)

    if not mech in mech_list:
        mech_list.append(mech)

    if not dt in dt_list:
        dt_list.append(dt)

for solver in solver_list:
    for mech in mech_list:

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        for dt in dt_list:
            solver_data = [x for x in data_list if x.solver == solver and 
                            x.mech == mech and
                            x.dt == dt]

            for data in solver_data:
                print data
                if dt not in mech_dict:
                    mech_dict[dt] = color_list.pop()

                if dt not in dt_dict:
                    dt_dict[dt] = marker_list.pop()

                bins, edges = np.histogram(data.div, num_bins, range=(0., 1.))
                formatter = FuncFormatter(to_percent)

                x = (edges[:-1] + edges[1:]) / 2
                y = 100. * bins / float(np.sum(bins))
                zeros = np.where(y==0)
                if zeros:
                    y[zeros[0]] = -1000

                power = int(np.log10(float(dt)))
                if dt_list.index(dt):
                    plt.plot(x, y, 
                        label=r'$\delta t = 1 \times 10^{{{}}}$'.format(power), 
                        linestyle='', 
                        marker=dt_dict[dt],
                        markerfacecolor=mech_dict[data.dt],
                        markeredgewidth=0,
                        markersize=12)
                else:
                    plt.plot(x, y, 
                        label=r'$\delta t = 1 \times 10^{{{}}}$'.format(power), 
                        linestyle='', 
                        marker=dt_dict[dt],
                        markerfacecolor='None',
                        markeredgecolor=mech_dict[data.dt],
                        markeredgewidth=3,
                        markersize=15)

        ps.legend_style['fontsize']=22
        ps.legend_style['loc']= 'upper left'
        plt.legend(**ps.legend_style)
        plt.xlabel('Divergence Measure $D$')
        plt.ylabel('Percent of Total Warps')
        ax.set_ylim((0, 105))
        plt.gca().yaxis.set_major_formatter(formatter)
        ps.finalize()
        plt.savefig('figures/{}_{}_div.pdf'.format(mech, solver))
        plt.close()