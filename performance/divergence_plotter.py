#! /usr/bin/env python2.7
import os
import matplotlib
import numpy as np
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

agg_preamble = [r'\usepackage{siunitx}',
                r'\sisetup{detect-all}']
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

marker_list = ['o', 's']
color_list = ['r', 'b']

dt_dict = {}
mech_dict = {}
num_bins = 25

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
for f in os.listdir('divergence/'):
    if not f.endswith('div.txt'):
        continue
    args = f.split('_')
    mech = args[0].strip()
    solver = args[1].strip()
    dt = args[2].strip()

    div = []
    with open('divergence/' + f) as file:
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

                bins, edges = np.histogram(data.div, num_bins, (0., 1.))
                x = (edges[:-1] + edges[1:]) / 2
                y = bins[:]
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

        plt.legend(loc='upper left', fontsize=22, numpoints=1,
                        shadow=True, fancybox=True)
        plt.xlabel('Divergence Measure $D$')
        plt.ylabel('Number of Occurances')
        ax.set_ylim((0, None))
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(20)
        plt.savefig('figures/{}_{}_div.pdf'.format(mech, solver))
        plt.close()