#! /usr/bin/env python

import data_parser as parser
import plot_styles as ps
from thresholds import get_threshold

data = parser.get_series()

from optionloop import OptionLoop as op
import numpy as np
import matplotlib.pyplot as plt

oploop = op({'dt' : [1e-6, 1e-4],
            'gpu' : [True, False],
            'mech' : data.keys()})

smem = False
normalize = False

#guarentee the same colors between plots
name_list = set()
for mech in data:
    name_list = name_list.union([s.name for s in data[mech]])
color_dict = {}
color_list = iter(ps.color_wheel)
for name in name_list:
    color_dict[name] = color_list.next()

for state in oploop:
    dt = state['dt']
    gpu = state['gpu']
    mech = state['mech']

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_xscale("log", nonposx='clip')
    ax.set_yscale("log", nonposy='clip')

    series = [s for s in data[mech] if 
                s.gpu == gpu and
                s.dt == dt and
                (not s.gpu or (s.gpu and s.smem == smem))
                and s.finite_difference == False
                and s.cache_opt == False]
    series = sorted(series, key=lambda x: x.name)
    print mech, 'gpu' if gpu else 'cpu'

    names = set()
    # print mech
    for i, s in enumerate(sorted(series, key=lambda x:x.name)):
        print s
        assert s.name in ps.marker_dict
        marker, hollow = ps.marker_dict[s.name]
        color = color_dict[s.name]
        if hollow:
            s.set_clear_marker(marker=marker, color=color, **ps.clear_marker_style)
        else:
            s.set_marker(marker=marker, color=color, **ps.marker_style)

        if normalize:
            for i in range(len(s.data)):
                s.data[i] = (s.data[i][0], s.data[i][1] / s.data[i][0], s.data[i][2] / s.data[i][0])

        s.plot(ax, ps.pretty_names, show_dev=True)
        names = names.union([s.name])

    #draw threshold
    
    if normalize:
        x_t = get_threshold(mech, gpu, dt)
        plt.axvline(x_t, color='k')
    #make legend
    plt.legend(**ps.legend_style)

    plt.xlabel('Number of ODEs')
    if normalize:
        plt.ylabel('Runtime / ODE (s)')
    else:
        plt.ylabel('Runtime (s)')
    #final stylings
    ps.finalize()
    plt.savefig('figures/{}_{:.0e}_{}{}.pdf'.format(mech, dt,
        'gpu' if gpu else 'cpu', '' if normalize else '_nonorm'))
    plt.close()