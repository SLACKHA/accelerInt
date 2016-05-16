#! /usr/bin/env python

import parser as parser
import plot_styles as ps

data = parser.get_series()

from optionloop import OptionLoop as op
import numpy as np
import matplotlib.pyplot as plt

oploop = op({'dt' : [1e-6, 1e-4],
            'gpu' : [True, False],
            'mech' : data.keys()})

normalize=True

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
                (not s.gpu or (s.gpu and s.smem))
                and s.finite_difference == False
                and s.cache_opt == False]
    series = sorted(series, key=lambda x: 0 if x.gpu else 1)
    print mech, 'gpu' if gpu else 'cpu'

    def name_fun(series):
        return '\\texttt{{\\textbf{{{}}}}}'.format(ps.pretty_names(series.name))

    color_list = iter(ps.color_wheel)
    names = set()
    # print mech
    for i, s in enumerate(series):
        print s
        assert s.name in ps.marker_dict
        marker, hollow = ps.marker_dict[s.name]
        color = color_list.next()
        if hollow:
            s.set_clear_marker(marker=marker, color=color, **ps.clear_marker_style)
        else:
            s.set_marker(marker=marker, color=color, **ps.marker_style)

        if normalize:
            for i in range(len(s.data)):
                s.data[i] = (s.data[i][0], s.data[i][1] / s.data[i][0], s.data[i][2] / s.data[i][0])

        s.plot(ax, name_fun)
        names = names.union([s.name])

    #make legend
    plt.legend(**ps.legend_style)

    plt.xlabel('Number of ODEs')
    if normalize:
        plt.ylabel('Runtime / ODE (s)')
    else:
        plt.ylabel('Runtime (s)')
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
         ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(ps.tick_font_size)
    plt.savefig('figures/{}_{:.0e}_{}.pdf'.format(mech, dt,
        'gpu' if gpu else 'cpu'))
    plt.close()