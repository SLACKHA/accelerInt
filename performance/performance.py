#! /usr/bin/env python

import parser as parser
import plot_styles as ps

data = parser.get_series()

import numpy as np
import matplotlib.pyplot as plt

for gpu in [True, False]:
    for mech in data:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.set_xscale("log", nonposx='clip')
        ax.set_yscale("log", nonposy='clip')

        series = [s for s in data[mech] if 
                    s.gpu == gpu and
                    (not s.gpu or (s.gpu and s.smem))
                    and s.finite_difference == False
                    and s.cache_opt == False]
        series = sorted(series, key=lambda x: 0 if x.gpu else 1)
        print mech, 'gpu' if gpu else 'cpu'

        def name_fun(x):
            return x.name + (' - gpu' if x.gpu else '')

        names = set()
        # print mech
        for i, s in enumerate(series):
            print s
            assert s.name in ps.marker_dict
            marker = ps.marker_dict[s.name]
            color = ps.get_color(s, lambda x: x.dt==1e-6)
            s.set_clear_marker(marker=marker, color=color, **ps.clear_marker_style)

            #normalize per dt
            for i in range(len(s.data)):
                s.data[i] = (s.data[i][0], s.data[i][1] / fac, s.data[i][2] / fac)

            s.plot(ax, name_fun)
            names = names.union([s.name])

        #make legend
        ps.make_legend(names, patch_names=[r'$\delta = \num{e-6}$', r'$\delta = \num{e-4}$'])

        plt.xlabel('Number of ODEs')
        plt.ylabel('Runtime (s)')
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(ps.tick_font_size)
        plt.savefig('figures/{}_{}.pdf'.format(mech, 'gpu' if gpu else 'cpu'))
        plt.close()