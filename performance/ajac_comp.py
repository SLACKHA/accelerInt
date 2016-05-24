#! /usr/bin/env python

"""
ajac_comp.py - a simple script to compare chemical kinetic integration performance using
analytical and finite difference jacobian's on various platforms
"""

import copy

import data_parser as parser
import plot_styles as ps

data = parser.get_series()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

def make_comp_legend(names, loc=0, patch_names=None):
    artists = []
    labels = []
    for name in names:
        show = ps.pretty_names(name)

        artist = plt.Line2D((0,1),(0,0),
                    linestyle='',
                    marker=ps.marker_dict[name][0],
                    markersize=15,
                    markerfacecolor='k',
                    markeredgecolor='none')
        artists.append(artist)
        labels.append(show)

    if patch_names is not None:
        artists.append(mpatches.Patch(facecolor='None', edgecolor=ps.color_wheel[0]))
        labels.append(patch_names[0])

        artists.append(mpatches.Patch(facecolor='None', edgecolor=ps.color_wheel[1]))
        labels.append(patch_names[1])

    plt.legend(artists, labels, **ps.legend_style)

dt_list = [1e-6, 1e-4]
#plot of smem vs non-smem
for dt in dt_list:
    for mech in data:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.set_xscale("log", nonposx='clip')
        ax.set_yscale("log", nonposy='clip')

        series = [s for s in data[mech] if 
                    (s.gpu == False or (s.gpu and s.smem == False)) and s.dt == dt
                    and s.cache_opt == False]
        series = sorted(series, key=lambda x: 0 if x.gpu else 1)
        print mech, dt

        names = set([s.name for s in series if s.name != 'cvodes'])
        plot_items = []
        for name in names:
            ajac_gpu = next((s for s in series if s.name == name and s.gpu and not s.finite_difference), None)
            fd_gpu = next((s for s in series if s.name == name and s.gpu and s.finite_difference), None)
            if ajac_gpu and fd_gpu:
                plot_items.append((fd_gpu, ajac_gpu))

            ajac_cpu = next((s for s in series if s.name == name and not s.gpu and not s.finite_difference), None)
            fd_cpu = next((s for s in series if s.name == name and not s.gpu and s.finite_difference), None)
            if ajac_cpu and fd_cpu:
                plot_items.append((fd_cpu, ajac_cpu))

        def name_fun(x):
            return x.name + (' - gpu' if x.gpu else '')

        names = set()
        # print mech
        for i, s in enumerate(sorted(plot_items, key=lambda x: x[0].name)):
            #create the ratio series
            dummy = copy.copy(s[0])
            dummy.data = []
            for i in range(len(s[0].data)):
                point1 = s[0].data[i]
                point2 = next(point for point in s[1].data if point[0] == point1[0])

                #compute ratio of FD / ajac
                ratio = point1[1] / point2[1]

                #uncertainty propigation
                dev = ratio * np.sqrt(np.power(point1[2] / point1[1], 2.) + np.power(point2[2] / point2[1], 2.))

                #add to the dummy series to plot
                dummy.data.append((point1[0], ratio, dev))

            assert dummy.name in ps.marker_dict
            marker, hollow = ps.marker_dict[dummy.name]
            color = ps.color_wheel[0] if dummy.gpu else ps.color_wheel[1]
            if hollow:
                dummy.set_clear_marker(marker=marker, color=color, **ps.clear_marker_style)
            else:
                dummy.set_marker(marker=marker, color=color, **ps.marker_style)
            dummy.plot(ax, name_fun, show_dev=True)
            names = names.union([dummy.name])

        max_x = ax.get_xlim()[1]
        dummy_x = np.linspace(0, max_x, num=1000, endpoint=True)
        dummy_y = np.ones_like(dummy_x)
        plt.plot(dummy_x, dummy_y, 'k')

        #top left
        ps.legend_style['loc'] = 2
        #make legend
        make_comp_legend(names, patch_names=['GPU', 'CPU'])

        plt.xlabel('Number of ODEs')
        plt.ylabel(r'$\lvert \textbf{R}_{FD}\rvert\slash\lvert \textbf{R}_{AJ}\rvert$')
        ps.finalize()
        plt.savefig('figures/{}_{:.0e}_ajac_comp.pdf'.format(mech, dt))
        plt.close()