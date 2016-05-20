#! /usr/bin/env python

import data_parser as parser
import plot_styles as ps

data = parser.get_series()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def get_color(series):
    if series.smem:
        return ps.color_wheel[0]
    else:
        return ps.color_wheel[1]

dt_list = [1e-6, 1e-4]
name_list = set()
#plot of smem vs non-smem
for dt in dt_list:
    for mech in data:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.set_xscale("log", nonposx='clip', base=2)
        ax.set_yscale("log", nonposy='clip')

        series = [s for s in data[mech] if 
                    s.gpu and s.dt == dt and
                    not s.cache_opt and
                    not s.finite_difference]

        for i, s in enumerate(series):
            marker, filled = ps.marker_dict[s.name]
            if s.smem:
                s.set_marker(marker=marker, color=get_color(s),
                                **ps.marker_style)
            else:
                s.set_clear_marker(marker=marker, color=get_color(s),
                                    **ps.clear_marker_style)
            s.plot(ax)
            name_list = name_list.union([s.name])

        for name in name_list:
            smem = next((s for s in series if s.name == name and s.smem), None)
            nsmem = next((s for s in series if s.name == name and not s.smem), None)
            if smem and nsmem:
                l = min(len(smem.x), len(nsmem.x))
                ydiff = smem.y[:l] / nsmem.y[:l]
                x = smem.x[np.where(ydiff == np.min(ydiff))[0]]
                print mech, dt, name, np.mean(ydiff), np.std(ydiff), np.min(ydiff), x

        artists = []
        labels = []
        for name in name_list:
            color = 'k'
            show = '\\texttt{{\\textbf{{{}}}}}'.format(name)
            artist = plt.Line2D((0,1),(0,0), 
                markerfacecolor=color, marker=ps.marker_dict[name][0],
                markeredgecolor=color, linestyle='',
                markersize=15)
            artists.append(artist)
            labels.append(show)

        artists.append(mpatches.Patch(color=ps.color_wheel[0]))
        labels.append('SMEM Caching')

        artists.append(mpatches.Patch(color=ps.color_wheel[1]))
        labels.append('No SMEM Caching')

        plt.legend(artists, labels, **ps.legend_style)
        plt.xlabel('Number of ODEs', )
        plt.ylabel('Runtime (s)')
        ps.finalize()
        plt.savefig('figures/{}_{:.0e}_smem.pdf'.format(mech, dt))
        plt.close()