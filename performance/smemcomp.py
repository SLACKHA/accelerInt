#! /usr/bin/env python

import parser as parser

data = parser.get_series()

import matplotlib
import numpy as np
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

blacklist = ['cvodes']
marker_list = ['v', '*', '>']
marker_dict = {}
smem_color = 'b'
nosmem_color = 'r'
def get_color(series):
    if series.smem:
        return smem_color
    else:
        return nosmem_color

dt_list = [1e-6, 1e-4]
#plot of smem vs non-smem
for dt in dt_list:
    for mech in data:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.set_xscale("log", nonposx='clip', base=2)
        ax.set_yscale("log", nonposy='clip')

        series = [s for s in data[mech] if 
                    s.gpu and s.dt == dt]

        for i, s in enumerate(series):
            if any(s.name == x for x in blacklist):
                continue
            if s.name == 'radau2a':
                if s.smem:
                    radau_smem = s
                else:
                    radau_nosmem = s
            if not s.name in marker_dict:
                marker_dict[s.name] = marker_list.pop()
            if s.smem:
                s.set_marker(marker=marker_dict[s.name], color=get_color(s),
                                size=15)
            else:
                s.set_clear_marker(marker=marker_dict[s.name], color=get_color(s),
                                    size=17)
            s.plot(ax)

        for name in marker_dict:
            smem = next((s for s in series if s.name == name and s.smem), None)
            nsmem = next((s for s in series if s.name == name and not s.smem), None)
            if smem and nsmem:
                ydiff = smem.y / nsmem.y
                x = smem.x[np.where(ydiff == np.min(ydiff))[0]]
                print mech, dt, name, np.mean(ydiff), np.std(ydiff), np.min(ydiff), x

        artists = []
        labels = []
        for name in marker_dict:
            color = 'k'
            show = '\\texttt{\\textbf{cvode}}' if name == 'cvodes-analytic' else \
                    '\\texttt{{\\textbf{{{}}}}}'.format(name)
            artist = plt.Line2D((0,1),(0,0), 
                markerfacecolor=color, marker=marker_dict[name],
                markeredgecolor=color, linestyle='',
                markersize=15)
            artists.append(artist)
            labels.append(show)

        artists.append(mpatches.Patch(color=smem_color))
        labels.append('SMEM Caching')

        artists.append(mpatches.Patch(color=nosmem_color))
        labels.append('No SMEM Caching')

        plt.legend(artists, labels, loc=0, fontsize=16, numpoints=1,
                    shadow=True, fancybox=True)
        plt.xlabel('Number of ODEs', )
        plt.ylabel('Runtime (s)')
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(20)
        plt.savefig('figures/{}_{:.0e}_smem.pdf'.format(mech, dt))
        plt.close()