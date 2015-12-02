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
marker_list = ['v', '*', '>', 's', 'o']
marker_dict = {}
gpu_color = 'b'
cpu_color = 'r'
def get_color(series):
    if series.gpu:
        return gpu_color
    else:
        return cpu_color

dt_list = [1e-6, 1e-4]
#plot of smem vs non-smem
for dt in dt_list:
    for mech in data:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.set_xscale("log", nonposx='clip')
        ax.set_yscale("log", nonposy='clip')

        series = [s for s in data[mech] if 
                    ((s.gpu == False and s.threads==6) or 
                    (s.gpu and s.smem)) and s.dt == dt]

        radau_gpu = next(s for s in series if s.name == 'radau2a' and s.gpu)
        cvode = next(s for s in series if s.name == 'cvodes-analytic')

        def name_fun(x):
            return x.name + (' - gpu' if x.gpu else '')

        cutoff = 1024
        print mech
        for i, s in enumerate(series):
            if any(s.name == x for x in blacklist):
                continue
            if not s.name in marker_dict:
                marker_dict[s.name] = marker_list.pop()
            if s.gpu:
                s.set_marker(marker=marker_dict[s.name], color=get_color(s), size=15)
            else:
                s.set_clear_marker(marker=marker_dict[s.name], color=get_color(s),
                                    size=17)
            s.plot(ax, name_fun)
            perc = 100. * s.z / s.y
            mycut = cutoff
            if 'cvodes' in s.name or 'radau2a' in s.name:
                mycut = 65536
            if not s.gpu or s.smem:
                name = s.name[:min(7, len(s.name))]
                print "{}\t{}\t{}\t{}\t{}\t{}\t{}".format(name, s.gpu, s.dt, np.mean(perc), np.std(perc), np.mean(perc[np.where(s.x > mycut)]), np.std(perc[np.where(s.x > mycut)]))
        print

        real_x = radau_gpu.x
        x = np.where(real_x > cutoff)
        ydiff = radau_gpu.y / cvode.y
        faster = np.mean(ydiff[x]) < 1
        best = np.min(ydiff)
        if faster:
            ydiff = cvode.y / radau_gpu.y
            best = np.max(ydiff)
        #print faster, mech, dt, np.mean(ydiff[x]), np.std(ydiff[x]), best

        artists = []
        labels = []
        for name in marker_dict:
            if name == 'cvodes-analytic':
                show = '\\texttt{\\textbf{cvode}}'
                markerfacecolor='None'
                markeredgecolor='r'
                markeredgewidth=1
            else:
                markerfacecolor='k'
                markeredgecolor='k'
                markeredgewidth=0
                show = '\\texttt{{\\textbf{{{}}}}}'.format(name)
            artist = plt.Line2D((0,1),(0,0), 
                markerfacecolor=markerfacecolor, marker=marker_dict[name],
                markeredgecolor=markeredgecolor, linestyle='',
                markeredgewidth=markeredgewidth, markersize=15)
            artists.append(artist)
            labels.append(show)

        artists.append(mpatches.Patch(color=gpu_color))
        labels.append('GPU')

        artists.append(mpatches.Patch(color=cpu_color))
        labels.append('CPU')

        plt.legend(artists, labels, loc=0, fontsize=16, numpoints=1,
                    shadow=True, fancybox=True)
        plt.xlabel('Number of ODEs')
        plt.ylabel('Runtime (s)')
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(20)
        plt.savefig('figures/{}_{:.0e}_cpuvsgpu.pdf'.format(mech, dt))
        plt.close()