#! /usr/bin/env python

import parser as parser

data = parser.get_series()

import matplotlib
import numpy as np
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def pretty_names(name):
    if name == 'cvodes-analytic':
        return 'CVODE'
    elif name == 'radau2a':
        return 'Radau-IIA'
    return name

blacklist = ['cvodes', 'hradau2a']
marker_list = ['v', 's', 'o', '>', '*']
marker_dict = {}
gpu_color = 'b'
cpu_color = 'r'
def get_color(series):
    if series.gpu:
        return gpu_color
    else:
        return cpu_color

def __get_name(series):
    return '{}-GPU'.format(series.name) if series.gpu else series.name 
def __print_stats(series1, series2, mech, dt, cutoff=1024):
    real_x = series1.x
    x = np.where(real_x > cutoff)
    ydiff = series1.y / series2.y

    ymin = np.min(ydiff[x])
    minx = real_x[np.where(ydiff==ymin)][0]
    ymax = np.max(ydiff[x])
    maxx = real_x[np.where(ydiff==ymax)][0]
    print __get_name(series1), __get_name(series2), mech, dt, (minx, ymin), (maxx, ymax)


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
        series = sorted(series, key=lambda x: 0 if x.gpu else 1)

        radau_gpu = next(s for s in series if s.name == 'radau2a' and s.gpu)
        radau_cpu = next(s for s in series if s.name == 'radau2a' and not s.gpu)
        exp4_cpu = next(s for s in series if s.name == 'exp4' and not s.gpu)
        exp4_gpu = next((s for s in series if s.name == 'exp4' and s.gpu), None)
        exprb43_cpu = next(s for s in series if s.name == 'exprb43' and not s.gpu)
        exprb43_gpu = next((s for s in series if s.name == 'exprb43' and s.gpu), None)
        cvode = next(s for s in series if s.name == 'cvodes-analytic')

        def name_fun(x):
            return x.name + (' - gpu' if x.gpu else '')

        cutoff = 8192
        # print mech
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
            # perc = 100. * s.z / s.y
        #     mycut = cutoff
        #     if 'cvodes' in s.name or 'radau2a' in s.name:
        #         mycut = 65536
        #     if not s.gpu or s.smem:
        #         name = s.name[:min(7, len(s.name))]
        #         print "{}\t{}\t{}\t{}\t{}\t{}\t{}".format(name, s.gpu, s.dt, np.mean(perc), np.std(perc), np.mean(perc[np.where(s.x > mycut)]), np.std(perc[np.where(s.x > mycut)]))
        # print

        __print_stats(radau_gpu, cvode, mech, dt, cutoff)
        __print_stats(radau_cpu, cvode, mech, dt, cutoff)
        if exp4_gpu:
            __print_stats(exp4_gpu, exp4_cpu, mech, dt, cutoff)
        if exprb43_gpu:
            __print_stats(exprb43_gpu, exprb43_cpu, mech, dt, cutoff)
        __print_stats(exprb43_cpu, exp4_cpu, mech, dt, cutoff)
        __print_stats(exprb43_cpu, cvode, mech, dt, cutoff)
        __print_stats(exp4_cpu, cvode, mech, dt, cutoff)
        print

        artists = []
        labels = []
        for name in marker_dict:
            show = '\\texttt{{\\textbf{{{}}}}}'.format(pretty_names(name))

            artist = plt.Line2D((0,1),(0,0),
                        linestyle='',
                        marker=marker_dict[name],
                        markersize=15,
                        markerfacecolor='k')
            if name == 'cvodes-analytic':
                artist = plt.Line2D((0,1),(0,0),
                        linestyle='',
                        marker=marker_dict[name],
                        markersize=15,
                        markeredgecolor=cpu_color,
                        markeredgewidth=1,
                        markerfacecolor='None')
            #     markerfacecolor='None'
            #     markeredgecolor=cpu_color
            #     markeredgewidth=1
            #     artist = plt.Line2D((0,1),(0,0), 
            #         markerfacecolor=markerfacecolor, marker=marker_dict[name],
            #         markeredgecolor=markeredgecolor, linestyle='',
            #         markeredgewidth=markeredgewidth, markersize=15)
            # else:
            #     markerfacecolor=gpu_color
            #     markeredgecolor=gpu_color

            #     markeredgewidth=0
            #     show = '\\texttt{{\\textbf{{{}}}}}'.format(name)
            #     artist = plt.Line2D((0,1),(0,0), 
            #         markerfacecolor=markerfacecolor, marker=marker_dict[name],
            #         linestyle='', markerfacecoloralt='None', 
            #         fillstyle='bottom', markersize=15)
            artists.append(artist)
            labels.append(show)

        artists.append(mpatches.Patch(color=gpu_color))
        labels.append('GPU')

        artists.append(mpatches.Patch(facecolor='None', edgecolor=cpu_color))
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