#a single consolidated place to import
#such that all figures have identical styling (when possible)

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

#setup latex
plt.rc('text', usetex=True)
plt.rc('text.latex', 
    preamble=r'\usepackage{amsmath},\usepackage{siunitx}')
plt.rc('font', family='serif')

legend_style = {'loc':0,
    'fontsize':16,
    'numpoints':1,
    'shadow':True,
    'fancybox':True
}

tick_font_size = 20

marker_style = {
    'size' : 15
}

clear_marker_style = {
    'size' : 17
}

marker_dict = {'cvodes' : '*',
'radau2a' : '>',
'exp4' : 'o',
'exprb43' : 's'
}

def pretty_names(name):
    if name == 'cvodes':
        return 'CVODE'
    elif name == 'radau2a':
        return 'Radau-IIA'
    return name

color_wheel = ['b', 'r']
def get_color(series, decision_fun):
    if decision_fun(series):
        return color_wheel[0]
    else:
        return color_wheel[1]

def make_legend(names, loc=0, patch_names=None):
    artists = []
    labels = []
    for name in names:
        show = '\\texttt{{\\textbf{{{}}}}}'.format(pretty_names(name))

        artist = plt.Line2D((0,1),(0,0),
                    linestyle='',
                    marker=marker_dict[name],
                    markersize=15,
                    markerfacecolor='k')
        artists.append(artist)
        labels.append(show)

    if patch_names is not None:
        artists.append(mpatches.Patch(facecolor='None', edgecolor=color_wheel[0]))
        labels.append(patch_names[0])

        artists.append(mpatches.Patch(facecolor='None', edgecolor=color_wheel[1]))
        labels.append(patch_names[1])

    plt.legend(artists, labels, **legend_style)