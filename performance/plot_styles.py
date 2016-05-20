#a single consolidated place to import
#such that all figures have identical styling (when possible)

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from data_parser import data_series

#setup latex
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
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

marker_dict = {'cvodes' : ('.', False),
'radau2a' : ('>', True),
'exp4' : ('o', True),
'exprb43' : ('s', True)
}

def pretty_names(pname):
    if isinstance(pname, data_series):
        pname = pname.name
    if pname == 'cvodes':
        pname = 'CVODE'
    elif pname == 'radau2a':
        pname = 'Radau-IIA'
    return '\\texttt{{\\textbf{{{}}}}}'.format(pname)

color_wheel = ['b', 'r', 'g', 'k', 'y']

def finalize():
    ax = plt.gca()
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
     ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(tick_font_size)