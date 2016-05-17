#a single consolidated place to import
#such that all figures have identical styling (when possible)

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

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

marker_dict = {'cvodes' : ('.', False),
'radau2a' : ('>', True),
'exp4' : ('o', True),
'exprb43' : ('s', True)
}

def pretty_names(name):
    if name == 'cvodes':
        return 'CVODE'
    elif name == 'radau2a':
        return 'Radau-IIA'
    return name

color_wheel = ['b', 'r', 'g', 'k', 'y']