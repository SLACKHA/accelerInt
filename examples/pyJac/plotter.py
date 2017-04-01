# imports
import matplotlib.pyplot as plt
import numpy as np
import os

# load log files
files = [os.path.join('log', file) for file in os.listdir('./log/')
         if file.endswith('.bin')]
files = [file for file in files if os.path.isfile(file)]

linestyles = ['-', '--', '-.']
colorwheel = ['r', 'g', 'b', 'k']

# load data
for i, file in enumerate(files):
    arr = np.fromfile(file)
    # reshape matrix (nsteps x 15)
    arr = np.reshape(arr, (-1, 15))

    # get the solver name
    label = file.split('-')[0]
    label = label[label.index('/') + 1:]
    label += '-GPU' if 'gpu' in file else ''

    # plot Temperature
    plt.plot(arr[:, 0], arr[:, 1],
             linestyle=linestyles[i % len(linestyles)],
             label=label, color=colorwheel[i % len(colorwheel)])

# make legend
plt.legend(loc=0)

# title and labels
plt.title('H2/CO Constant Pressure Ignition')
plt.xlabel('Time(s)')
plt.ylabel('Temerature (K)')

# and save fig
plt.savefig('h2ign.png', dpi=300, size=(5, 3))
