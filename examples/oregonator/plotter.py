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
    # reshape to a 2001 x 3 matrix (time, y1, y2)
    arr = np.reshape(arr, (2001, 3))

    # get the solver name
    label = file.split('-')[0]
    label = label[label.index('/') + 1:]
    label += '-GPU' if 'gpu' in file else ''

    # plot y1
    plt.plot(arr[:, 0][::10], arr[:, 1][::10],
             linestyle=linestyles[i % len(linestyles)],
             label=label, color=colorwheel[i % len(colorwheel)])

# make legend
plt.legend(loc=0)

# title and labels
plt.title('Oregonator equation')
plt.xlabel('Time(s)')
plt.ylabel('$y_1$')

# and save fig
plt.savefig('oregonator.png', dpi=300, size=(5, 3))
