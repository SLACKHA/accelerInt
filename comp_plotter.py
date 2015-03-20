#comp plotter
import numpy as np
import matplotlib.pyplot as plt
from os import listdir, getcwd
from os.path import isfile, join

def read_file(fname):
    data = None
    with open(fname) as thefile:
        lines = [line.strip() for line in thefile.readlines() if len(line.strip())]
    if len(lines):
        num_vars = len([x for x in lines[0].split(',') if len(x.strip())])
        data = np.zeros((len(lines), num_vars))
        for i, line in enumerate(lines):
            vals = [x.strip() for x in line.split(',')]
            vals = [x for x in vals if len(x)]
            for j in range(len(vals)):
                data[i, j] = vals[j]
    return data


onlyfiles = [ f for f in listdir(getcwd()) if isfile(join(getcwd(),f)) if f.endswith('log.txt') and not 'kry' in f and not 'reject' in f]
if 'cvodes-analytical-int-log.txt' in onlyfiles:
    key = 'cvodes-analytical-int-log.txt'
else:
    key = 'cvodes-int-log.txt'
data_dict = {}
fig = plt.figure()
plot = fig.add_subplot(1,1,1)
for thefile in onlyfiles:
    data = read_file(thefile)
    if data is not None:
        data_dict[thefile] = data
        plot.plot(data[:,0], data[:, 1], label = thefile[:thefile.index('log.txt')])

plot.legend()
plot.set_ylabel("Temperature(K)")
plot.set_xlabel("Time (s)")
plt.tight_layout()
plt.savefig('Temperature Comp.png')
plt.close()

for thefile in data_dict:
    fig = plt.figure()
    plot = fig.add_subplot(1,1,1)
    key_data = data_dict[key]
    fdata = data_dict[thefile]
    if thefile == key:
        continue
    else:
        T_diff = [100.0 * abs(a - b) / a for a, b in zip(key_data[:, 1], fdata[:, 1])]
    plot.plot(key_data[:, 0], T_diff)
    print(thefile)
    label = thefile[:thefile.index('log.txt')]
    label = "% Diff in Temperature (cvodes:{})".format(label)
    plot.set_ylabel(label)
    plot.set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig('{} vs {}.png'.format(key, thefile))
    plt.close()
