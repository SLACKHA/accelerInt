#comp plotter
import numpy as np
import matplotlib.pyplot as plt
from os import listdir, getcwd
from os.path import isfile, join

#T_Diff = 5 #k
t_Diff = 20 #timesteps
def get_filtered(x, y, offset = 0):
    new_x_data = [x[offset]]
    new_y_data = [y[offset]]
    old_x = offset
    old_y = y[offset]
    for i in range(offset, len(x)):
        if i > old_x + t_Diff:# or y[i] > old_y + T_Diff:
            new_x_data.append(x[i])
            new_y_data.append(y[i])
            old_x = i
            old_y = y[i]
    return new_x_data, new_y_data

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
keylabel = key[:key.index('-log.txt')]
data_dict = {}
fig = plt.figure()
plot = fig.add_subplot(1,1,1)
for thefile in onlyfiles:
    data = read_file(thefile)
    if data is not None:
        data_dict[thefile] = data
        x, y = get_filtered(data[:,0], data[:, 1], offset=10*onlyfiles.index(thefile))
        plot.plot(x, y, linestyle = '', marker = '.', markersize = 5, label = thefile[:thefile.index('-log.txt')])

plot.legend(fontsize=10, loc=0)
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
    label = thefile[:thefile.index('-log.txt')]
    ylabel = "% Diff in Temperature ({}:{})".format(keylabel, label)
    plot.set_ylabel(ylabel)
    plot.set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig('{} vs {}.png'.format(keylabel, label))
    plt.close()
