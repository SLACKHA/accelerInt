#! /usr/bin/env python2.7
import matplotlib
import numpy as np
matplotlib.use('agg')
import matplotlib.pyplot as plt
import re

with open('logfile', 'r') as file:
	lines = [l.strip() for l in file.readlines() if l.strip()]

data = {}
for i in range(len(lines)):
	line = lines[i]
	match = re.search('t=(\d\.?\d*(?:e-)?\d*)', line)
	if match:
		tstep = float(match.group(1))
		i += 1
		while True and i < len(lines):
			match = re.search('log/(\w+)-int-log', lines[i])
			if not match:
				break
			solver = match.group(1)
			if not solver in data:
				data[solver] = []
			max_err, norm_err = [float(x) for x in lines[i + 1].split()]
			data[solver].append((tstep, max_err, norm_err))
			i += 2

for solver in data:
	data[solver] = sorted(data[solver], key=lambda x: x[0])
	data[solver] = zip(*data[solver])
	data[solver] = [np.array(x) for x in data[solver]]
	print solver, np.min(data[solver][0]), np.min(data[solver][2])
	plt.loglog(1. / data[solver][0], data[solver][2], label=solver)

plt.legend(loc=0, fontsize=8)
plt.savefig('error.pdf')
plt.close()