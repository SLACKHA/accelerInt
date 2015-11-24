#! /usr/bin/env python2.7
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import re

with open('logfile', 'r') as file:
	lines = [l.strip() for l in file.readlines() if l.strip()]

xdata = []
data = {}
for i in range(len(lines)):
	line = lines[i]
	match = re.search('t=(\d(?:e|\.)-?\d+)', line)
	if match:
		xdata.append(float(match.group(1)))
		i += 1
		while True and i < len(lines):
			match = re.search('log/(\w+)-int-log', lines[i])
			if not match:
				break
			solver = match.group(1)
			if not solver in data:
				data[solver] = []
			max_err, norm_err = [float(x) for x in lines[i + 1].split()]
			data[solver].append((max_err, norm_err))
			i += 2

for solver in data:
	data[solver] = zip(*data[solver])
	print xdata, data[solver][1]
	plt.loglog(xdata, data[solver][1], label=solver)

plt.legend(loc=0, fontsize=8)
plt.savefig('error.pdf')