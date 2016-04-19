#! /usr/bin/env python2.7
import matplotlib
import numpy as np
matplotlib.use('agg')
import matplotlib.pyplot as plt
import re
from optionloop import optionloop

with open('logfile', 'r') as file:
	lines = [l.strip() for l in file.readlines() if l.strip()]

class data_series(object):
	def __init__(self, pasr, opt, smem, solver):
		self.pasr = pasr
		self.opt = opt
		self.smem = smem
		self.solver = solver
		self.lang = 'c' if 'gpu' not in solver else 'cuda'
		self.xy = []
	def state_eq(self, pasr=None, opt=None, smem=None, solver=None, lang=None):
		if pasr is not None and self.pasr != pasr:
			return False
		if opt is not None and self.opt != opt:
			return False
		if smem is not None and self.smem != smem:
			return False
		if solver is not None and self.solver != solver:
			return False
		if lang is not None and self.lang != lang:
			return False
		return True

series_list = []
PaSR = None
opt = None
smem = None
timestep = None
for line in lines:
	if not line.strip():
		continue
	if 'PaSR ICs' in line:
		PaSR = True
		continue
	elif 'Same ICs' in line:
		PaSR = False
		continue
	match = re.search(r'cache_opt:\s*(\w+)', line)
	if match:
		opt = match.group(1) == 'True'
		continue
	match = re.search(r'shared_mem:\s*(\w+)', line)
	if match:
		smem = match.group(1) == 'True'
		continue
	match = re.search(r't_step=(\d+e(?:-)?\d+)', line)
	if match:
		timestep = float(match.group(1))
		continue
	match = re.search(r'log/([\w\d-]+)-log.bin', line)
	if match:
		solver = match.group(1)
		continue
	match = re.search(r'L2 \(max, mean\) = (\d+\.\d+e(?:[-])?\d+)', line)
	if match:
		yval = float(match.group(1))
		series = next((x for x in series_list if x.state_eq(PaSR, opt, smem, solver)), None)
		if series is None:
			series = data_series(PaSR, opt, smem, solver)
			series_list.append(series)
		series.xy.append((timestep, yval))
		continue

c_params = optionloop({'lang' : 'c', 
            'opt' : [True, False],
            'same_ics' : [False]}, lambda: False)
cuda_params = optionloop({'lang' : 'cuda', 
            'opt' : [True, False],
            'smem' : [True, False],
            'same_ics' : [False]}, lambda: False)
op = c_params + cuda_params
for state in op:
	lang = state['lang']
	opt = state['opt']
	smem = state['smem']
	pasr = not state['same_ics']

	data_list = [x for x in series_list if x.state_eq(lang=lang, opt=opt, smem=smem, pasr=pasr)]

	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)

	for series in data_list:
		x, y = zip(*sorted(series.xy, key=lambda x: x[0]))
		print series.solver[:series.solver.index('-int')]
		print x, y
		plt.loglog(x, y, label=series.solver[:series.solver.index('-int')],
					linestyle='', marker='o')

	plt.xlabel('$\Delta_t (s)$')
	plt.ylabel('$Weighted Error$')

	plt.legend(loc=0, fontsize=8)
	plt.savefig('{}_{}_{}_error.pdf'.format(
		lang, 'co' if opt else 'nco', 
		'smem' if smem else 'nosmem',
		'pasr' if pasr else 'same'))
	plt.close()