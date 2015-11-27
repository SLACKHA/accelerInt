#! /usr/bin/env python2.7
import matplotlib
import numpy as np
matplotlib.use('agg')
import matplotlib.pyplot as plt
import re
import os

base = os.getcwd()
dir_list = sorted([os.path.join(base, name) for name in os.listdir(base)
            if os.path.isdir(os.path.join(base, name)) and
            os.path.isdir(os.path.join(base, name, 'output')) 
            and not 'USC' in name])

class data_series(object):
	def __init__(self, name, gpu=False, smem=False, threads=6):
		self.name = name
		self.threads=threads
		self.gpu=gpu
		self.smem = smem
		self.data = []
		self.sorted = False

	def __eq__(self, other):
		return self.name == other.name and 
				self.threads == other.threads and
				self.gpu == other.gpu and
				self.smem == other.smem

	def add_x_y(self, x, y):
		y_avg = np.mean(y)
		y_dev = np.stddev(y)
		self.data.append((x, y_avg, y_dev))

	def plot(self, plt, name_fn=None):
		if not self.sorted:
			self.data = sorted(self.data, key=lambda x:x[0])
			self.x, self.y, self.y_dev = zip(*self.data)
		markersize=7
		if self.gpu:
			markerfacecolor='None'
			markeredgewidth=1
		plt.loglog(self.x, self.y, y_err=self.y_dev, label=self.name if not name_fn else name_fn(self),
					linestyle='', markersize=markersize, markeredgewidth=markeredgewidth, 
					markerfacecolor=markerfacecolor)

data_re = re.compile(r'^Time: (\d+\.\d+e[+-]\d+)$')
def read_data(file_name):
	y = []
	for line in lines:
		match = data_re.search(line)
		if match:
			y.append(float(match.group(1)))
	return y


gpu_re = re.compile(r'gpu_(\d+)_')
cpu_re = re.compile(r'int_(\d+)_(\d+)_')
data = {}
for mechanism in dir_list:
	data[mechanism] = []
	for file_name in os.path.listdir(os.path.join(base, mechanism, 'output')):
		name = file_name[:file_name.index('-int')]
		gpu = 'gpu' in file_name
		smem = 'nosmem' not in file_name
		if gpu:
			num_cond = int(gpu_re.search(file_name).group(1))
			num_thread = None
		else:
			match = cpu_re.search(file_name)
			num_cond = int(match.group(1))
			num_thread = int(match.group(2))
		series = data_series(name, gpu, smem, num_thread)
		if not series in data[mechanism]:
			data[mechanism].append(series)
		series = next(s for x in data[mechanism] if s == series)
		series.add_x_y(num_cond, read_data(file_name))