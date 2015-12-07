#! /usr/bin/env python2.7
import matplotlib
import numpy as np
matplotlib.use('agg')
import matplotlib.pyplot as plt
import re
import os

class data_series(object):
	def __init__(self, name, gpu=False, smem=False, threads=6, dt=1e-6):
		self.name = name
		self.threads=threads
		self.gpu=gpu
		self.smem = smem
		self.dt = dt
		self.data = []
		self.sorted = False

		self.kwargs = {}
		self.kwargs['linestyle']=''
		self.kwargs['marker']='v'
		self.kwargs['markersize']=7
		self.kwargs['markeredgewidth']=0
		self.kwargs['markeredgecolor']='k'
		self.kwargs['markerfacecolor']='k'

	def __str__(self):
		return 'solver: {}\tthreads={}\tgpu={}\tsmem={}\tdt={}'.format(
			self.name, self.threads, self.gpu, self.smem, self.dt)

	def __repr__(self):
		return self.__str__()

	def __eq__(self, other):
		return self.name == other.name and \
				self.threads == other.threads and \
				self.gpu == other.gpu and \
				self.smem == other.smem and \
				self.dt == other.dt

	def set_clear_marker(self, marker=None, color=None, size=None):
		self.kwargs['markerfacecolor']='None'
		self.kwargs['markeredgewidth']=1
		if marker is not None:
			self.kwargs['marker'] = marker
		if color is not None:
			self.kwargs['markeredgecolor'] = color
		if size is not None:
			self.kwargs['markersize'] = size

	def set_marker(self, marker=None, color=None, size=None):
		if marker is not None:
			self.kwargs['marker'] = marker
		if color is not None:
			self.kwargs['markerfacecolor'] = color
			self.kwargs['markeredgecolor'] = color
			self.kwargs['markeredgewidth'] = 1
		if size is not None:
			self.kwargs['markersize'] = size

	def add_x_y(self, x, y):
		if y:
			y_avg = np.mean(y)
			y_dev = np.std(y)
			self.data.append((x, y_avg, y_dev))

	def plot(self, plt, name_fn=None, show_dev=False):
		if not self.sorted:
			self.data = sorted(self.data, key=lambda x:x[0])
			self.x, self.y, self.z = zip(*self.data)
			self.x = np.array(self.x)
			self.y = np.array(self.y)
			self.z = np.array(self.z)
		args = [self.x, self.y]

		self.kwargs['label'] = self.name if name_fn is None else name_fn(self)

		if show_dev:
			args += [self.z, None, self.kwargs['linestyle']]
			self.kwargs['ecolor'] = self.kwargs['markerfacecolor'] \
					if self.kwargs['markerfacecolor'] != 'None' else \
					self.kwargs['markeredgecolor']
			plt.errorbar(*args, **self.kwargs)
		else:
			args += [self.kwargs['linestyle']]
			plt.plot(*args, **self.kwargs)

def read_data(file_name):
	data_re = re.compile(r'^Time: (\d+\.\d+e[+-]\d+) sec$')
	y = []
	with open(file_name, 'r') as file:
		lines = [line.strip() for line in file.readlines()]
	for line in lines:
		match = data_re.search(line)
		if match:
			y.append(float(match.group(1)))
	return y

def get_series():
	base = os.getcwd()
	dir_list = sorted([name for name in os.listdir(base)
	            if os.path.isdir(os.path.join(base, name)) and
	            os.path.isdir(os.path.join(base, name, 'output')) 
	            and not 'USC' in name])

	gpu_re = re.compile(r'gpu_(\d+)_.+(\d\.\d+e[+-]\d+)\.txt')
	cpu_re = re.compile(r'int_(\d+)_(\d+)_.+(\d\.\d+e[+-]\d+)\.txt')
	data = {}
	for mechanism in dir_list:
		data[mechanism] = []
		thedir = os.path.join(base, mechanism, 'output')
		for file_name in os.listdir(thedir):
			if not file_name.endswith('.txt'): continue
			open_name = os.path.join(thedir, file_name)
			name = file_name[:file_name.index('-int')]
			gpu = 'gpu' in file_name
			smem = 'nosmem' not in file_name
			if gpu:
				match =  gpu_re.search(file_name)
				num_cond = int(match.group(1))
				num_thread = None
				dt =  float(match.group(2))
			else:
				match = cpu_re.search(file_name)
				num_cond = int(match.group(1))
				num_thread = int(match.group(2))
				dt =  float(match.group(3))
			series = data_series(name, gpu, smem, num_thread, dt)
			if not series in data[mechanism]:
				data[mechanism].append(series)
			series = next(s for s in data[mechanism] if s == series)
			series.add_x_y(num_cond, read_data(open_name))
	return data

if __name__ == '__main__':
	get_series()