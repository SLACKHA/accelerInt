#! /usr/bin/env python

import data_parser as parser
import plot_styles as ps

data = parser.get_series()

from optionloop import OptionLoop as op
import numpy as np
import matplotlib.pyplot as plt

oploop = op({'dt' : [1e-6, 1e-4],
            'gpu' : [True, False],
            'mech' : data.keys()})

smem = True
normalize=True
CPU_CORE_COUNT = 40.
num_odes = 1e7
steps = 1e3
num_odes *= steps

#guarentee the same colors between plots
dt_list = set()
for mech in data:
    dt_list = dt_list.union([s.dt for s in data[mech]])
color_dict = {}
color_list = iter(ps.color_wheel)
for dt in dt_list:
    color_dict[dt] = color_list.next()

gpu_marker = 's'
cpu_marker = 'o'

with open('thresholds.txt', 'w'):
    pass


slopes = {}
for state in oploop:
    dt = state['dt']
    gpu = state['gpu']
    mech = state['mech']

    series = [s for s in data[mech] if 
                s.gpu == gpu and
                s.dt == dt and
                (not s.gpu or (s.gpu and s.smem == smem))
                and s.finite_difference == False
                and s.cache_opt == False]
    series = sorted(series, key=lambda x: x.name)
    print mech, 'gpu' if gpu else 'cpu'

    #get cost per ode
    for i, s in enumerate(series):
        if normalize:
            for i in range(len(s.data)):
                s.data[i] = (s.data[i][0], s.data[i][1] / s.data[i][0], s.data[i][2] / s.data[i][0])

        s.sort()

    to_calc = next((s for s in series if s.name == "radau2a" and s.gpu), None)
    if to_calc is None:
        to_calc = next((s for s in series if s.name == "cvodes"), None)

    last_value = s.y[-1] * 1.1

    #draw threshold
    
    x_index = np.where(to_calc.y <= last_value)[0][0]
    x_t = to_calc.x[x_index]

    with open('thresholds.txt', 'a') as file:
        file.write('{}\t{}\t{:.0e}\t{}\n'.format(mech, 'gpu' if gpu else 'cpu', dt, x_t))

    #(s * unit)/ODE
    sec_per_ode = np.mean(to_calc.y[x_index:])
    if not to_calc.gpu:
        sec_per_ode *= CPU_CORE_COUNT

    if mech not in slopes:
        slopes[mech] = {}
    if dt not in slopes[mech]:
        slopes[mech][dt] = {}
    if gpu:
        if 'gpu' not in slopes[mech][dt]:
            slopes[mech][dt]['gpu'] = sec_per_ode
        else:
            raise Exception
    else:
        if 'cpu' not in slopes[mech][dt]:
            slopes[mech][dt]['cpu'] = sec_per_ode
        else:
            raise Exception

for mech in slopes:
    for dt in slopes[mech]:
        print mech, dt, slopes[mech][dt]['cpu'] / slopes[mech][dt]['gpu']