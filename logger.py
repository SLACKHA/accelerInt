#! /usr/bin/env python2.7
import math
import subprocess
import os, glob
import sys
import numpy as np
import re

NUM_ODES = 1000
N_THREADS = 12

def check_reorder(cache_opt, arr, order):
    if cache_opt:
        sub_arr = arr[:, :3]
        temp = arr[:, 3:]
        return np.hstack(sub_arr, temp[:, order])
    else:
        return arr


#force remake
subprocess.check_call(['make', '-j24', 'DEBUG=FALSE', 'FAST_MATH=FALSE', 'IGN=TRUE', 'PRINT=FALSE', 'LOG_OUTPUT=TRUE', 'SHUFFLE=TRUE', 'LARGE_STEP=TRUE'])

NVAR = None
with open('src/mechanism.h') as file:
    for line in file.readlines():
        match = re.search('^\s*#define\s+NN\s+(\d+)', line)
        if match:
            NVAR = int(match.group(1)) + 2 #t, T, P, num_spec

assert NVAR is not None


GPU_CACHE_OPT = False
with open('src/mechanism.cu') as file:
    start = False
    for line in file.readlines():
        if 'apply_mask' in line:
            start = True
        elif 'temp' in line and not start:
            GPU_CACHE_OPT = True
            break
        elif start and '}' in line:
            break

CPU_CACHE_OPT = False
with open('src/mechanism.c') as file:
    start = False
    for line in file.readlines():
        if 'apply_mask' in line:
            start = True
        elif 'temp' in line and not start:
            CPU_CACHE_OPT = True
            break
        elif start and '}' in line:
            break

spec_ordering = None
if GPU_CACHE_OPT or CPU_CACHE_OPT:
    #need the reorderings
    with open('src/optimized.pickle', 'rb') as file:
        dummy = pickle.load(file)
        dummy = pickle.load(file)
        dummy = pickle.load(file)
        dummy = pickle.load(file)
        dummy = pickle.load(file)
        dummy = pickle.load(file)
        spec_ordering = pickle.load(file)
        dummy = pickle.load(file)

all_exes = []
for file in glob.glob('*-int*'):
    all_exes.append(file)


skip = []
for exe in all_exes:
    if 'exp' in exe:
    #if 'gpu' in exe and 'exp' in exe:
        skip.append(exe)
    print exe
    if 'gpu' in exe:
        subprocess.check_call([os.path.join(os.getcwd(), exe), str(NUM_ODES)])
    else:
        subprocess.check_call([os.path.join(os.getcwd(), exe), str(N_THREADS), str(NUM_ODES)])

files = [f for f in os.listdir('log') if os.path.isfile(os.path.join('log', f)) and f.endswith('.bin') and not any(s in f for s in skip)]

key_file = [f for f in files if 'cvodes' in f and 'analytical' in f]
if not key_file:
    print "cvodes-analytic not found!"
    sys.exit(-1)
key_file = key_file[0]
print key_file

the_data = {}
with open('error_results.txt', 'w') as outfile:
    for file in files:
        array = np.fromfile(os.path.join('log', file), dtype='float64')
        array = array.reshape((array.shape[0] / (NVAR * NUM_ODES), (NVAR * NUM_ODES)))
        the_data[file] = array

    for file in files:
        print file
        outfile.write(file + '\n')
        max_err_ode = -1
        max_zero_err_ode = -1
        for ode in range(NUM_ODES):
            start_ind = NVAR * ode
            key_arr = the_data[key_file][:, start_ind : start_ind + NVAR]

            key_arr = check_reorder(CPU_CACHE_OPT, key_arr, spec_ordering)

            if file == key_file:
                continue
            data_arr = the_data[file][:, start_ind : start_ind + NVAR]
            data_arr = check_reorder(GPU_CACHE_OPT if 'gpu' in file else CPU_CACHE_OPT, data_arr, spec_ordering)

            #now compare column by column and get max err
            max_err = 0
            max_zero_err = 0
            max_err_col = 'N/A'
            max_zero_err_col = 'N/A'
            #skip time col
            for col in range(1, NVAR):
                for row in range(key_arr.shape[0]):
                    zero_err = False
                    if np.abs(key_arr[row, col]) < 1e-15:
                        err = np.abs(key_arr[row, col] - data_arr[row, col])
                        zero_err = True
                    else:
                        err = 100.0 * np.abs(key_arr[row, col] - data_arr[row, col]) / key_arr[row, col]
                    if zero_err:
                        if err > max_zero_err:
                            max_zero_err = err
                            max_zero_err_ode = ode
                            if col == 1:
                                max_zero_err_col = 'T'
                            elif col == 2:
                                max_zero_err_col = 'P'
                            else:
                                max_zero_err_col = 'Y_{}'.format(col - 3)
                    else:
                        if err > max_err:
                            max_err = err
                            max_err_ode = ode
                            if col == 1:
                                max_err_col = 'T'
                            elif col == 2:
                                max_err_col = 'P'
                            else:
                                max_err_col = 'Y_{}'.format(col - 3)
        outfile.write("max zero err: {} in col {} for ODE {}\n".format(max_zero_err, max_zero_err_col, max_zero_err_ode + 1))
        outfile.write("max err: {} in col {} for ODE {}\n".format(max_err, max_err_col, max_err_ode + 1))
