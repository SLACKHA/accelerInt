#! /usr/bin/env python2.7

#parameter_study.py
#designed to run timing tests for a given CUDA solver for a variety of different conditions
#to determine optimal running conditions

#directory structure
#parent/
#   this_folder/
#           parameter_study.py
#           makefile
#           timing_plotter.py
#   create_jacobian/create_jacobian.py
#   mechs/mechanism_files

import sys, os
lib_path = os.path.abspath(os.path.join('..', 'create_jacobian'))
import subprocess
from argparse import ArgumentParser
import shutil
import matplotlib.pyplot as plt
from timing_plotter import parse_gpu
import re

MAX_BLOCKS_PER_SM = 8
BLOCK_LIST = [4, 6, 8]
THREAD_LIST = [32, 64, 128, 256]
NUM_ODES = 65536

class jac_params:
    def __init__(self, mech_name, therm_name, optimize_cache, num_blocks, \
        num_threads, no_shared, L1_Preferred):
        self.lang = 'cuda'
        self.mech_name = mech_name
        self.therm_name = therm_name
        self.optimize_cache = optimize_cache
        self.num_blocks = num_blocks
        self.num_threads = num_threads
        self.no_shared = no_shared
        self.L1_Preferred = L1_Preferred

def create_copy_and_run(jparam, mechanism_src, src, exe, file_name_out, check):
    if os.path.isfile(file_name_out):
        return
    #create
    args = [os.path.join(lib_path, 'create_jacobian.py')]
    args.append('-l={}'.format(jparam.lang))
    args.append('-i={}'.format(jparam.mech_name))
    if jparam.therm_name != '':
        args.append('-t={}'.format(jparam.therm_name))
    if not jparam.optimize_cache:
        args.append('-nco')
    args.append('-nb={}'.format(jparam.num_blocks))
    args.append('-nt={}'.format(jparam.num_threads))
    if jparam.no_shared:
        args.append('-nosmem')
    if not jparam.L1_Preferred:
        args.append('-pshare')
    subprocess.call(args)
    #copy
    subprocess.call('cp -r {} {}'.format(os.path.join(mechanism_src, '*'), src), shell = True)
    #make and test rates
    devnull = open('/dev/null', 'w')
    subprocess.call(['make', 'gpuratestest', 'SAME_IC=FALSE', '-j12'], stdout = devnull)
    subprocess.call([os.path.join(os.getcwd(), 'gpuratestest')])
    file = open('ratescomp_output', 'w')
    subprocess.call([os.path.join(lib_path, 'ratescomp.py'), '-n=' + os.path.join(os.getcwd(), 'rates_data.txt'), '-b=' + os.path.join(os.getcwd(),'rates_and_jacob/baseline_new_withspec.txt')], stdout=file)
    file.flush()
    file.close()
    with open('ratescomp_output', 'r') as file:
        lines = [line.strip() for line in file.readlines()]
        for line in lines:
            match = re.search('(\d+\.\d+(?:e-\d+)?)%', line)
            if match:
                perc = float(match.groups()[0])
                if perc > 0.0003:
                    raise Exception("Invalid Jacobian/Rates detected!")

    #do actual parameter run
    subprocess.call(['make', exe, 'SAME_IC=FALSE', '-j12'], stdout = devnull)
    #run
    devnull.close()
    file = open(file_name_out, 'w')
    subprocess.call([os.path.join(os.getcwd(), exe), str(NUM_ODES)], stdout=file)
    file.flush()
    file.close()

parser = ArgumentParser(description='Runs a parameter study on the given mechanism for various CUDA optimizations')
parser.add_argument('-dir', '--directory',
                    type=str,
                    default=os.path.abspath(os.path.join('..', 'mechs')),
                    required=False,
                    help = 'The directory the mechanism files are stored in')
parser.add_argument('-i', '--input',
                    type=str,
                    required=True,
                    help = 'The mechanism to test on.')
parser.add_argument('-t', '--thermo',
                    type=str,
                    default = '',
                    required=False,
                    help = 'The (optional) thermo-database file.')
parser.add_argument('-s', '--solver',
                    type=str,
                    default='radau2a-int-gpu',
                    required=False,
                    help = 'The solver to test')
parser.add_argument('-check', '--checkjacobian',
                    default=False,
                    required=False,
                    help = 'Check the output of the Jacobian for each case')
args = parser.parse_args()

out_dir = os.path.abspath('cuda_parameter_study')
#make sure the output directory exists
if not os.path.isdir(out_dir):
    os.mkdir(out_dir)
mech_path = os.path.abspath(args.directory)
mechanism = os.path.abspath(os.path.join(mech_path, args.input))
have_thermo = args.thermo != ''
thermo = os.path.abspath(os.path.join(mech_path, args.thermo)) if have_thermo else None
mechanism_src = os.path.abspath('out')
src = os.path.abspath('src')

options = ['-nco', '-pshare', '-nosmem']

for block in BLOCK_LIST:
    #reset params
    params = jac_params(mechanism, thermo, False, MAX_BLOCKS_PER_SM, THREAD_LIST[0], True, True)
    params.num_blocks = block
    #do the base ones
    for thread in THREAD_LIST:
        params.num_threads = thread
        outname = os.path.abspath(os.path.join(out_dir, '{}_base_{}_{}.txt'.format(args.solver, block, thread)))
        create_copy_and_run(params, mechanism_src, src, args.solver, outname, args.checkjacobian)

    #next turn on shared memory
    params.no_shared=False
    for thread in THREAD_LIST:
        params.num_threads = thread
        outname = os.path.abspath(os.path.join(out_dir, '{}_smem_{}_{}.txt'.format(args.solver, block, thread)))
        create_copy_and_run(params, mechanism_src, src, args.solver, outname, args.checkjacobian)

    #next turn on cache optimizations
    params.no_shared=True
    params.optimize_cache = True
    for thread in THREAD_LIST:
        params.num_threads = thread
        outname = os.path.abspath(os.path.join(out_dir, '{}_cache_opt_{}_{}.txt'.format(args.solver, block, thread)))
        create_copy_and_run(params, mechanism_src, src, args.solver, outname, args.checkjacobian)

    #next turn on shared memory
    params.no_shared=False
    for thread in THREAD_LIST:
        params.num_threads = thread
        outname = os.path.abspath(os.path.join(out_dir, '{}_cache_opt_smem_{}_{}.txt'.format(args.solver, block, thread)))
        create_copy_and_run(params, mechanism_src, src, args.solver, outname, args.checkjacobian)

    #finally prefer shared
    params.L1_Preferred = False
    for thread in THREAD_LIST:
        params.num_threads = thread
        outname = os.path.abspath(os.path.join(out_dir, '{}_cache_opt_smem_pref_{}_{}.txt'.format(args.solver, block, thread)))
        create_copy_and_run(params, mechanism_src, src, args.solver, outname, args.checkjacobian)

#and finally plot
results = []
files = os.listdir(out_dir)
for file in files:
    val = parse_gpu(os.path.join(out_dir, file))
    if val is not None:
        odes, block, time = val
        if not odes == NUM_ODES:
            print "invalid file: {}".format(file)
        results.append((file, block, time))
results = sorted(results, key=lambda x: x[2])
for r in results:
    print r