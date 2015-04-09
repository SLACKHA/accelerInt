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
import timing_plotter

MAX_BLOCKS_PER_SM = 8
BLOCK_LIST = [4, 6, 8]
THREAD_LIST = [32, 64, 128]
NUM_ODES = 4096

class jac_params:
    def __init__(self, mech_name, therm_name, optimize_cache, inital_state, num_blocks, \
        num_threads, no_shared, L1_Preferred):
        self.lang = 'cuda'
        self.mech_name = mech_name
        self.therm_name = therm_name
        self.optimize_cache = optimize_cache
        self.inital_state = inital_state
        self.num_blocks = num_blocks
        self.num_threads = num_threads
        self.no_shared = no_shared
        self.L1_Preferred = L1_Preferred

def create_copy_and_run(jparam, mechanism_src, src, exe, file_name_out):
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
    args.append('-ic={}'.format(jparam.inital_state))
    args.append('-nb={}'.format(jparam.num_blocks))
    args.append('-nt={}'.format(jparam.num_threads))
    if jparam.no_shared:
        args.append('-nosmem')
    if not jparam.L1_Preferred:
        args.append('-pshare')
    subprocess.call(args)
    #copy
    files = os.listdir(mechanism_src)
    for file_name in files:
        full_file_name = os.path.join(mechanism_src, file_name)
        out_file_name = os.path.join(src, file_name)
        if os.path.isfile(full_file_name):
            shutil.copyfile(full_file_name, out_file_name)
    files = os.listdir(os.path.join(mechanism_src, 'jacobs'))
    for file_name in files:
        full_file_name = os.path.join(mechanism_src, 'jacobs', file_name)
        out_file_name = os.path.join(src, file_name)
        if os.path.isfile(full_file_name):
            shutil.copyfile(full_file_name, out_file_name)
    #make
    subprocess.call(['make', exe, '-j12'])
    #run
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
parser.add_argument('-ic', '--initial-conditions',
                    type=str,
                    dest='initial_conditions',
                    required=True,
                    help = 'A comma separated list of initial initial conditions to set in the set_same_initial_conditions method. \
                            Expected Form: T,P,Species1=...,Species2=...,...\n\
                            Temperature in K\n\
                            Pressure in Atm\n\
                            Species in moles')
args = parser.parse_args()

#make sure the output directory exists
if not os.path.isdir('cuda_parameter_study'):
    os.mkdir('cuda_parameter_study')

mech_path = os.path.abspath(args.directory)
mechanism = os.path.abspath(os.path.join(mech_path, args.input))
have_thermo = args.thermo != ''
thermo = os.path.abspath(os.path.join(mech_path, args.thermo)) if have_thermo else None
mechanism_src = os.path.abspath('out')
src = os.path.abspath('src')

options = ['-nco', '-pshare', '-nosmem']

params = jac_params(mechanism, thermo, False, args.initial_conditions, MAX_BLOCKS_PER_SM, THREAD_LIST[0], True, True)
for block in BLOCK_LIST:
    params.num_blocks = block
    #do the base ones
    for thread in THREAD_LIST:
        params.num_threads = thread
        outname = os.path.abspath(os.path.join('cuda_parameter_study', '{}_base_{}_{}.txt'.format(args.solver, block, thread)))
        create_copy_and_run(params, mechanism_src, src, args.solver, outname)

    #next turn on cache optimizations
    params.optimize_cache = True
    for thread in THREAD_LIST:
        params.num_threads = thread
        outname = os.path.abspath(os.path.join('cuda_parameter_study', '{}_cache_opt_{}_{}.txt'.format(args.solver, block, thread)))
        create_copy_and_run(params, mechanism_src, src, args.solver, outname)

    #next turn on shared memory
    params.no_shared=False
    for thread in THREAD_LIST:
        params.num_threads = thread
        outname = os.path.abspath(os.path.join('cuda_parameter_study', '{}_cache_opt_smem_{}_{}.txt'.format(args.solver, block, thread)))
        create_copy_and_run(params, mechanism_src, src, args.solver, outname)

    #finally prefer shared
    params.L1_Preferred = False
    for thread in THREAD_LIST:
        params.num_threads = thread
        outname = os.path.abspath(os.path.join('cuda_parameter_study', '{}_cache_opt_smem_pref_{}_{}.txt'.format(args.solver, block, thread)))
        create_copy_and_run(params, mechanism_src, src, args.solver, outname)

#and finally plot
timing_plotter.time_plotter('CUDA Parameter Study - {} ODES'.format(NUM_ODES), 'cuda_parameter_study', os.getcwd(), True)