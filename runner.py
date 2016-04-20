#! /usr/bin/env python2.7
import math
import subprocess
import os, glob, shutil
import errno
import sys
from argparse import ArgumentParser
import stat
import cantera as ct
import numpy as np
import multiprocessing
import re
from pyjac import create_jacobian
from optionLoop import optionloop
from collections import defaultdict

scons = subprocess.check_output('which scons', shell=True).strip()

def check_dir(dir, force):
    old_files = [file for file in os.listdir(dir) if '.timing' in file and os.path.isfile(os.path.join(dir, file))]
    if len(old_files) and not force:
        raise Exception("Old data found in /{}/... stopping".format(dir))

def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

def get_executables(blacklist, inverse=None):
    exes = []
    if inverse is None:
        inverse = []
    executable = stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH
    for filename in os.listdir(os.getcwd()):
        if os.path.isfile(filename) and \
            not filename.endswith('.py'):
            st = os.stat(filename)
            mode = st.st_mode
            if mode & executable:
                if not any(b in filename for b in blacklist) \
                    and all(i in filename for i in inverse):
                    exes.append(filename)
    return exes

def get_powers(num_cond):
    powers = [1]
    while powers[-1] * 2 < num_cond:
        powers.append(powers[-1] * 2)
    if powers[-1] != num_cond:
        powers.append(num_cond)
    return powers

def get_diff_ics_cond(thedir, mechanism):
    gas = ct.Solution(mechanism)
    data = np.fromfile(os.path.join(thedir, 'data.bin'), dtype='float64')
    num_c = data.shape[0] / float(gas.n_species + 3)
    assert int(num_c) == num_c
    return int(num_c)

def check_file(filename):
    count = 0
    if not os.path.isfile(filename):
        return count
    with open(filename, 'r') as file:
        for line in file.readlines():
            if re.search(r'^Time: \d\.\d+e[+-]\d+ sec$', line):
                count += 1
    return count

def run(thedir, blacklist=[], force=False,
        repeats=5, num_cond=131072,
        threads=[6, 12], langs=['c', 'cuda'],
        atol=1e-10,
        rtol=1e-7):
    jthread = str(multiprocessing.cpu_count())

    make_sure_path_exists(os.path.join(thedir, 'output'))

    try:
        mechanism = os.path.join(thedir, glob.glob(os.path.join(thedir, '*.cti'))[0])
        with open(os.path.join(thedir, 'ics.txt'), 'r') as file:
            ic_str = file.readline().strip()
    except:
        print "Mechanism file not found in {}, skipping...".format(thedir)
        return

    home = os.getcwd()
    same_powers = get_powers(num_cond)
    diff_powers = get_powers(get_diff_ics_cond(thedir, mechanism))

    #copy the datafile
    shutil.copy(os.path.join(thedir, 'data.bin'),
                os.path.join(home, 'ign_data.bin'))

    opt_list = [True, False]
    smem_list = [True, False]
    t_list = [1e-6, 1e-4]
    ics_list = [False]
    fd_list = [True, False]

    c_params = optionloop({'lang' : 'c', 
                'opt' : opt_list,
                't_step' : t_list,
                'same_ics' : ics_list,
                'FD' : fd_list}, lambda: False)
    cuda_params = optionloop({'lang' : 'cuda', 
                'opt' : opt_list,
                't_step' : t_list,
                'smem' : smem_list,
                'same_ics' : ics_list,
                'FD' : fd_list}, lambda: False)
    op = c_params + cuda_params
    for state in op:
        opt = state['opt']
        smem = state['smem']
        t_step = state['t_step']
        same = state['same_ics']
        FD = state['FD']
        thepow = same_powers if same else diff_powers

        #custom rules so evaluation doesn't take so damn long
        if opt and t_step == 1e-4:
            continue
        if not smem and t_step == 1e-4:
            continue

        #generate mechanisms
        if 'c' in langs:
            mech_dir = 'cpu_{}'.format('co' if opt else 'nco')
            mech_dir = os.path.join(thedir, mech_dir) + os.path.sep
            make_sure_path_exists(mech_dir)
            create_jacobian(lang='c', mech_name=mechanism,
                            optimize_cache=opt, initial_state=ic_str,
                            build_path=mech_dir, multi_thread=int(jthread))

        if 'cuda' in langs:
            gpu_mech_dir = 'gpu_{}_{}'.format('co' if opt else 'nco', 'smem' if smem else 'nosmem')
            gpu_mech_dir = os.path.join(thedir, gpu_mech_dir)
            make_sure_path_exists(gpu_mech_dir)
            if opt and 'c' in langs:
                #copy the pickle from the cpu folder
                shutil.copy(os.path.join(mech_dir, 'optimized.pickle'), 
                    os.path.join(gpu_mech_dir, 'optimized.pickle'))
            create_jacobian(lang='cuda', mech_name=mechanism,
                        optimize_cache=opt, initial_state=ic_str,
                        build_path=gpu_mech_dir, no_shared=not smem, multi_thread=int(jthread))

        #now build and run
        args = ['-j', jthread, 'DEBUG=False', 'FAST_MATH=False',
             'LOG_OUTPUT=False','SHUFFLE=False', 'LOG_END_ONLY=False',
             'PRINT=False',
             't_step={}'.format(t_step),
             't_end={}'.format(t_step),
             'DIVERGENCE_WARPS=0', 'CV_HMAX=0', 'CV_MAX_STEPS=-1',
             'ATOL={:.0e}'.format(atol),
             'RTOL={:.0e}'.format(rtol),
             'FINITE_DIFFERENCE={}'.format(FD)]
        args.append('SAME_IC={}'.format(same))

        #run with repeats
        if 'c' in langs:
            run_me = get_executables(blacklist + ['gpu'], inverse=['int'])
            subprocess.check_call([scons, 'cpu'] + args + ['mechanism_dir={}'.format(mech_dir)])
            for exe in run_me:
                for thread in threads:
                    for cond in thepow:
                        filename = os.path.join(thedir, 'output',
                            exe + '_{}_{}_{}_{}_{}_{:e}.txt'.format(cond,
                            thread, 'co' if opt else 'nco',
                            'sameic' if same else 'psric', 'FD' if FD else 'AJ', t_step))
                        my_repeats = repeats - check_file(filename)
                        with open(filename, 'a') as file:
                            for repeat in range(my_repeats):
                                subprocess.check_call([os.path.join(home, exe), str(thread), str(cond)], stdout=file)

        if 'cuda' in langs:
            #run with repeats
            subprocess.check_call([scons, 'gpu'] + args + ['mechanism_dir={}'.format(gpu_mech_dir)])
            run_me = get_executables(blacklist, inverse=['int-gpu'])
            for exe in run_me:
                for cond in thepow:
                    filename = os.path.join(thedir, 'output',
                        exe + '_{}_{}_{}_{}_{}_{:e}.txt'.format(cond,
                        'co' if opt else 'nco', 'smem' if smem else 'nosmem',
                        'sameic' if same else 'psric', 'FD' if FD else 'AJ', t_step))
                    my_repeats = repeats - check_file(filename)
                    with open(filename, 'a') as file:
                        for repeat in range(my_repeats):
                            subprocess.check_call([os.path.join(home, exe), str(cond)], stdout=file)


if __name__ == '__main__':
    parser = ArgumentParser(description='Runs timing runs for the various integrators')
    parser.add_argument('-f', '--force',
                        required=False,
                        default=False,
                        action='store_true',
                        help='Force reuse of past data files')
    parser.add_argument('-b', '--base_dir',
                        required=False,
                        default='performance',
                        help='The base directory containing a folder per mechanism')
    parser.add_argument('-s', '--solver_blacklist',
                        required=False,
                        default='rk78',
                        help='The solvers to not run')
    parser.add_argument('-nc', '--num_cond',
                        type=int,
                        required=False,
                        default=131072,
                        help='The number of conditions to run for the same ics')
    parser.add_argument('-r', '--repeats',
                        required=False,
                        type=int,
                        default=5,
                        help='The number of timing repeats to run')
    parser.add_argument('-nt', '--num_threads',
                        type=str,
                        required=True,
                        help='Comma separated list of # of threads to test with for CPU integrators')
    parser.add_argument('-l', '--langs',
                        type=str,
                        required=False,
                        default='c,cuda',
                        help='Comma separated list of languages to test.')
    parser.add_argument('-atol', '--absolute_tolerance',
                        required=False,
                        default=1e-10,
                        help='The absolute tolerance for the integrators')
    parser.add_argument('-rtol', '--relative_tolerance',
                        required=False,
                        default=1e-7,
                        help='The relative tolerance for the integrators')
    args = parser.parse_args()

    num_threads = [int(x) for x in args.num_threads.split(',')]
    home = os.getcwd()
    a_dir = os.path.join(os.getcwd(), args.base_dir)
    dir_list = sorted([os.path.join(a_dir, name) for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))])

    for d in dir_list:
        run(d, blacklist=[x.strip() for x in 
                    args.solver_blacklist.split(',') if x.strip()], 
            force=args.force,
            num_cond=args.num_cond,
            repeats=args.repeats,
            threads=num_threads,
            langs=[x.strip() for x in 
                    args.langs.split(',') if x.strip()],
            atol=args.absolute_tolerance,
            rtol=args.relative_tolerance)