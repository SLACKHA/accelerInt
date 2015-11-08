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
        if os.path.isfile(filename):
            st = os.stat(filename)
            mode = st.st_mode
            if mode & executable:
                if not any(b in filename for b in blacklist) \
                    and all(i in filename for i in inverse):
                    exes.append(filename)
    return exes

def get_powers(num_cond):
    powers = [1]
    while powers[-1] < num_cond:
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

def run(thedir, run_me, force=False, pyjac='', repeats=5, num_cond=131072):
    jthread = str(multiprocessing.cpu_count())

    mechanism = os.path.join(thedir, glob.glob(os.path.join(thedir, '*.cti'))[0])
    with open(os.path.join(thedir, 'ics.txt'), 'r') as file:
        ic_str = file.readline().strip()

    home = os.getcwd()
    threads = [6, 12]
    same_powers = get_powers(num_cond)
    diff_powers = get_powers(get_diff_ics_cond(thedir, mechanism))

    #copy the datafile
    shutil.copy(os.path.join(thedir, 'data.bin'),
                os.path.join(home, 'ign_data.bin'))

    #generate mechanisms
    cache_opt = [False]#[True, False]
    use_smem = [True, False]
    same_ics = [True, False]
    for opt in cache_opt:
        mech_dir = 'cpu_{}'.format('co' if opt else 'nco')
        mech_dir = os.path.join(thedir, mech_dir) + os.path.sep
        make_sure_path_exists(mech_dir)
        args = ['python2.7', os.path.join(pyjac, 'pyJac.py'), '-l', 'c', '-i', mechanism]
        if not opt:
            args.append('-nco')
        args.extend(['-ic', ic_str])
        args.extend(['-b', mech_dir])
        subprocess.check_call(args)

        for smem in use_smem:
            gpu_mech_dir = 'gpu_{}_{}'.format('co' if opt else 'nco', 'smem' if smem else 'nosmem')
            gpu_mech_dir = os.path.join(thedir, gpu_mech_dir)
            make_sure_path_exists(gpu_mech_dir)
            if opt:
                #copy the pickle from the cpu folder
                shutil.copy(os.path.join(mech_dir, 'optimized.pickle'), 
                    os.path.join(gpu_mech_dir, 'optimized.pickle'))
            args = ['python2.7', os.path.join(pyjac, 'pyJac.py'), '-l', 'cuda', '-i', mechanism]
            if not opt:
                args.append('-nco')
            if not smem:
                args.append('-nosmem')
            args.extend(['-ic', ic_str])
            args.extend(['-b', mech_dir])
            subprocess.check_call(args)

    #now build and run
    for same in same_ics:
        thepow = same_powers if same else diff_powers
        for opt in cache_opt:
            mech_dir = 'cpu_{}'.format('co' if opt else 'nco')
            mech_dir = os.path.join(thedir, mech_dir) + os.path.sep
            args = ['scons', 'cpu', '-j', jthread, 'DEBUG=False', 'FAST_MATH=FALSE',
                 'LOG_OUTPUT=FALSE','SHUFFLE=FALSE',
                 'PRINT=FALSE', 'mechanism_dir={}'.format(mech_dir)]
            args.append('SAME_IC={}'.format(same))
            #rebuild for performance
            subprocess.check_call(args)
            #run with repeats
            for exe in run_me:
                for thread in threads:
                    for cond in thepow:
                        with open(exe + '_{}_{}_{}.txt'.format(cond, thread, 
                            'co' if opt else 'nco'), 'a') as file:
                            for repeat in range(repeats):
                                subprocess.check_call([exe, str(thread), str(cond)], stdout=file)

            for smem in use_smem:
                gpu_mech_dir = 'gpu_{}_{}'.format('co' if opt else 'nco', 'smem' if smem else 'nosmem')
                gpu_mech_dir = os.path.join(thedir, gpu_mech_dir)
                args = ['scons', 'gpu', '-j', jthread, 'DEBUG=False', 'FAST_MATH=FALSE',
                 'LOG_OUTPUT=FALSE','SHUFFLE=FALSE',
                 'PRINT=FALSE', 'mechanism_dir={}'.format(gpu_mech_dir)]
                args.append('SAME_IC={}'.format(same))
                subprocess.check_call(args)
                #run with repeats
                for exe in run_me:
                    for cond in thepow:
                        with open(exe + '{}_{}_{}.txt'.format(cond,
                            'co' if cache_opt else 'nco', 'smem' if smem else 'nosmem'), 'a') as file:
                            for repeat in range(repeats):
                                subprocess.check_call([exe, str(cond)], stdout=file)


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
    parser.add_argument('-p', '--pyjac_dir',
                        required=False,
                        default='~/pyJac/',
                        help='The base directory of pyJac')
    parser.add_argument('-s', '--solver_blacklist',
                        required=False,
                        default='',
                        help='The solvers to not run')
    parser.add_argument('-n', '--num_cond',
                        type=int,
                        required=False,
                        default=131072,
                        help='The number of conditions to run for the same ics')
    parser.add_argument('-r', '--repeats',
                        required=False,
                        type=int,
                        default=5,
                        help='The number of timing repeats to run')
    args = parser.parse_args()

    home = os.getcwd()
    a_dir = os.path.join(os.getcwd(), args.base_dir)
    dir_list = sorted([os.path.join(a_dir, name) for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))])

    run_me = get_executables(args.solver_blacklist.split(','))
    for d in dir_list:
        run(d, run_me, force=args.force, 
            pyjac=os.path.expanduser(args.pyjac_dir), 
            num_cond=args.num_cond,
            repeats=args.repeats)