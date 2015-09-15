#! /usr/bin/env python2.7
import math
import subprocess
import os, glob, shutil
import errno
import sys
from argparse import ArgumentParser
import stat

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
    for filename in os.listdir('.'):
        if os.path.isfile(filename):
            st = os.stat(filename)
            mode = st.st_mode
            if mode & executable:
                if not any(b in filename for b in blacklist) \
                    and all(i in filename for i in inverse):
                    exes.append(filename)
    return exes

def run(force=False, pyjac='', blacklist=None, repeats=5):
    if blacklist is None:
        blacklist = []

    mechanism = glob.glob('*.cti')[0]
    with open('ics.txt', 'r') as file:
        ic_str = file.readline().strip()
    with open('num_cond.txt', 'r') as file:
        num_cond = int(file.readline().strip())

    home = os.getcwd()
    make_sure_path_exists(os.path.join(home, 'log'))
    threads = [6, 12]
    powers = [1]
    while powers[-1] < num_cond:
        powers.append(powers[-1] * 2)
    if powers[-1] != num_cond:
        powers.append(num_cond)

    #generate mechanisms
    cache_opt = [True, False]
    use_smem = [True, False]
    same_ics = [True, False]
    for opt in cache_opt:
        mech_dir = 'cpu_{}'.format('co' if cache_opt else 'nco')
        mech_dir = os.path.join(home, mech_dir)
        make_sure_path_exists(mech_dir)
        args = ['python2.7', os.path.join(pyjac, 'pyJac.py'), '-lc', '-i{}'.format(mechanism)]
        if not opt:
            args.append('-nco')
        args.append('-ic{}'.format(ic_str))
        args.append('-b{}'.format(mech_dir))
        subprocess.check_call(args)

        for smem in use_smem:
            gpu_mech_dir = 'gpu_{}_{}'.format('co' if cache_opt else 'nco', 'smem' if smem else 'nosmem')
            gpu_mech_dir = os.path.join(home, gpu_mech_dir)
            make_sure_path_exists(gpu_mech_dir)
            if opt:
                #copy the pickle from the cpu folder
                shutil.copy(os.path.join(mech_dir, 'optimized.pickle'), 
                    os.path.join(gpu_mech_dir, 'optimized.pickle'))
            args = ['python2.7', os.path.join(pyjac, 'pyJac.py'), '-lc', '-i{}'.format(mechanism)]
            if not opt:
                args.append('-nco')
            if not smem:
                args.append('-nosmem')
            args.append('-ic{}'.format(ic_str))
            args.append('-b{}'.format(mech_dir))
            subprocess.check_call(args)

    #now build and run
    for same in same_ics:
        for opt in cache_opt:
            mech_dir = 'cpu_{}'.format('co' if cache_opt else 'nco')
            mech_dir = os.path.join(home, mech_dir)
            args = ['scons', 'cpu', 'mech_dir={}'.format(mech_dir)]
            if same:
                args.append('SAME_IC=True')
                log_dir = 'cpu_{}_{}'.format('same' if same else 'diff',
                'co' if cache_opt else 'nco')
                subprocess.check_call(args + ['LOG_OUTPUT=True'])
                #get all executables
                run_me = get_executables(blacklist + ['gpu'])
                #run log
                for exe in run_me:
                    subprocess.check_call([exe, '1', '1'])
                #move log files
                make_sure_path_exists(log_dir)
                for f in glob.glob(os.path.join(home, 'log', '*.txt')):
                    try:
                        shutil.move(os.path.join(home, 'log', f), os.path.join(home, log_dir, f))
                    except shutil.Error, e:
                        pass

            #rebuild for performance
            subprocess.check_call(args)
            #run with repeats
            for exe in run_me:
                for thread in threads:
                    for cond in powers:
                        with open(exe + '_{}_{}_{}.txt'.format(cond, thread, 
                            'co' if cache_opt else 'nco'), 'a') as file:
                            for repeat in range(repeats):
                                subprocess.check_call([exe, str(thread), str(cond)], stdout=file)

            for smem in use_smem:
                gpu_mech_dir = 'gpu_{}_{}'.format('co' if cache_opt else 'nco', 'smem' if smem else 'nosmem')
                gpu_mech_dir = os.path.join(home, gpu_mech_dir)
                log_dir = 'gpu_{}_{}_{}'.format('same' if same else 'diff',
                'co' if cache_opt else 'nco', 'smem' if smem else 'nosmem')
                args = ['scons', 'gpu', 'mech_dir={}'.format(gpu_mech_dir)]
                if same:
                    args.append('SAME_IC=True')
                    subprocess.check_call(args)
                    #get all executables
                    run_me = get_executables(blacklist, ['gpu'])
                    #run log
                    for exe in run_me:
                        subprocess.check_call([exe, '1'])
                    #move log files
                    make_sure_path_exists(log_dir)
                    for f in glob.glob(os.path.join(home, 'log', '*.txt')):
                        try:
                            shutil.move(os.path.join(home, 'log', f), os.path.join(home, log_dir, f))
                        except shutil.Error, e:
                            pass
                #rebuild for performance
                subprocess.check_call(args)
                #run with repeats
                for exe in run_me:
                    for cond in powers:
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
                        default='exp4-int,exp4-int-gpu,exprb43-int-gpu',
                        help='The solvers to not run')
    args = parser.parse_args()

    home = os.getcwd()
    a_dir = os.path.join(os.getcwd(), args.base_dir)
    dir_list = sorted([os.path.join(a_dir, name) for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))])

    for d in dir_list:
        os.chdir(d)
        run(force=args.force, pyjac=args.pyjac_dir, blacklist=args.solver_blacklist.split(','))
        os.chdir(home)