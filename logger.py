#! /usr/bin/env python2.7
import sys
import numpy as np
import shutil
from os.path import join as pjoin
from os import getcwd as cwd
from os.path import expanduser
from glob import glob
import re
import subprocess
import multiprocessing
from argparse import ArgumentParser
sys.path.append(pjoin(expanduser('~'),
                         'pyJac'))
import pyJac
np.set_printoptions(precision=15)

keyfile = 'cvodes-analytical-int-log.bin'

class Tee(object):
    def __init__(self, name, mode):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self
    def __enter__(self):
        return self
    def __exit__(self, type, value, traceback):
        sys.stdout = self.stdout
        self.file.close()
    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

def __check_error(builder, num_conditions, nvar, validator, outfile):
    globtxt = '*-gpu-log.bin' if builder == 'gpu' else '*-int-log.bin'
    key_arr = validator[:, 1:]
    for f in glob(pjoin('log', globtxt)):
        if keyfile in f:
            continue
        array = np.fromfile(f, dtype='float64')
        array = array.reshape((-1, 1 + num_conditions * nvar))

        outfile.write(f + '\n')
        data_arr = array[:, 1:]
        #now compare column by column and get max err
        max_err = 0
        max_zero_err = 0
        
        rnz, cnz = np.where(np.abs(key_arr) > 1e-30)
        err = 100. * np.abs((key_arr[rnz, cnz] - data_arr[rnz, cnz]) / key_arr[rnz, cnz])
        #print err
        max_err =  np.max(err)
        norm_err = np.linalg.norm(err)

        rz, cz = np.where(np.abs(key_arr) <= 1e-30)
        err = 100. * np.abs((key_arr[rz, cz] - data_arr[rz, cz]))
        max_zero_err =  np.max(err)
        zero_norm_err = np.linalg.norm(err)


        outfile.write("max non-zero err: {}\n"
                      "norm non-zero err: {}\n"
                      "max zero err: {}\n"
                      "norm zero err: {}\n".format(max_err, norm_err,
                        max_zero_err, zero_norm_err))

def __execute(builder, num_threads, num_conditions, logger):
    if builder == 'gpu':
        for exe in glob('*-gpu'):
            logger.write('\n' + exe + '\n')
            subprocess.check_call([pjoin(cwd(), exe), str(num_conditions)])
    else:
        for exe in glob('*-int'):
            logger.write('\n' + exe + '\n')
            subprocess.check_call([pjoin(cwd(), exe), str(num_threads), str(num_conditions)])

def __run_and_check(mech, thermo, initial_conditions, build_path,
        num_threads, num_conditions, test_data, nvar, outfile):
        #first compile and run cvodes to get the baseline
        pyJac.create_jacobian('c', mech, thermo,
            initial_state=initial_conditions,
            optimize_cache=False,
            build_path=build_path)
        arg_list = ['-j{}'.format(num_threads),
                'DEBUG=FALSE', 'FAST_MATH=FALSE', 'LOG_OUTPUT=TRUE', 'SHUFFLE=FALSE', 'PRINT=FALSE']
        if initial_conditions is not None:
            arg_list.append('SAME_IC=TRUE')
        else:
            arg_list.append('SAME_IC=FALSE')
        subprocess.check_call(['scons', 'cpu'] + arg_list)
        #run
        subprocess.check_call([pjoin(cwd(), 'cvodes-analytic-int'), str(num_threads), str(num_conditions)])
        #copy to saved data
        shutil.copy(pjoin(cwd(), 'log', keyfile),
                    pjoin(cwd(), 'log', 'valid.bin'))

        validator = np.fromfile(pjoin('log', 'valid.bin'), dtype='float64')
        validator = validator.reshape((-1, 1 + num_conditions * nvar))
        langs = ['c', 'cuda']
        builder = {'c':'cpu', 'cuda':'gpu'}
        opt = [True, False]
        smem = [True, False]
        #now check for various options
        for lang in langs:
            for cache_opt in opt:
                if lang == 'cuda':
                    for shared_mem in smem:
                        outfile.write('\ncache_opt: {}\n'
                                      'shared_mem: {}\n'.format(
                                        cache_opt, shared_mem))
                        pyJac.create_jacobian(lang, mech, thermo, 
                        initial_state=initial_conditions, 
                        optimize_cache=opt,
                        multi_thread=num_threads,
                        no_shared=not shared_mem,
                        build_path=build_path)

                        subprocess.check_call(['scons', builder[lang]] + arg_list)
                        __execute(builder[lang], num_threads, num_conditions, outfile)
                        __check_error(builder[lang], num_conditions, nvar, validator, outfile)

                else:
                    outfile.write('\ncache_opt: {}\n'.format(cache_opt))
                    pyJac.create_jacobian(lang, mech, thermo, 
                    initial_state=initial_conditions, 
                    optimize_cache=opt,
                    multi_thread=num_threads,
                    build_path=build_path)

                    subprocess.check_call(['scons', builder[lang]] + arg_list)
                    __execute(builder[lang], num_threads, num_conditions, outfile)
                    __check_error(builder[lang], num_conditions, nvar, validator, outfile)

def run_log(mech, thermo, initial_conditions, build_path,
        num_threads, num_conditions, test_data):
    with Tee(pjoin('log', 'log_results.txt'), 'w') as myTee:
        nvar = None
        #get num vars
        with open(pjoin(build_path, 'mechanism.h'), 'r') as file:
            for line in file.readlines():
                if 'NN' in line:
                    match = re.search(r'\b(\d+)$', line.strip())
                    if match:
                        nvar = int(match.group(1))
                        break
        assert nvar is not None
        if initial_conditions is not None:
            __run_and_check(mech, thermo, initial_conditions, build_path,
            num_threads, num_conditions, None, nvar, myTee)
        if test_data is not None:
            __run_and_check(mech, thermo, None, build_path,
            num_threads, num_conditions, test_data, nvar, myTee)

if __name__ == '__main__':
    parser = ArgumentParser(description='logger: Log and compare solver output for the various ODE Solvers')

    parser.add_argument('-i', '--input',
                        type=str,
                        required=True,
                        help='The mechanism to check.')
    parser.add_argument('-t', '--thermo',
                    type=str,
                    default=None,
                    help='Thermodynamic database filename (e.g., '
                         'therm.dat), or nothing if in mechanism.')
    parser.add_argument('-ic', '--initial_conditions',
                        type=str,
                        required=False,
                        default=None,
                        help='The initial conditions for same initial condition testing. '
                             'If not supplied, this is skipped.')
    parser.add_argument('-b', '--build_path',
                        required=False,
                        default='./out/',
                        help='The folder to generate the mechanism in')
    parser.add_argument('-nt', '--num_threads',
                        required=False,
                        default=multiprocessing.cpu_count(),
                        help='The number of threads to run the CPU ode solvers with.')
    parser.add_argument('-nc', '--num_conditions',
                        required=False,
                        default=1,
                        help='The number of conditions to test')
    parser.add_argument('-td', '--test_data',
                        required=False,
                        default=None,
                        help='A numpy file output generated from the partially_stirred_reactor component '
                             'of pyJac.  Used for testing if supplied.')
    args = parser.parse_args()

    assert not (args.test_data is None and args.initial_conditions is None), \
    "Either a test data file or initial conditions must be specified"

    run_log(args.input, args.thermo, args.initial_conditions, args.build_path,
        args.num_threads, args.num_conditions, args.test_data)