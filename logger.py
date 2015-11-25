#! /usr/bin/env python2.7
import sys
import numpy as np
import shutil
from os.path import join as pjoin
from os import getcwd as cwd
from os.path import expanduser
from os.path import sep
from os import makedirs
import errno
from glob import glob
import re
import subprocess
import multiprocessing
from argparse import ArgumentParser
sys.path.append(pjoin(expanduser('~'),
                         'pyJac'))
import pyJac
np.set_printoptions(precision=15)

keyfile = 'cvodes-analytic-int-log.bin'

def create_dir(path):
    try:
        makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

def __check_exit(x):
    if x != 0:
        sys.exit(x)

def __check_error(builder, num_conditions, nvar, t, validator, atol, rtol):
    globtxt = '*-gpu-log.bin' if builder == 'gpu' else '*-int-log.bin'
    match = np.where(np.isclose(validator[:, 0], t, atol=atol))
    print match, validator[match, 0]
    print validator[:, 0]
    key_arr = validator[match, 1:]
    non_zero = np.where(key_arr != 0)
    kview = key_arr[non_zero]
    with open('logfile', 'a') as file:
        file.write('t={}\n'.format(t))
        for f in glob(pjoin('log', globtxt)):
            if keyfile in f:
                continue
            array = np.fromfile(f, dtype='float64')
            array = array.reshape((-1, 1 + num_conditions * nvar))

            file.write(f + '\n')
            data_arr = array[1:, 1:]
            #now compare column by column and get max err

            aview = data_arr[non_zero]

            err = np.abs(aview - kview) / kview
            err = np.sum(np.abs(err)**2)
            err = np.sum(err)
            norm_err = np.sqrt(err) / float(nvar * (num_conditions - 1))
            
            file.write("{:.16e}\t{:.16e}\t\n".format(max_err, norm_err))

def __execute(builder, num_threads, num_conditions):
    with open('logfile', 'a') as file:
        if builder == 'gpu':
            for exe in glob('*-gpu'):
                file.write('\n' + exe + '\n')
                subprocess.check_call([pjoin(cwd(), exe), str(num_conditions)], stdout=file)
        else:
            for exe in glob('*-int'):
                if exe in keyfile:
                    continue
                file.write('\n' + exe + '\n')
                subprocess.check_call([pjoin(cwd(), exe), str(num_threads), str(num_conditions)], stdout=file)

def __run_and_check(mech, thermo, initial_conditions, build_path,
        num_threads, num_conditions, test_data, skip_c, skip_cuda,
        atol, rtol):
        #first compile and run cvodes to get the baseline
        __check_exit(pyJac.create_jacobian(lang='c', 
            mech_name=mech, 
            therm_name=thermo,
            initial_state=initial_conditions,
            optimize_cache=False,
            build_path=build_path))
        small_step = atol
        t_step = np.linspace(-10, -3)
        t_step = np.array([10.**x for x in t_step])
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
        arg_list = ['-j{}'.format(num_threads),
                'DEBUG=FALSE', 'FAST_MATH=FALSE', 'LOG_OUTPUT=TRUE', 
                'SHUFFLE=FALSE', 'PRINT=FALSE', 'mechanism_dir={}'.format(build_path),
                'ATOL={}'.format(atol), 'RTOL={}'.format(rtol)]
        if initial_conditions:
            arg_list.append('SAME_IC=TRUE')
            num_conditions = num_threads #they're all the same, so do a reasonable #
        else:
            arg_list.append('SAME_IC=FALSE')
        with open('logfile', 'a') as file:
            subprocess.check_call(['scons', 'cpu'] + arg_list +
                        ['t_step={}'.format(small_step),
                         'num_steps={}'.format(int(t_step[-1]/small_step))], stdout=file)
            #run
            subprocess.check_call([pjoin(cwd(), 'cvodes-analytic-int'), 
                str(num_threads), str(num_conditions)],
                stdout=file)
            #copy to saved data
            shutil.copy(pjoin(cwd(), 'log', keyfile),
                        pjoin(cwd(), 'log', 'valid.bin'))

        validator = np.fromfile(pjoin('log', 'valid.bin'), dtype='float64')
        validator = validator.reshape((-1, 1 + num_conditions * nvar))

        arg_list.append('num_steps=1')
        langs = []
        if not skip_c:
            langs += ['c']
        if not skip_cuda:
            langs += ['cuda']
        if langs == []:
            raise Exception('No languages to test specified')
        builder = {'c':'cpu', 'cuda':'gpu'}
        opt = [False]#[True, False]
        smem = [False, True]
        #now check for various options
        for lang in langs:
            for cache_opt in opt:
                if lang == 'cuda':
                    for shared_mem in smem:
                        for t in t_step:
                            with open('logfile', 'a') as file:
                                #subprocess.check_call(['scons', '-c'])
                                file.write('\ncache_opt: {}\n'
                                       'shared_mem: {}\n'.format(
                                        cache_opt, not shared_mem))
                                __check_exit(pyJac.create_jacobian(lang=lang, 
                                mech_name=mech, 
                                therm_name=thermo, 
                                initial_state=initial_conditions, 
                                optimize_cache=cache_opt,
                                multi_thread=num_threads,
                                no_shared=shared_mem,
                                build_path=build_path))

                                subprocess.check_call(['scons', builder[lang]] + arg_list + 
                                        ['t_step={}'.format(t)], stdout=file)
                                __execute(builder[lang], num_threads, num_conditions)
                                __check_error(builder[lang], num_conditions, nvar, t,
                                                validator, atol, rtol)

                else:
                    for t in t_step:
                        with open('logfile', 'a') as file:
                            file.write('\ncache_opt: {}\n'.format(
                                        cache_opt))
                            __check_exit(pyJac.create_jacobian(lang=lang, 
                            mech_name=mech, 
                            therm_name=thermo, 
                            initial_state=initial_conditions, 
                            optimize_cache=cache_opt,
                            multi_thread=num_threads,
                            build_path=build_path))

                            subprocess.check_call(['scons', builder[lang]] + arg_list + 
                                        ['t_step={}'.format(t)], stdout=file)
                            __execute(builder[lang], num_threads, num_conditions)
                            __check_error(builder[lang], num_conditions, nvar, t,
                                            validator, atol, rtol)

def run_log(mech, thermo, initial_conditions, build_path,
        num_threads, num_conditions, test_data, skip_c, skip_cuda,
        atol, rtol):
    with open('logfile', 'w') as file:
        pass
    if initial_conditions is not None:
        with open('logfile', 'a') as file:
            file.write('Running Same ICs\n')
        __run_and_check(mech, thermo, initial_conditions, build_path, 
            num_threads, num_conditions, None, skip_c, skip_cuda,
            atol, rtol)
    if test_data is not None:
        with open('logfile', 'a') as file:
            file.write('PaSR ICs\n')
        try:
            shutil.copyfile(test_data, 'ign_data.bin')
        except shutil.Error:
            pass
        __run_and_check(mech, thermo, '', build_path,
        num_threads, num_conditions, test_data, skip_c, skip_cuda,
        atol, rtol)

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
                        type=str,
                        required=False,
                        default=pjoin(cwd(),'out') + sep,
                        help='The folder to generate the mechanism in')
    parser.add_argument('-nt', '--num_threads',
                        type=int,
                        required=False,
                        default=multiprocessing.cpu_count(),
                        help='The number of threads to run the CPU ode solvers with.')
    parser.add_argument('-nc', '--num_conditions',
                        type=int,
                        required=False,
                        default=1,
                        help='The number of conditions to test')
    parser.add_argument('-td', '--test_data',
                        type=str,
                        required=False,
                        default=None,
                        help='A numpy file output generated from the partially_stirred_reactor component '
                             'of pyJac.  Used for testing if supplied.')
    parser.add_argument('-sc', '--skip_c',
                        required=False,
                        default=False,
                        action='store_true',
                        help='Use to skip C testing. Note baseline is still calculated using CVODES')
    parser.add_argument('-scu', '--skip_cuda',
                        required=False,
                        default=False,
                        action='store_true',
                        help='Use to skip CUDA testing.')
    parser.add_argument('-atol', '--abs_tolerance',
                        required=False,
                        type=float,
                        default=1e-15,
                        help='The absolute tolerance to use during integration')
    parser.add_argument('-rtol', '--rel_tolerance',
                        required=False,
                        type=float,
                        default=1e-8,
                        help='The relative tolerance to use during integration')
    args = parser.parse_args()

    assert not (args.test_data is None and args.initial_conditions is None), \
    "Either a test data file or initial conditions must be specified"

    create_dir(args.build_path)
    create_dir('./log/')

    run_log(args.input, args.thermo, args.initial_conditions, args.build_path,
        args.num_threads, args.num_conditions, args.test_data,
        args.skip_c, args.skip_cuda, args.abs_tolerance, args.rel_tolerance)