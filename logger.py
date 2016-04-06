#! /usr/bin/env python2.7
import sys
import numpy as np
import shutil
from os.path import join as pjoin
from os import getcwd as cwd
from os.path import expanduser
from os.path import sep
from os import makedirs
from os.path import isfile as isfile
import errno
from glob import glob
import re
import subprocess
import multiprocessing
from argparse import ArgumentParser
from pyjac import create_jacobian
from optionLoop import optionloop
np.set_printoptions(precision=15)

scons = expanduser('~/.local/bin/scons')

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
    key_arr = validator[-1, 1:]
    with open('logfile', 'a') as file:
        for f in glob(pjoin('log', globtxt)):
            if keyfile in f:
                continue
            array = np.fromfile(f, dtype='float64')
            array = array.reshape((-1, 1 + num_conditions * nvar))

            file.write(f + '\n')
            data_arr = array[-1, 1:]
            data_arr = data_arr.reshape(-1, nvar)
            key_arr = key_arr.reshape(-1, nvar)
            #now compare column by column and get max err
            err = np.abs(data_arr - key_arr) / (atol + rtol * key_arr)
            L2_err = np.linalg.norm(err, axis=1, ord=2)
            Linf_err = np.linalg.norm(err, axis=1, ord=np.inf)

            file.write('L2 (max, mean) = {:.16e}, {:.16e}\n'.format(np.max(L2_err), np.mean(L2_err)))
            file.write('Linf (max, mean) = {:.16e}, {:.16e}\n'.format(np.max(Linf_err), np.mean(Linf_err)))

def __execute(builder, num_threads, num_conditions):
    with open('logerr', 'a') as file:
        if builder == 'gpu':
            for exe in glob('*-gpu'):
                file.write('\n' + exe + '\n')
                try:
                    subprocess.check_call([pjoin(cwd(), exe), str(num_conditions)], stdout=file)
                except CalledProcessError as e:
                    output = e.output
                    returncode = e.returncode
                    file.write('Error encountered running {}\n'.format(' '.join([exe, str(num_conditions)])))
                    file.write(output + '\n')
                    file.write('Error code: {}\n'.format(returncode))
                    sys.exit(-1)
        else:
            for exe in glob('*-int'):
                if exe in keyfile:
                    continue
                file.write('\n' + exe + '\n')
                try:
                    subprocess.check_call([pjoin(cwd(), exe), str(num_threads), str(num_conditions)], stdout=file)
                except CalledProcessError as e:
                    output = e.output
                    returncode = e.returncode
                    file.write('Error encountered running {}\n'.format(' '.join([exe, str(num_threads), str(num_conditions)])))
                    file.write(output + '\n')
                    file.write('Error code: {}\n'.format(returncode))
                    sys.exit(-1)

def __check_valid(nvar, num_conditions, t_end, t_step):
    if not isfile(pjoin('log', 'valid.bin')):
        return None
    validator = np.fromfile(pjoin('log', 'valid.bin'), dtype='float64')
    if validator.shape[0] % (1 + num_conditions * nvar) == 0:
        validator = validator.reshape((-1, 1 + num_conditions * nvar))
        if np.any(np.isclose(validator[:, 0], t_end, atol=t_step/10.)):
            return validator
    return None

def __run_and_check(mech, thermo, initial_conditions, build_path,
        num_threads, num_conditions, test_data, skip_c, skip_cuda,
        atol, rtol, t_end, small_atol, small_rtol, reuse_valid):
        #first compile and run cvodes to get the baseline
        __check_exit(create_jacobian(lang='c', 
            mech_name=mech, 
            therm_name=thermo,
            initial_state=initial_conditions,
            optimize_cache=False,
            build_path=build_path))
        t_end = 1e-4 #ms
        t_step = 1e-4 #ms
        small_atol = 1e-20
        small_rtol = 1e-15
        atol = 1e-10
        rtol = 1e-6
        nvar = None
        #get num vars
        with open(pjoin(build_path, 'mechanism.h'), 'r') as file:
            for line in file.readlines():
                if 'NN' in line:
                    match = re.search(r'\b(\d+)$', line.strip())
                    if match:
                        nvar = int(match.group(1))
                        break
        if num_conditions is None and test_data is not None:
            num_conditions = np.fromfile('ign_data.bin').reshape((-1, nvar + 2)).shape[0]
        assert nvar is not None
        arg_list = ['-j{}'.format(num_threads),
                'DEBUG=FALSE', 'FAST_MATH=FALSE', 'LOG_OUTPUT=TRUE', 'LOG_END_ONLY=TRUE',
                'SHUFFLE=FALSE', 'PRINT=FALSE', 
                'mechanism_dir={}'.format(build_path), 't_end={:.0e}'.format(t_end)]

        if initial_conditions:
            arg_list.append('SAME_IC=TRUE')
            num_conditions = 1 #they're all the same
        else:
            arg_list.append('SAME_IC=FALSE')

        validator = __check_valid(nvar, num_conditions, t_end, t_end)
        if validator is None or not reuse_valid:
            with open('logerr', 'a') as file:
                extra_args = ['ATOL={:.0e}'.format(small_atol), 'RTOL={:.0e}'.format(small_rtol),
                                't_step={:.0e}'.format(t_end)]
                subprocess.check_call([scons, 'cpu'] + arg_list + extra_args, stdout=file)
                #run
                subprocess.check_call([pjoin(cwd(), 'cvodes-analytic-int'), 
                    str(num_threads), str(num_conditions)],
                    stdout=file)
                #copy to saved data
                shutil.copy(pjoin(cwd(), 'log', keyfile),
                            pjoin(cwd(), 'log', 'valid.bin'))

            validator = np.fromfile(pjoin('log', 'valid.bin'), dtype='float64')
            validator = validator.reshape((-1, 1 + num_conditions * nvar))

        langs = []
        if not skip_c:
            langs += ['c']
        if not skip_cuda:
            langs += ['cuda']
        if langs == []:
            raise Exception('No languages to test specified')
        builder = {'c':'cpu', 'cuda':'gpu'}
        oploop = optionloop({'lang' : ['c'], 
                             'cache_opt' : [False, True],
                             'smem' : [False]})
        oploop += optionloop({'lang' : ['cuda'], 
                             'cache_opt' : [False, True],
                             'smem' : [False, True]})
        arg_list += ['ATOL={}'.format(atol), 'RTOL={}'.format(rtol)]
        with open('logfile', 'a') as file:
            with open('logerr', 'a') as errfile:
                for op in oploop:
                    lang = op['lang']
                    cache_opt = op['cache_opt']
                    shared_mem = op['smem']
                    __check_exit(create_jacobian(lang=lang, 
                                    mech_name=mech, 
                                    therm_name=thermo, 
                                    initial_state=initial_conditions, 
                                    optimize_cache=cache_opt,
                                    multi_thread=num_threads,
                                    no_shared=shared_mem,
                                    build_path=build_path))
                    file.write('\ncache_opt: {}\n'
                               'shared_mem: {}\n'.format(
                               cache_opt, not shared_mem))
                    file.flush()
                    for ts in range(-4, -10, -1):
                        t_step = np.power(10., ts)
                        file.write('t_step={:.0e}\n'.format(t_step))
                        file.flush()
                        the_args = arg_list + ['t_step={:.0e}'.format(t_step)]
                        subprocess.check_call([scons, builder[lang]] + the_args, stdout=errfile, stderr=errfile)
                        __execute(builder[lang], num_threads, num_conditions)
                        __check_error(builder[lang], num_conditions, nvar, t_end,
                                        validator, atol, rtol)

def run_log(mech, thermo, initial_conditions, build_path,
        num_threads, num_conditions, test_data, skip_c, skip_cuda,
        atol, rtol, t_end, small_atol, small_rtol, reuse_valid):
    with open('logfile', 'w') as file:
        pass
    with open('logerr', 'w') as file:
        pass
    if initial_conditions is not None:
        with open('logfile', 'a') as file:
            file.write('Running Same ICs\n')
        __run_and_check(mech, thermo, initial_conditions, build_path, 
            num_threads, 1 if num_conditions is None else num_conditions,
            None, skip_c, skip_cuda, atol, rtol, t_end,
            small_atol, small_rtol, reuse_valid)
    if test_data is not None:
        with open('logfile', 'a') as file:
            file.write('PaSR ICs\n')
        try:
            shutil.copyfile(test_data, 'ign_data.bin')
        except shutil.Error:
            pass
        __run_and_check(mech, thermo, '', build_path,
        num_threads, num_conditions, test_data, skip_c, skip_cuda,
        atol, rtol, t_end, small_atol, small_rtol, 
        reuse_valid)

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
                        default=None,
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
                        default=1e-10,
                        help='The absolute tolerance to use during integration')
    parser.add_argument('-rtol', '--rel_tolerance',
                        required=False,
                        type=float,
                        default=1e-6,
                        help='The relative tolerance to use during integration')
    parser.add_argument('-satol', '--abs_tolerance_small',
                        required=False,
                        type=float,
                        default=1e-20,
                        help='The absolute tolerance to use during integration')
    parser.add_argument('-srtol', '--rel_tolerance_small',
                        required=False,
                        type=float,
                        default=1e-15,
                        help='The relative tolerance to use during integration')
    parser.add_argument('-tend', '--end_time',
                        required=False,
                        type=float,
                        default=1e-3,
                        help='The end time of the simulation')
    parser.add_argument('-ru', '--reuse_valid',
                        required=False,
                        default=False,
                        action='store_true')
    args = parser.parse_args()

    assert not (args.test_data is None and args.initial_conditions is None), \
    "Either a test data file or initial conditions must be specified"

    create_dir(args.build_path)
    create_dir('./log/')

    run_log(args.input, args.thermo, args.initial_conditions, args.build_path,
        args.num_threads, args.num_conditions, args.test_data,
        args.skip_c, args.skip_cuda, args.abs_tolerance, args.rel_tolerance,
        args.end_time, args.abs_tolerance_small, args.rel_tolerance_small, args.reuse_valid)