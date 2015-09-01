#! /usr/env/bin python2.7

#A SConstruct file for accerlerInt, heavily adapted from Cantera

import os
import re
import sys
import SCons

valid_commands = ('build', 'clean', 'test')

for command in COMMAND_LINE_TARGETS:
    if command not in valid_commands:
        print 'ERROR: unrecognized command line target: %r' % command
        sys.exit(0)

print 'INFO: SCons is using the following Python interpreter:', sys.executable

home = os.getcwd()

def get_files(directory, extension, file_filter=None, inverse_filter=None):
    file_list = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))
     and f.endswith(extension)]
    if file_filter:
        if isinstance(file_filter, list):
            file_list = [f for f in file_list if all(re.search(filt, f) for filt in file_filter)]
        else:
            file_list = [f for f in file_list if re.search(file_filter, f)]
    if inverse_filter:
        if isinstance(inverse_filter, list):
            file_list = [f for f in file_list if not any(re.search(filt, f) for filt in inverse_filter)]
        else:
            file_list = [f for f in file_list if not re.search(inverse_filter, f)]
    return file_list

def removeFiles(directory, extension):
    """ Remove file (if it exists) and print a log message """
    file_list = get_files(directory, extension)
    for name in file_list:
        if os.path.exists(name):
            print 'Removing file "%s"' % name
            os.remove(name)

extraEnvArgs = {}

if 'clean' in COMMAND_LINE_TARGETS:
    removeFiles(os.path.join(home, 'exponential_integrators', 'obj'), '.o')
    removeFiles(os.path.join(home, 'exponential_integrators', 'exp4', 'obj'), '.o')
    removeFiles(os.path.join(home, 'exponential_integrators', 'exprb43', 'obj'), '.o')
    removeFiles(os.path.join(home, 'cvodes', 'obj'), '.o')
    removeFiles(os.path.join(home, 'radau2a', 'obj'), '.o')
    removeFiles(os.path.join(home, 'generic', 'obj'), '.o')
    print 'Done removing output files.'
    if COMMAND_LINE_TARGETS == ['clean']:
        # Just exit if there's nothing else to do
        sys.exit(0)
    else:
        Alias('clean', [])

# ******************************************************
# *** Set system-dependent defaults for some options ***
# ******************************************************

opts = Variables('accelerInt.conf')
env = Environment()

class defaults: pass

compiler_options = [
    ('NVCC',
     'The CUDA compiler to use.',
     env['NVCC']),
    ('NCC',
     'The CUDA C compiler to use.',
     env['CC'])
    ('CC',
     """The C compiler to use. """,
     env['CC'])]
opts.AddVariables(*compiler_options)
opts.Update(env)

opts = Variables('accerlerInt.conf')
defaults.cFlags = '-m64 -std=c99'
defaults.cudaFlags = '-m64'
defaults.optimizeCFlags = '-O3 -funroll-loops'
defaults.debugCFlags = '-O0 -g -fbounds-check -Wunused-variable -Wunused-parameter' \
                       '-Wall -ftree-vrp'
defaults.optimizeCudaFlags = '-O3'
defaults.debugCudaFlags = '-g -G -O0'
defaults.cLink = '-fPIC -lm'
defaults.cudaLink = '-fPIC -lcuda -lcudart -lstdc++'
defaults.warningFlags = '-Wall'
defaults.fftwIncDir = os.path.join('usr', 'local', 'include')
defaults.fftwLibDir = os.path.join('usr', 'local', 'lib')
defaults.sundialsLibDir = os.path.join('usr', 'local', 'lib')
defaults.sundialsIncDir = os.path.join('usr', 'local', 'include')

if 'gcc' in env.subst('$CC'):
    defaults.optimizeCFlags += ' -Wno-inline'
    defaults.optimizeCFlags += ' -mtune=native'
elif 'icc' in env.subst('$CC'):
    defaults.cFlags = '-vec-report0'
    defaults.warningFlags = '-Wcheck'
    defaults.optimizeCFlags += ' -xhost -ipo'
elif 'clang' in env.subst('$CC'):
    defaults.cFlags = '-fcolor-diagnostics'
else:
    print "WARNING: Unrecognized C compiler '{}'".format(env['CC'])

if env['OS'] == 'Windows':
    print 'Windows is unsupported'
    sys.exit(-1)
if env['OS'] == 'Darwin':
    defaults.threadFlags = ''
else:
    defaults.threadFlags = '-pthread'

defaults.env_vars = 'LD_LIBRARY_PATH'

# Transform lists into strings to keep accelerInt.conf clean
for key,value in defaults.__dict__.items():
    if isinstance(value, (list, tuple)):
        defaults.__dict__[key] = ' '.join(value)

default_lapack_dir = os.path.join('usr', 'local', 'lib')
try:
    default_lapack_dir = os.path.join(env.subst('MKLROOT'), 'lib', 'intel64')
except:
    pass
default_cuda_path = os.path.join(env['NVCC'][:env['NVCC'].index('nvcc')], 'lib64')
default_cuda_sdk_path = os.path.join(env['NVCC'][:env['NVCC'].index('bin')], 'samples', 'common', 'inc')

config_opts = [
    ('blas_lapack_libs',
        """Comma separated list of blas/lapack libraries to use for the various solvers, 
        set blas_lapack_libs to the the list of libraries that should be passed to the linker,
         separated by commas,
        e.g. "lapack,blas" or "lapack,f77blas,cblas,atlas".""",
        'mkl_rt,mkl_intel_lp64,mkl_core,mkl_gnu_thread,dl,mkl_mc,mkl_def'
        ),
    PathVariable('blas_lapack_dir',
        """Directory containing the libraries specified by 'blas_lapack_libs'.""",
        default_lapack_dir, PathVariable.PathAccept),
    ('nvcc_flags',
     'Compiler flags passed to the CUDA compiler, regardless of optimization level.',
     defaults.cudaFlags),
    ('c_flags',
     'Compiler flags passed to both the C compiler, regardless of optimization level',
     defaults.cFlags),
    ('thread_flags',
     'Compiler and linker flags for POSIX multithreading support.',
     defaults.threadFlags),
    ('openmp_flags',
     'Compiler and linker flags for OpenMP support.',
     '-fopenmp'),
    ('compute_level',
     'The CUDA compute level of your GPUs',
     'sm_20'),
    ('sundials_inc_dir',
     'The directory where the sundials headers are located',
     defaults.sundialsIncDir),
    ('sundials_lib_dir',
     'The directory where the sundials libraries are located',
     defaults.sundialsLibDir),
    ('fftw3_inc_dir',
     'The directory where the FFTW3 headers are located',
     defaults.fftwIncDir),
    ('fftw3_inc_dir',
     'The directory where the FFTW3 libraries are located',
     defaults.fftwLibDir),
    BoolVariable(
        'DEBUG', 'Compiles with Debugging flags and information.', False),
    ('ATOL', 'Absolute Tolerance for integrators', '1e-15'),
    ('RTOL', 'Relative Tolerance for integrators', '1e-8'),
    ('step_size', 'Step size for integrator', '1e-6'),
    ('N_RA', 'The size of the Rational Approximant for the Exponential Integrators.', '10'),
    BoolVariable(
        'SAME_ICS', 'Use the same initial conditions (specified during mechanism creation) during integration.', False),
    BoolVariable(
        'SHUFFLE', 'Shuffle the PaSR initial conditions.', False),
    BoolVariable(
        'SHUFFLE', 'Shuffle the PaSR initial conditions.', False),
    BoolVariable(
        'PRECONDITION', 'Precondition (via clustering) the PaSR initial conditions.', False),
    BoolVariable(
        'PRINT', 'Log output to screen.', False),
    BoolVariable(
        'LOG_OUTPUT', 'Log output to file.', False),
    BoolVariable(
        'IGN', 'Log ignition time.', False),
    BoolVariable(
        'FAST_MATH', 'Compile with Fast Math.', False)
]

opts.AddVariables(*config_opts)
opts.Update(env)
opts.Save('accelerInt.conf', env)

#now finalize flags

cFlags = env['cFlags']
cudaFlags = env['cudaFlags']
cLink = defaults.cLink
cudaLink = defaults.cudaLink
if env['DEBUG']:
    cFlags += defaults.debugCFlags
    cudaFlags += defaults.debugCudaFlags
else:
    cFlags += defaults.optimizeCFlags
    cudaFlags += defaults.optimizeCudaFlags

#open mp and reg counts
cFlags += env['openmp_flags']

reg_count = ''
with open(os.path.join(mech_dir, 'regcount'), 'r') as file:
    reg_count = file.readline().strip()
cudaFlags += '-maxrregcount {} -Xcompiler {}'.format(reg_count, env['openmp_flags'])

#link lines
cLink += ' -L{}'.format(env['blas_lapack_dir']) + \
    ''.join([' -l{}'.format(x) for x in env['blas_lapack_libs'].split(',')) + \
    ' ' + env['openmp_flags'] + ' ' + defaults.threadFlags
cudaLink += ' -Xlinker -rpath {}/lib64'.format(env['CUDA_PATH'])

#options
if not env['FAST_MATH']:
    if env['CC'] == 'icc':
        cFlags += ' -fp-model precise'
    cudaFlags ' --ftz=false --prec-div=true --prec-sqrt=true --fmad=false'
else:
    if env['CC'] == 'icc':
        cFlags += ' -fp-model fast=2'
    elif env['CC'] == 'gcc':
        cFlags == ' -ffast-math'
    cudaFlags += ' --use_fast_math'




#directories
mech_dir = os.path.join(home, 'mechanism', 'src')
mech_obj = os.path.join(home, 'mechanism', 'obj')
generic_dir = os.path.join(home, 'generic', 'src')
generic_obj = os.path.join(home, 'generic', 'obj')
radau2a_dir = os.path.join(home, 'radau2a', 'src')
radau2a_obj = os.path.join(home, 'radau2a', 'obj')
exp_int_dir = os.path.join(home, 'exponential_integrators', 'src')
exp_int_obj = os.path.join(home, 'exponential_integrators', 'obj')
exp4_int_dir = os.path.join(home, 'exponential_integrators', 'exp4', 'src')
exp4_int_obj = os.path.join(home, 'exponential_integrators', 'exp4', 'obj')
exprb43_int_dir = os.path.join(home, 'exponential_integrators', 'exprb43', 'src')
exprb43_int_obj = os.path.join(home, 'exponential_integrators', 'exprb43', 'obj')
cvodes_dir = os.path.join(home, 'cvodes', 'src')
cvodes_obj = os.path.join(home, 'cvodes', 'obj')

#common file lists
mechanism_src = get_files(mech_dir, '.c')
mechanism_cuda_src = get_files(mech_dir, '.cu') + get_files(mech_dir, '.c', file_filter='mass_mole')

generic_src = get_files(generic_dir, '.c', inverse_filter='fd_jacob')
generic_cuda_src = get_files(generic_dir, '.cu', inverse_filter='fd_jacob')

solver_and_mech = mechanism_src + generic_src
solver_and_mech_cuda = mechanism_cuda_src + generic_cuda_src

exp_int_src = get_files(exp_int_dir, '.c')
exp_int_cuda_src = get_files(exp_int_dir, '.cu') + get_files(exp_int_dir, '.c', file_filter='linear-algebra')

exp_solver_and_mech = solver_and_mech + exp_int_src
exp_solver_and_mech_cuda = solver_and_mech + exp_int_cuda_src

cvodes_base_src = filter(lambda x: not 'jacob' in x, mechanism_src) + get_files(generic_dir, '.c', inverse_filter='solver_generic')
cvodes_analytical_src = mechanism_src + get_files(generic_dir, '.c', inverse_filter='solver_generic')

#set up targets
target_list = []
terget_list.append(
    env.Program(target='radau2a-int', sources=solver_and_mech + get_files(radau2a_dir, '.c')]))
target_list.append(
    env.Program(target='radau2a-int-gpu', sources=solver_and_mech_cuda + get_files(radau2a_dir, '.cu')]))
target_list.append(
    env.Program(target='exp4-int', sources=exp_solver_and_mech + get_files(exp4_int_dir, '.c')]))
target_list.append(
    env.Program(target='exp4-int-gpu', sources=exp_solver_and_mech_cuda + get_files(exp4_int_dir, '.cu')]))
target_list.append(
    env.Program(target='exprb4-int', sources=exp_solver_and_mech + get_files(exprb4_int_dir, '.c')]))
target_list.append(
    env.Program(target='exprb4-int-gpu', sources=exp_solver_and_mech_cuda + get_files(exprb4_int_dir, '.cu')]))
target_list.append(
    env.Program(target='cvodes-int', sources=cvodes_base_src + get_files(cvodes_dir, '.c', inverse_filter='cvodes_jac')]))
target_list.append(
    env.Program(target='cvodes-analytical-int', sources=cvodes_base_src + get_files(cvodes_dir, '.c', inverse_filter='cvodes_jac')]))
