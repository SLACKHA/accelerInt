#! /usr/env/bin python2.7

#A SConstruct file for accerlerInt, heavily adapted from Cantera

import os
import re
import sys
import SCons
import platform
from buildutils import *

valid_commands = ('build', 'test', 'help')

for command in COMMAND_LINE_TARGETS:
    if command not in valid_commands:
        print 'ERROR: unrecognized command line target: %r' % command
        sys.exit(0)

print 'INFO: SCons is using the following Python interpreter:', sys.executable

home = os.getcwd()

# ******************************************************
# *** Set system-dependent defaults for some options ***
# ******************************************************

opts = Variables('accelerInt.conf')
env = Environment(tools=['default', 'cuda'])

class defaults: pass

compiler_options = [
    ('CC',
     """The C compiler to use. """,
     env['CC'])]
opts.AddVariables(*compiler_options)
opts.Update(env)

defaults.CCFlags = '-m64 -std=c99'
defaults.NVCCFLAGS = '-m64'
defaults.optimizeCCFlags = '-O3 -funroll-loops'
defaults.debugCCFlags = '-O0 -g -fbounds-check -Wunused-variable -Wunused-parameter' \
                       '-Wall -ftree-vrp'
defaults.optimizeNVCCFlags = '-O3'
defaults.debugNVCCFlags = '-g -G -O0'
defaults.CCLinkFlags = ['-fPIC']
defaults.NVCCLinkFlags = ['-fPIC']
defaults.CCLibs = ['m']
defaults.NVCCLibs = ['stdc++', 'cuda', 'cudart']
defaults.warningFlags = '-Wall'
defaults.fftwIncDir = os.path.join('usr', 'local', 'include')
defaults.fftwLibDir = os.path.join('usr', 'local', 'lib')
defaults.sundialsLibDir = os.path.join('usr', 'local', 'lib')
defaults.sundialsIncDir = os.path.join('usr', 'local', 'include')

if 'gcc' in env.subst('$CC'):
    defaults.optimizeCCFlags += ' -Wno-inline'
    defaults.optimizeCCFlags += ' -mtune=native'
elif 'icc' in env.subst('$CC'):
    defaults.CCFlags += ' -vec-report0'
    defaults.CCFlags += ' -Wcheck'
    defaults.optimizeCCFlags += ' -xhost -ipo'
elif 'clang' in env.subst('$CC'):
    defaults.CCFlags += ' -fcolor-diagnostics'
else:
    print "WARNING: Unrecognized C compiler '{}'".format(env['CC'])

env['OS'] = platform.system()
if env['OS'] == 'Windows':
    print 'Windows is unsupported'
    sys.exit(-1)
if env['OS'] == 'Darwin':
    defaults.threadFlags = ''
else:
    defaults.threadFlags = '-pthread'

defaults.env_vars = 'LD_LIBRARY_PATH'

defaults.mechanism_dir = os.path.join(home, 'mechanism')

# Transform lists into strings to keep accelerInt.conf clean
for key,value in defaults.__dict__.items():
    if isinstance(value, (list, tuple)):
        defaults.__dict__[key] = ' '.join(value)

default_lapack_dir = os.path.join('usr', 'local', 'lib')
try:
    default_lapack_dir = os.path.join(os.environ['MKLROOT'], 'lib', 'intel64')
except:
    pass

config_options = [
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
    ('NVCCFLAGS',
     'Compiler flags passed to the CUDA compiler, regardless of optimization level.',
     defaults.NVCCFLAGS),
    ('CCFLAGS',
     'Compiler flags passed to both the C compiler, regardless of optimization level',
     defaults.CCFlags),
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
    ('fftw3_lib_dir',
     'The directory where the FFTW3 libraries are located',
     defaults.fftwLibDir),
    ('mechanism_dir',
     'The directory where mechanism files are located.',
     defaults.mechanism_dir),
    BoolVariable(
        'DEBUG', 'Compiles with Debugging flags and information.', False),
    ('ATOL', 'Absolute Tolerance for integrators', '1e-15'),
    ('RTOL', 'Relative Tolerance for integrators', '1e-8'),
    ('t_step', 'Step size for integrator', '1e-6'),
    ('N_RA', 'The size of the Rational Approximant for the Exponential Integrators.', '10'),
    BoolVariable(
        'SAME_IC', 'Use the same initial conditions (specified during mechanism creation) during integration.', False),
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

opts.AddVariables(*config_options)
opts.Update(env)
opts.Save('accelerInt.conf', env)

if 'help' in COMMAND_LINE_TARGETS:
    ### Print help about configuration options and exit.
    print """
        *****************************************************
        *   Configuration options for building accelerInt   *
        *****************************************************
The following options can be passed to SCons to customize the accelerInt
build process. They should be given in the form:
    scons build option1=value1 option2=value2
Variables set in this way will be stored in the 'accelerInt.conf' file and reused
automatically on subsequent invocations of scons. Alternatively, the
configuration options can be entered directly into 'accelerInt.conf' before
running 'scons build'. The format of this file is:
    option1 = 'value1'
    option2 = 'value2'
        **************************************************
"""

    for opt in opts.options:
        print '\n'.join(formatOption(env, opt))
    sys.exit(0)

valid_arguments = (set(opt[0] for opt in compiler_options) |
                   set(opt[0] for opt in config_options))
for arg in ARGUMENTS:
    if arg not in valid_arguments:
        print 'Encountered unexpected command line argument: %r' % arg
        sys.exit(0)

#now finalize flags

CCFlags = listify(env['CCFLAGS'])
NVCCFlags = listify(env['NVCCFLAGS'])
CCLibs = listify(defaults.CCLibs)
CCLinkFlags = listify(defaults.CCLinkFlags)
NVCCLibs = listify(defaults.NVCCLibs)
NVCCLinkFlags = listify(defaults.NVCCLinkFlags)
if env['DEBUG']:
    CCFlags.extend(listify(defaults.debugCCFlags))
    NVCCFlags.extend(listify(defaults.debugNVCCFlags))
else:
    CCFlags.extend(listify(defaults.optimizeCCFlags))
    NVCCFlags.extend(listify(defaults.optimizeNVCCFlags))

#open mp and reg counts
CCFlags.append(env['openmp_flags'])
CCLinkFlags.append(env['openmp_flags'])

#directories
mech_dir = os.path.join(home, env['mechanism_dir'])
generic_dir = os.path.join(home, 'generic')
radau2a_dir = os.path.join(home, 'radau2a')
exp_int_dir = os.path.join(home, 'exponential_integrators')
exp4_int_dir = os.path.join(home, 'exponential_integrators', 'exp4')
exprb43_int_dir = os.path.join(home, 'exponential_integrators', 'exprb43')
cvodes_dir = os.path.join(home, 'cvodes')

common_dir_list = [generic_dir, mech_dir]

reg_count = ''
build_cuda = True
try:
    with open(os.path.join(mech_dir, 'regcount'), 'r') as file:
        reg_count = file.readline().strip()
except:
    build_cuda = False
    print 'Could not find register count, skipping CUDA integrators'
NVCCFlags.append(['-maxrregcount {}'.format(reg_count), '-Xcompiler {}'.format(env['openmp_flags'])])

#link lines
CCLibDirs = [env['blas_lapack_dir']]
CCLibs += listify(env['blas_lapack_libs'])
NVCCLinkFlags.append([env['openmp_flags'], '-Xlinker -rpath {}/lib64'.format(env['CUDA_TOOLKIT_PATH'])])

#options
if not env['FAST_MATH']:
    if env['CC'] == 'icc':
        CCFlags += ['-fp-model precise']
    NVCCFlags += listify('--ftz=false --prec-div=true --prec-sqrt=true --fmad=false')
else:
    if env['CC'] == 'icc':
        CCFlags += ['-fp-model fast=2']
    elif env['CC'] == 'gcc':
        CCFlags += ['-ffast-math']
    NVCCFlags += ['--use_fast_math']

#write the solver_options file
#scons will automagically decide what needs recompilation based on this
#hooray!
with open(os.path.join(generic_dir, 'solver_options.h'), 'w') as file:
    file.write("""
        /*
        solver_options.h

        A file that in conjunction with scons to specify the various options
        to the solvers

        */
        #ifndef SOLV_OPT_HEAD
        #define SOLV_OPT_HEAD

        /* Tolerances and Timestep */
        #define ATOL ({})
        #define RTOL ({})
        #define t_step ({})

        /** Machine precision constant. */
        #define EPS DBL_EPSILON
        #define SMALL DBL_MIN

        /** type of rational approximant (n, n) */
        #define N_RA ({})

        /** Unsigned int typedef. */
        typedef unsigned int uint;
        /** Unsigned short int typedef. */
        typedef unsigned short int usint;

        /* CVodes Parameters */
        //#define CV_MAX_ORD (5) //maximum order for method, default for BDF is 5
        #define CV_MAX_STEPS (20000) // maximum steps the solver will take in one timestep
        //#define CV_HMAX (0)  //upper bound on step size (integrator step, not global timestep)
        //#define CV_HMIN (0) //lower bound on step size (integrator step, not global timestep)
        #define CV_MAX_HNIL (1) //maximum number of t + h = t warnings
        #define CV_MAX_ERRTEST_FAILS (5) //maximum number of error test fails before an error is thrown

        //#define COMPILE_TESTING_METHODS //comment out to remove unit testing stubs

        //turn on to log the krylov space and step sizes to log.txt
        #ifdef DEBUG
          #if defined(RB43) || defined(EXP4)
            #define LOG_KRYLOV_AND_STEPSIZES
          #endif
        #endif
        """.format(env['ATOL'], env['RTOL'], env['t_step'], env['N_RA'])
        )

    if env['SAME_IC']:
        file.write("""
    // load same initial conditions for all threads
    #define SAME_IC
        """
        )

    if env['DEBUG']:
        file.write("""
    // load same initial conditions for all threads
    #define DEBUG
        """
        )

    if env['SHUFFLE']:
        file.write("""
    // shuffle initial conditions randomly
    #define SHUFFLE
        """)

    if env['PRINT']:
        file.write("""
    //print the output to screen
    #define PRINT
        """)

    if env['IGN']:
        file.write("""
    // output ignition time
    #define IGN
        """)

    if env['LOG_OUTPUT']:
        file.write("""
    //log output to file
    #define LOG_OUTPUT
        """)
    file.write("""
    #endif
        """)

NVCCFlags = listify(NVCCFlags)
CCFlags = listify(CCFlags)
CCLinkFlags = listify(CCLinkFlags)
NVCCLinkFlags = listify(NVCCLinkFlags)
CCLibs = listify(CCLibs)
NVCCLibs = listify(NVCCLibs)
CCLibDirs = listify(CCLibDirs)

env['CCFLAGS'] = CCFlags
env['LINKFLAGS'] = CCLinkFlags
env['LIBPATH'] = CCLibDirs
env['LIBS'] = CCLibs
env['NVCCFLAGS'] = NVCCFlags
env['NVCCLINKFLAGS'] = NVCCLinkFlags
env['NVCCLIBS'] = NVCCLibs

variant = 'release' if not env['DEBUG'] else 'debug'
env['variant']=variant

env['CPPPATH'] = common_dir_list
env['NVCCPATH'] += common_dir_list

target_list = []

#copy a good SConscript into the mechanism dir
import shutil
env_save = env.Clone()
Export('env')
try:
    shutil.copyfile(os.path.join(defaults.mechanism_dir, 'SConscript'), os.path.join(mech_dir, 'SConscript'))
except shutil.Error:
    pass


mech_c, mech_cuda = SConscript(os.path.join(mech_dir, 'SConscript'), variant_dir=os.path.join(mech_dir, variant))

gen_c, gen_cuda = SConscript(os.path.join(generic_dir, 'SConscript'), variant_dir=os.path.join(generic_dir, variant))

env['CPPDEFINES'] = ['RADAU2A']
env['CPPPATH'] += [radau2a_dir]
env['NVCCDEFINES'] = ['RADAU2A']
env['NVCCPATH'] += [radau2a_dir]
Export('env')
rad_c, rad_cuda = SConscript(os.path.join(radau2a_dir, 'SConscript'), variant_dir=os.path.join(radau2a_dir, variant))
target_list.append(
    env.Program(target='radau2a-int', source=mech_c + gen_c + rad_c, variant_dir=os.path.join(radau2a_dir, variant)))
dlink = env.CUDADLink(target='radua2a-int-gpu', source=mech_cuda + gen_cuda + rad_cuda, variant_dir=os.path.join(radau2a_dir, variant))
target_list.append(dlink)
target_list.append(
    env.CUDAProgram(target='radau2a-int-gpu', source=dlink + mech_cuda + gen_cuda + rad_cuda, variant_dir=os.path.join(radau2a_dir, variant)))

env = env_save.Clone()
env['CPPPATH'] += [env['fftw3_inc_dir'], exp_int_dir, exp4_int_dir]
env['LIBPATH'].append(env['fftw3_lib_dir'])
env['LIBS'].append('fftw3')
env['NVCCPATH'] += [exp_int_dir, exp4_int_dir]
env['CPPDEFINES'] = ['EXP4']
env['NVCCDEFINES'] = ['EXP4']
Export('env')
exp_c, exp_cuda = SConscript(os.path.join(exp_int_dir, 'SConscript'), variant_dir=os.path.join(exp_int_dir, variant))
exp4_c, exp4_cuda = SConscript(os.path.join(exp4_int_dir, 'SConscript'), variant_dir=os.path.join(exp4_int_dir, variant))
env.Install(os.path.join(exp4_int_dir, variant), exp_c + exp_cuda)
target_list.append(
    env.Program(target='exp4-int', source=mech_c + gen_c + exp_c + exp4_c, variant_dir=os.path.join(exp4_int_dir, variant)))
dlink = env.CUDADLink(target='exp4-int-gpu', source=mech_cuda + gen_cuda + exp_cuda + exp4_cuda, variant_dir=os.path.join(exp4_int_dir, variant))
target_list.append(dlink)
target_list.append(
    env.CUDAProgram(target='exp4-int-gpu', source=dlink + mech_cuda + gen_cuda + exp_cuda + exp4_cuda, variant_dir=os.path.join(exp4_int_dir, variant)))

env = env_save.Clone()
env['CPPPATH'] += [env['fftw3_inc_dir'], exp_int_dir, exprb43_int_dir]
env['LIBPATH'].append(env['fftw3_lib_dir'])
env['LIBS'].append('fftw3')
env['NVCCPATH'] += [exp_int_dir, exprb43_int_dir]
env['CPPDEFINES'] = ['RB43']
env['NVCCDEFINES'] = ['RB43']
Export('env')
exp_c, exp_cuda = SConscript(os.path.join(exp_int_dir, 'SConscript'), variant_dir=os.path.join(exp_int_dir, variant))
rb43_c, rb43_cuda = SConscript(os.path.join(exprb43_int_dir, 'SConscript'), variant_dir=os.path.join(exprb43_int_dir, variant))
exp_c = env.Install(os.path.join(exprb43_int_dir, variant), exp_c)
exp_cuda = env.Install(os.path.join(exprb43_int_dir, variant), exp_cuda)
target_list.append(
    env.Program(target='exprb43-int', source=mech_c + gen_c + exp_c + rb43_c, variant_dir=os.path.join(exprb43_int_dir, variant)))
dlink = env.CUDADLink(target='exprb43-int-gpu', source=mech_cuda + gen_cuda + exp_cuda + rb43_cuda, variant_dir=os.path.join(exprb43_int_dir, variant))
target_list.append(dlink)
target_list.append(
    env.CUDAProgram(target='exprb43-int-gpu', source=dlink + mech_cuda + gen_cuda + exp_cuda + rb43_cuda, variant_dir=os.path.join(exprb43_int_dir, variant)))

env = env_save.Clone()
env['CPPDEFINES'] = ['CVODES']
env['CPPPATH'] += [cvodes_dir, env['sundials_inc_dir']]
env['LIBPATH'] += [env['sundials_lib_dir']]
env['LIBS'] += ['sundials_cvodes', 'sundials_nvecserial']
Export('env')
cvodes_c = SConscript(os.path.join(cvodes_dir, 'SConscript'), variant_dir=os.path.join(cvodes_dir, variant))
env = env_save.Clone()
env['CPPDEFINES'] = ['CVODES']
env['CPPPATH'] += [cvodes_dir, env['sundials_inc_dir']]
env['LIBPATH'] += [env['sundials_lib_dir']]
env['LIBS'] += ['sundials_cvodes', 'sundials_nvecserial']
Export('env')
fd_cvodes = [x for x in cvodes_c if not 'analytic' in str(x)]
analytic_cvodes = [x for x in cvodes_c if not 'cvodes_init' in str(x)]
cv_gen_c = [x for x in gen_c if not 'solver_generic' in str(x)]
target_list.append(
    env.Program(target='cvodes-int', source=mech_c + cv_gen_c + fd_cvodes, variant_dir=os.path.join(cvodes_dir, variant)))
env['CPPDEFINES'] += ['SUNDIALS_ANALYTIC_JACOBIAN']
Export('env')
target_list.append(
    env.Program(target='cvodes-analytic-int', source=mech_c + cv_gen_c + analytic_cvodes, variant_dir=os.path.join(cvodes_dir, variant)))


Alias('build', target_list)
Default(target_list)