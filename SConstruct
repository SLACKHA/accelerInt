#! /usr/env/bin python2.7

#A SConstruct file for accerlerInt, heavily adapted from Cantera

import os
import re
import sys
import SCons
import platform
from buildutils import *
import shutil

valid_commands = ('build', 'test', 'cpu', 'gpu', 'help')

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
build_cuda = True
try:
    env = Environment(tools=['default', 'cuda'])
except:
    env = Environment(tools=['default'])
    print 'CUDA not found, no GPU integrators will be built'
    build_cuda = False

class defaults: pass

compiler_options = [
    ('CC',
     """The C compiler to use. """,
     env['CC'])]
opts.AddVariables(*compiler_options)
opts.Update(env)

defaults.CCFlags = '-m64 -std=c99'
defaults.NVCCFLAGS = '-m64'
defaults.NVCCArch = 'sm_20'
defaults.optimizeCCFlags = '-O3 -funroll-loops'
defaults.debugCCFlags = '-O0 -g -fbounds-check -Wunused-variable -Wunused-parameter' \
                       ' -Wall -ftree-vrp'
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
     defaults.NVCCArch),
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
    ('num_steps', 'Total number of integrator steps to take', '1'),
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
NVCCFlags += ['-arch={}'.format(env['compute_level'])]
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
if build_cuda:
    try:
        with open(os.path.join(mech_dir, 'regcount'), 'r') as file:
            reg_count = file.readline().strip()
    except:
        print('Register count not found, skipping cuda integrators')
        build_cuda = False
NVCCFlags.append(['-maxrregcount {}'.format(reg_count), '-Xcompiler {}'.format(env['openmp_flags'])])

#extra jacobians
try:
    have_extras = False
    with open(os.path.join(mech_dir, 'jacob.h'), 'r') as file:
        for line in file.readlines():
            if "#include \"jacobs/jac_include.h\"" in line:
                have_extras = True
                break
    if have_extras:
        with open(os.path.join(mech_dir, 'jacobs', 'jac_list_c'), 'r') as file:
            vals = [os.path.join(mech_dir, 'jacobs', x) for x in 
                    file.readline().strip().split()]
        env['extra_c_jacobs'] = vals

        #copy a good SConscript into the mechanism dir
        try:
            shutil.copyfile(os.path.join(defaults.mechanism_dir, 'jacobs', 'SConscript'),
                                os.path.join(mech_dir, 'jacobs', 'SConscript'))
        except shutil.Error:
            pass
        except IOError, e:
            if e.errno == 2:
                pass
except:
    pass
try:
    have_extras = False
    with open(os.path.join(mech_dir, 'jacob.cuh'), 'r') as file:
        for line in file.readlines():
            if "#include \"jacobs/jac_include.cuh\"" in line:
                have_extras = True
                break
    if have_extras:
        with open(os.path.join(mech_dir, 'jacobs', 'jac_list_cuda'), 'r') as file:
            vals = [os.path.join(mech_dir, 'jacobs', x) for x in 
                    file.readline().strip().split()]
        env['extra_cuda_jacobs'] = vals
        #copy a good SConscript into the mechanism dir
        try:
            shutil.copyfile(os.path.join(defaults.mechanism_dir, 'jacobs', 'SConscript'),
                                os.path.join(mech_dir, 'jacobs', 'SConscript'))
        except shutil.Error:
            pass
        except IOError, e:
            if e.errno == 2:
                pass
except:
    pass


#link lines
CCLibDirs = [env['blas_lapack_dir']]
CCLibs += listify(env['blas_lapack_libs'])
if build_cuda:
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
    #include <float.h>
    #define ATOL ({})
    #define RTOL ({})
    #define t_step ({})
    #define num_steps ({})

    /** Machine precision constant. */
    #define EPS DBL_EPSILON
    #define SMALL DBL_MIN

    /** type of rational approximant (n, n) */
    #define N_RA ({})

    /* CVodes Parameters */
    //#define CV_MAX_ORD (5) //maximum order for method, default for BDF is 5
    #define CV_MAX_STEPS (20000) // maximum steps the solver will take in one timestep
    //#define CV_HMAX (0)  //upper bound on step size (integrator step, not global timestep)
    //#define CV_HMIN (0) //lower bound on step size (integrator step, not global timestep)
    #define CV_MAX_HNIL (1) //maximum number of t + h = t warnings
    #define CV_MAX_ERRTEST_FAILS (5) //maximum number of error test fails before an error is thrown

    //#define COMPILE_TESTING_METHODS //comment out to remove unit testing stubs
    """.format(env['ATOL'], env['RTOL'], env['t_step'], env['num_steps'], env['N_RA'])
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
    #include <fenv.h>
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
        //turn on to log the krylov space and step sizes
        #define LOG_KRYLOV_AND_STEPSIZES
        #define MAX_STEPS 10000
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
env['variant'] = variant

env['CPPPATH'] = common_dir_list
if build_cuda:
    env['NVCCPATH'] += common_dir_list

target_list = {}

#copy a good SConscript into the mechanism dir
try:
    shutil.copyfile(os.path.join(defaults.mechanism_dir, 'SConscript'), os.path.join(mech_dir, 'SConscript'))
except shutil.Error:
    pass
except IOError, e:
    if e.errno == 2:
        pass

def builder(env_save, cobj, cuobj, newdict, mydir, variant,
             target_base, target_list, additional_sconstructs=None,
             filter_out=None):
    #update the env
    env = env_save.Clone()
    for key, value in newdict.iteritems():
        if not isinstance(value, list):
            value = list(value)
        if not key in env:
            env[key] = value
        else:
            env[key] += value
    Export('env')
    int_c, int_cuda = SConscript(os.path.join(mydir, 'SConscript'),
        variant_dir=os.path.join(mydir, variant))
    #check for additional sconstructs
    if additional_sconstructs is not None:
        for thedir in additional_sconstructs:
            temp_c, temp_cu = SConscript(os.path.join(thedir, 'SConscript'),
                variant_dir=os.path.join(thedir, variant))
            int_c += temp_c
            int_cuda += temp_cu
    if filter_out is not None:
        int_c = [x for x in int_c if not filter_out in str(x)]
        int_cuda = [x for x in int_cuda if not filter_out in str(x)] 

    target_list[target_base] = []
    target_list[target_base].append(
        env.Program(target=target_base,
                    source=cobj + int_c,
                    variant_dir=os.path.join(mydir, variant)))
    if env['build_cuda']:
        target_list[target_base + '-gpu'] = []
        dlink = env.CUDADLink(
            target=target_base+'-gpu', 
            source=cuobj + int_cuda, 
            variant_dir=os.path.join(mydir, variant))
        target_list[target_base + '-gpu'].append(dlink)
        target_list[target_base + '-gpu'].append(
            env.CUDAProgram(target=target_base+'-gpu',
             source=cuobj + int_cuda + dlink, 
             variant_dir=os.path.join(mydir, variant)))

def cvodes_builder(env_save, cobj, newdict, mydir, variant,
                 target_list, additional_sconstructs=None):
    #update the env
    env = env_save.Clone()
    for key, value in newdict.iteritems():
        if not isinstance(value, list):
            value = list(value)
        if not key in env:
            env[key] = value
        else:
            env[key] += value
    Export('env')
    fd_c, ana_c = SConscript(os.path.join(mydir, 'SConscript'), 
        variant_dir=os.path.join(mydir, variant))
    #check for additional sconstructs
    if additional_sconstructs is not None:
        for thedir in additional_sconstructs:
            temp_c = SConscript(os.path.join(thedir, 'SConscript'),
                variant_dir=os.path.join(thedir, variant))
            int_c += temp_c
    target_list['cvodes-int'] = []
    target_list['cvodes-int'].append(
        env.Program(target='cvodes-int',
                    source=cobj + fd_c,
                    variant_dir=os.path.join(mydir, variant)))

    target_list['cvodes-analytic-int'] = []
    target_list['cvodes-analytic-int'].append(
        env.Program(target='cvodes-analytic-int',
                    source=cobj + ana_c,
                    variant_dir=os.path.join(mydir, variant)))

env['build_cuda'] = build_cuda
env_save = env.Clone()
Export('env')

mech_c, mech_cuda = SConscript(os.path.join(mech_dir, 'SConscript'), variant_dir=os.path.join(mech_dir, variant))
if 'extra_c_jacobs' in env or 'extra_cuda_jacobs' in env:
    cJacs, cudaJacs = SConscript(os.path.join(mech_dir, 'jacobs', 'SConscript'), 
        variant_dir=os.path.join(mech_dir, 'jacobs', variant))
    
    mech_c += cJacs
    cudaJacs += cudaJacs

gen_c, gen_cuda = SConscript(os.path.join(generic_dir, 'SConscript'), variant_dir=os.path.join(generic_dir, variant))

if build_cuda and os.path.isfile(os.path.join(mech_dir, 'launch_bounds.cuh')):
    solver_main_cu = [x for x in gen_cuda if 'solver_main' in str(x[0])]
    Depends(solver_main_cu, os.path.join(mech_dir, 'launch_bounds.cuh'))
    Depends(solver_main_cu, os.path.join(generic_dir, 'solver_options.h'))

#radua
new_defines = {}
new_defines['CPPDEFINES'] = ['RADAU2A']
new_defines['CPPPATH'] = [radau2a_dir]
new_defines['NVCCDEFINES'] = ['RADAU2A']
new_defines['NVCCPATH'] = [radau2a_dir]
builder(env_save, mech_c + gen_c, 
    mech_cuda + gen_cuda if build_cuda else None,
    new_defines, radau2a_dir,
    variant, 'radau2a-int', target_list)

#exp4
new_defines = {}
new_defines['CPPPATH'] = [env['fftw3_inc_dir'], exp_int_dir, exp4_int_dir]
new_defines['LIBPATH'] = [env['fftw3_lib_dir']]
new_defines['LIBS'] = ['fftw3']
new_defines['NVCCPATH'] = [exp_int_dir, exp4_int_dir]
new_defines['CPPDEFINES'] = ['EXP4']
new_defines['NVCCDEFINES'] = ['EXP4']
builder(env_save, mech_c + gen_c, 
    mech_cuda + gen_cuda,
    new_defines, exp4_int_dir,
    variant, 'exp4-int', target_list,
    [exp_int_dir])

#exprb43
new_defines = {}
new_defines['CPPPATH'] = [env['fftw3_inc_dir'], exp_int_dir, exprb43_int_dir]
new_defines['LIBPATH'] = [env['fftw3_lib_dir']]
new_defines['LIBS'] = ['fftw3']
new_defines['NVCCPATH'] = [exp_int_dir, exprb43_int_dir]
new_defines['CPPDEFINES'] = ['RB43']
new_defines['NVCCDEFINES'] = ['RB43']
builder(env_save, mech_c + gen_c, 
    mech_cuda + gen_cuda,
    new_defines, exprb43_int_dir,
    variant, 'exprb43-int', target_list,
    [exp_int_dir])

#fd cvodes
new_defines = {}
new_defines['CPPDEFINES'] = ['CVODES']
new_defines['CPPPATH'] = [cvodes_dir, env['sundials_inc_dir']]
new_defines['LIBPATH'] = [env['sundials_lib_dir']]
new_defines['LIBS'] = ['sundials_cvodes', 'sundials_nvecserial']
cv_gen_c = [x for x in gen_c if not 'solver_generic' in str(x)]
cvodes_builder(env_save, mech_c + cv_gen_c, new_defines,
    cvodes_dir, variant, target_list)

flat_values = []
cpu_vals = []
gpu_vals = []
for key, value in target_list.iteritems():
    flat_values.extend(value)
    if not 'gpu' in key:
        cpu_vals.extend(value)
    else:
        gpu_vals.extend(value)
Alias('build', flat_values)
Alias('cpu', cpu_vals)
Alias('gpu', gpu_vals)
Default(flat_values)