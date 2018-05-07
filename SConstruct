#! /usr/env/bin python2.7

# A SConstruct file for accerlerInt, heavily adapted from Cantera

from __future__ import print_function

import os
import re
import sys
import SCons
import platform
from buildutils import *
import shutil

# mirrored from pyjac
header_ext = dict(c='.h', cuda='.cuh')
"""dict: header extensions based on language"""

valid_commands = ('build', 'test', 'cpu', 'gpu', 'help')

for command in COMMAND_LINE_TARGETS:
    if command not in valid_commands:
        print('ERROR: unrecognized command line target: %r' % command)
        sys.exit(0)

print('INFO: SCons is using the following Python interpreter:', sys.executable)

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
    print('CUDA not found, no GPU integrators will be built')
    build_cuda = False


class defaults:
    pass


compiler_options = [
    ('toolchain',
     """The C/C++ compilers to use. """,
     'gnu')]
opts.AddVariables(*compiler_options)
opts.Update(env)
if env['toolchain'] == 'intel':
    env['CC'] = 'icc'
    env['CXX'] = 'icpc'
elif env['toolchain'] == 'gnu':
    env['CC'] = 'gcc'
    env['CXX'] = 'g++'
elif env['toolchain'] == 'clang':
    env['CC'] = 'clang'
    env['CXX'] = 'clang++'
else:
    print('Invalid toolchain specified {}, accepted values are {}'.format(
        env['toolchain'],
        ', '.join(['intel', 'gnu', 'clang'])))

defaults.CFlags = '-std=c99'
defaults.CCFlags = '-m64'
defaults.CXXFlags = ''
defaults.NVCCFLAGS = '-m64 -Xptxas -v'
defaults.NVCCArch = 'sm_20'
defaults.optimizeCCFlags = '-O3 -funroll-loops'
defaults.debugCCFlags = '-O0 -g -fbounds-check -Wunused-variable -Wunused-parameter' \
                       ' -Wall -ftree-vrp'
defaults.optimizeNVCCFlags = '-O3'
defaults.debugNVCCFlags = '-g -G -O0'
defaults.LinkFlags = ['-fPIC']
defaults.NVCCLinkFlags = ['-fPIC']
defaults.Libs = ['m']
defaults.NVCCLibs = ['stdc++', 'cuda', 'cudart']
defaults.warningFlags = '-Wall'
defaults.fftwIncDir = os.path.join('usr', 'local', 'include')
defaults.fftwLibDir = os.path.join('usr', 'local', 'lib')
defaults.sundialsLibDir = os.path.join('usr', 'local', 'lib')
defaults.sundialsIncDir = os.path.join('usr', 'local', 'include')
defaults.boostIncDir = os.path.join('usr', 'local', 'include')

if env['toolchain'] == 'gnu':
    defaults.optimizeCCFlags += ' -Wno-inline'
    defaults.optimizeCCFlags += ' -mtune=native'
elif env['toolchain'] == 'intel':
    defaults.CCFlags += ' -vec-report0'
    defaults.CCFlags += ' -Wcheck'
    defaults.optimizeCCFlags += ' -xhost -ipo'
elif env['toolchain'] == 'clang':
    defaults.CCFlags += ' -fcolor-diagnostics'

env['OS'] = platform.system()
if env['OS'] == 'Windows':
    print('Windows is unsupported')
    sys.exit(-1)
if env['OS'] == 'Darwin':
    defaults.threadFlags = ''
else:
    defaults.threadFlags = '-pthread'

defaults.env_vars = 'LD_LIBRARY_PATH'

defaults.mechanism_dir = os.path.join(home, 'mechanism')

# Transform lists into strings to keep accelerInt.conf clean
for key, value in defaults.__dict__.items():
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
    ('toolchain',
        'The compiler tools to use for C/C++ code',
        'gnu'),
    ('NVCCFLAGS',
     'Compiler flags passed to the CUDA compiler, regardless of optimization level.',
     defaults.NVCCFLAGS),
    ('CCFLAGS',
     'Compiler flags passed to both the C and C++ compiler, regardless of optimization level',
     defaults.CCFlags),
    ('CXXFLAGS',
     'Compiler flags passed to only the C++ compiler, regardless of optimization level',
     defaults.CXXFlags),
    ('CFLAGS',
     'Compiler flags passed to only the C compiler, regardless of optimization level',
     defaults.CFlags),
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
    ('boost_inc_dir',
     'The directory where the boost headers are located',
     defaults.boostIncDir),
    ('fftw3_inc_dir',
     'The directory where the FFTW3 headers are located',
     defaults.fftwIncDir),
    ('fftw3_lib_dir',
     'The directory where the FFTW3 libraries are located',
     defaults.fftwLibDir),
    ('mechanism_dir',
     'The directory where mechanism files are located.',
     defaults.mechanism_dir),
    EnumVariable('buildtype',
     'The type of build to run (exe or lib)', 'exe',
     allowed_values=('exe', 'lib')),
    BoolVariable(
        'DEBUG', 'Compiles with Debugging flags and information.', False),
    ('ATOL', 'Absolute Tolerance for integrators', '1e-10'),
    ('RTOL', 'Relative Tolerance for integrators', '1e-6'),
    ('t_step', 'Step size for integrator', '1e-6'),
    ('t_end', 'End time of the integrator', '1e-6'),
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
        'LOG_END_ONLY', 'Log only beginning and end states to file.', False),
    BoolVariable(
        'IGN', 'Log ignition time.', False),
    BoolVariable(
        'FAST_MATH', 'Compile with Fast Math.', False),
    BoolVariable(
        'FINITE_DIFFERENCE', 'Use a finite difference Jacobian (not recommended)', False),
    ('DIVERGENCE_WARPS', 'If specified, measure divergence in that many warps', '0'),
    ('CV_HMAX', 'If specified, the maximum stepsize for CVode', '0'),
    ('CV_MAX_STEPS', 'If specified, the maximum stepsize for CVode', '20000'),
    ('CONST_TIME_STEP', 'If specified, adaptive timestepping will be turned off (for logging purposes)', False)
]

opts.AddVariables(*config_options)
opts.Update(env)
opts.Save('accelerInt.conf', env)

if 'help' in COMMAND_LINE_TARGETS:
    ### Print help about configuration options and exit.
    print ("""
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
""")

    for opt in opts.options:
        print('\n'.join(formatOption(env, opt)))
    sys.exit(0)

valid_arguments = (set(opt[0] for opt in compiler_options) |
                   set(opt[0] for opt in config_options))
for arg in ARGUMENTS:
    if arg not in valid_arguments:
        print('Encountered unexpected command line argument: %r' % arg)
        sys.exit(0)

# now finalize flags

CFlags = listify(env['CFLAGS'])
CCFlags = listify(env['CCFLAGS'])
CXXFlags = listify(env['CXXFLAGS'])
NVCCFlags = listify(env['NVCCFLAGS'])
NVCCFlags += ['-arch={}'.format(env['compute_level'])]
Libs = listify(defaults.Libs)
LinkFlags = listify(defaults.LinkFlags)
NVCCLibs = listify(defaults.NVCCLibs)
NVCCLinkFlags = listify(defaults.NVCCLinkFlags)
if env['DEBUG']:
    CCFlags.extend(listify(defaults.debugCCFlags))
    NVCCFlags.extend(listify(defaults.debugNVCCFlags))
else:
    CCFlags.extend(listify(defaults.optimizeCCFlags))
    NVCCFlags.extend(listify(defaults.optimizeNVCCFlags))

# open mp and reg counts
CCFlags.append(env['openmp_flags'])
LinkFlags.append(env['openmp_flags'])

# directories
mech_dir = os.path.join(home, env['mechanism_dir'])
generic_dir = os.path.join(home, 'generic')
radau2a_dir = os.path.join(home, 'radau2a')
exp_int_dir = os.path.join(home, 'exponential_integrators')
exp4_int_dir = os.path.join(home, 'exponential_integrators', 'exp4')
exprb43_int_dir = os.path.join(home, 'exponential_integrators', 'exprb43')
cvodes_dir = os.path.join(home, 'cvodes')
rk78_dir = os.path.join(home, 'rk78')
rkc_dir = os.path.join(home, 'rkc')

common_dir_list = [generic_dir, mech_dir]

reg_count = ''
if build_cuda:
    try:
        with open(os.path.join(mech_dir, 'regcount'), 'r') as file:
            reg_count = file.readline().strip()
        NVCCFlags.append(['-maxrregcount {}'.format(reg_count)])
    except:
        print('Register count not found. Using default')
# add openmp
NVCCFlags.append(['-Xcompiler {}'.format(
    env['openmp_flags'])])


def check_extras(lang, subdir, check_str, check_file, list_file):
    if check_file is None:
        check_file = os.path.join(mech_dir, subdir + header_ext[lang])
    else:
        check_file = os.path.join(mech_dir, check_file)
    have_extras = False
    try:
        with open(check_file, 'r') as file:
            for line in file.readlines():
                if "#include \"{}\"".format(check_str) in line:
                    have_extras = True
                    break
    except IOError, e:
        if e.errno == 2:
            pass

    if have_extras:
        with open(os.path.join(mech_dir, subdir, list_file), 'r') as file:
            vals = [os.path.join(mech_dir, subdir, x) for x in
                    file.readline().strip().split()]
        env['extra_{}_{}'.format(lang, subdir)] = vals

        # copy a good SConscript into the mechanism dir
        try:
            shutil.copyfile(os.path.join(
                                defaults.mechanism_dir, subdir, 'SConscript'),
                            os.path.join(mech_dir, subdir, 'SConscript'))
        except shutil.Error:
            pass
        except IOError, e:
            if e.errno == 2:
                pass

# check for additional files


if not env['FINITE_DIFFERENCE']:
    check_extras('c', 'jacobs', 'jacobs/jac_include.h', 'jacob.h', 'jac_list_c')
    check_extras('cuda', 'jacobs', 'jacobs/jac_include.cuh', 'jacob.cuh', 'jac_list_cuda')
check_extras('c', 'rates', 'rates/rates_include.h', 'rxn_rates.c', 'rate_list_c')
check_extras('cuda', 'rates', 'rates/rates_include.cuh', 'rxn_rates.cu', 'rate_list_cuda')

# link lines
LibDirs = listify(env['blas_lapack_dir'])
Libs += listify(env['blas_lapack_libs'])
if build_cuda:
    NVCCLinkFlags.append([env['openmp_flags'], '-Xlinker -rpath {}/lib64'.format(env['CUDA_TOOLKIT_PATH'])])

# options
if not env['FAST_MATH']:
    if env['toolchain'] == 'intel':
        CCFlags += ['-fp-model precise']
    NVCCFlags += listify('--ftz=false --prec-div=true --prec-sqrt=true --fmad=false')
else:
    if env['toolchain'] == 'intel':
        CCFlags += ['-fp-model fast=2']
    elif env['toolchain'] == 'gnu':
        CCFlags += ['-ffast-math']
    NVCCFlags += ['--use_fast_math']


def write_options(lang, dir):
    # write the solver_options file
    # scons will automagically decide what needs recompilation based on this
    # hooray!
    with open(os.path.join(dir, 'solver_options{}'.format(header_ext[lang])), 'w') as file:
        file.write("""
        /*! \file

        \brief A file generated by Scons that specifies various options to the solvers

        This file, autogenerated by SCons contains a number of definitions that can be
        enabled via command line options.  These control the integrator behaviour and
        alternately enable special behaviour such as logging, divergence measuring, etc.

        Note that it is not typical for all these options to be turned on at the same time,
        but it is done here for documentation purposes.

        */
        #ifndef SOLV_OPT_HEAD
        #define SOLV_OPT_HEAD

        /* Tolerances and Timestep */

        #include <float.h>

        /*! Absolute solver tolerance */
        #define ATOL ({})
        /*! Relative solver tolerance */
        #define RTOL ({})
        /*! Solver timestep (may be used to force multiple timesteps per global integration step) */
        #define t_step ({})
        /*! Global integration timestep */
        #define end_time ({})

        /*! Machine precision constant. */
        #define EPS DBL_EPSILON
        /*! Smallest representable double */
        #define SMALL DBL_MIN

        /** type of rational approximant (n, n) */
        #define N_RA ({})

        """.format(env['ATOL'], env['RTOL'], env['t_step'], env['t_end'], env['N_RA'])
                   )

        file.write(
                   """
        /* CVodes Parameters */
        //#define CV_MAX_ORD (5) //maximum order for method, default for BDF is 5
        /*! Maximum steps the solver will take in one timestep set to -1 (disabled) by default */
        #define CV_MAX_STEPS ({})
        {}  //upper bound on step size (integrator step, not global timestep)
        //#define CV_HMIN (0) //lower bound on step size (integrator step, not global timestep)
        /*! Number of t + h == t warnings emitted by CVODE (used to cleanup output) */
        #define CV_MAX_HNIL (1)
        /*! Maximum number of CVODE error test fails before an error is thrown */
        #define CV_MAX_ERRTEST_FAILS (5) //maximum number of error test fails before an error is thrown
        """.format(env['CV_MAX_STEPS'],
                   '//#define CV_HMAX (0)' if env['CV_HMAX'] == '0' else
                   '#define CV_HMAX ({})'.format(env['CV_HMAX']))
                  )

        if env['SAME_IC']:
            file.write("""
        /*! Load same initial conditions (defined in mechanism.c or mechanism.cu) for all threads */
        #define SAME_IC
            """)

        if env['DEBUG']:
            file.write("""
        //*! Turn on debugging symbols, and use O0 optimization */
        #define DEBUG
        #include <fenv.h>
            """)

        if env['SHUFFLE']:
            file.write("""
        /*! Use shuffled initial conditions */
        #define SHUFFLE
            """)

        if env['PRINT']:
            file.write("""
        /*! Print the output to screen */
        #define PRINT
            """)

        if env['IGN']:
            file.write("""
        /*! Output ignition time (determined by simple T0 + 400 criteria) */
        #define IGN
            """)

        if env['LOG_OUTPUT'] or env['LOG_END_ONLY']:
            file.write("""
        /*! Log output to binary file */
        #define LOG_OUTPUT
            """)

            file.write("""
        /*! Turn on to log the krylov space and step sizes */
        #define LOG_KRYLOV_AND_STEPSIZES
        """)

        if env['LOG_END_ONLY']:
                file.write("""
        /*! Log output to binary file only on final timestep */
        #define LOG_END_ONLY
        """)

        if env['FINITE_DIFFERENCE']:
            file.write("""
            /*! Use a Finite Difference Jacobian */
            #define FINITE_DIFFERENCE
            """)

        if int(env['DIVERGENCE_WARPS']) > 0:
            file.write("""
        /*! Measure the thread divergence for this many initial conditions */
        #define DIVERGENCE_TEST ({})
        """.format(int(float(env['DIVERGENCE_WARPS']) * 32)))

        if env['CONST_TIME_STEP']:
            file.write("""
            /*! Define to turn off adaptive time stepping */
            #define CONST_TIME_STEP
            """)

        file.write("""
        #endif
            """)


write_options('c', generic_dir)
if build_cuda:
    write_options('cuda', generic_dir)

NVCCFlags = listify(NVCCFlags)
CFlags = listify(CFlags)
CCFlags = listify(CCFlags)
CXXFlags = listify(CXXFlags)
LinkFlags = listify(LinkFlags)
NVCCLinkFlags = listify(NVCCLinkFlags)
Libs = listify(Libs)
NVCCLibs = listify(NVCCLibs)
LibDirs = listify(LibDirs)

env['CFLAGS'] = CFlags
env['CCFLAGS'] = CCFlags
env['CXXFLAGS'] = CXXFlags
env['LINKFLAGS'] = LinkFlags
env['LIBPATH'] = LibDirs
env['LIBS'] = Libs
env['NVCCFLAGS'] = NVCCFlags
env['NVCCLINKFLAGS'] = NVCCLinkFlags
env['NVCCLIBS'] = NVCCLibs

variant = 'release' if not env['DEBUG'] else 'debug'
env['variant'] = variant

env['CPPPATH'] = common_dir_list
if 'NVCC_INC_PATH' not in env:
    env['NVCC_INC_PATH'] = []
if build_cuda:
    env['NVCC_INC_PATH'] += common_dir_list

build_lib = env['buildtype'] == 'lib'

target_list = {}

# copy a good SConscript into the mechanism dir
try:
    shutil.copyfile(os.path.join(defaults.mechanism_dir, 'SConscript'),
                    os.path.join(mech_dir, 'SConscript'))
except shutil.Error:
    pass
except IOError, e:
    if e.errno == 2:
        pass


def builder(env_save, cmech, cumech, newdict, mydir, variant,
            target_base, target_list, additional_sconstructs=None,
            filter_out=None):

    # update the env
    env = env_save.Clone()
    for key, value in newdict.iteritems():
        if not isinstance(value, list):
            value = list(value)
        if key not in env:
            env[key] = value
        else:
            env[key] += value

    Export('env')

    mygendir = os.path.join(generic_dir,
                            os.path.basename(os.path.normpath(mydir)))
    cgen, cugen = SConscript(os.path.join(generic_dir, 'SConscript'),
                             variant_dir=os.path.join(mygendir, variant),
                             src_dir=generic_dir)

    cint, cuint = SConscript(os.path.join(mydir, 'SConscript'),
                             variant_dir=os.path.join(mydir, variant),
                             src_dir=mydir)
    # check for additional sconstructs
    if additional_sconstructs is not None:
        for thedir in additional_sconstructs:
            ctemp, cutemp = SConscript(os.path.join(thedir, 'SConscript'),
                                       variant_dir=os.path.join(thedir,
                                                                variant),
                                       src_dir=thedir)
            cint += ctemp
            cuint += cutemp

    ffilter = ['main'] if build_lib else ['interface']
    if filter_out is not None:
        if not isinstance(filter_out, list):
            filter_out = [filter_out]
        ffilter += filter_out
    if ffilter:
        cint = [x for x in cint if not any(y in str(x) for y in ffilter)]
        cmech = [x for x in cmech if not any(y in str(x) for y in ffilter)]
        cgen = [x for x in cgen if not any(y in str(x) for y in ffilter)]
        if cumech is not None:
            cuint = [x for x in cuint if not any(y in str(x[0]) for y in ffilter)]
            cumech = [x for x in cumech if not any(y in str(x[0]) for y in ffilter)]
            cugen = [x for x in cugen if not any(y in str(x[0]) for y in ffilter)]

    target_list[target_base] = []
    target_list[target_base].append(
        env.Program(target=target_base,
                    source=cmech + cgen + cint,
                    variant_dir=os.path.join(mydir, variant)))
    if env['build_cuda'] and cumech:
        target_list[target_base + '-gpu'] = []
        dlink = env.CUDADLink(
            target=target_base+'-gpu',
            source=cumech + cugen + cuint,
            variant_dir=os.path.join(mydir, variant))
        target_list[target_base + '-gpu'].append(dlink)
        target_list[target_base + '-gpu'].append(
            env.CUDAProgram(target=target_base+'-gpu',
                            source=cumech + cugen + cuint + dlink,
                            variant_dir=os.path.join(mydir, variant)))
        cuint += dlink
    return cgen + cint, cugen + cuint


env['build_cuda'] = build_cuda
env_save = env.Clone()
Export('env')

mech_c, mech_cuda = SConscript(os.path.join(mech_dir, 'SConscript'),
                               variant_dir=os.path.join(mech_dir, variant))
if 'extra_c_jacobs' in env or 'extra_cuda_jacobs' in env:
    cJacs, cudaJacs = SConscript(os.path.join(mech_dir, 'jacobs', 'SConscript'),
                                 variant_dir=os.path.join(mech_dir, 'jacobs', variant))
    mech_c += cJacs
    mech_cuda += cudaJacs
if 'extra_c_rates' in env or 'extra_cuda_rates' in env:
    cRates, cudaRates = SConscript(os.path.join(mech_dir, 'rates', 'SConscript'),
                                   variant_dir=os.path.join(mech_dir, 'rates', variant))
    mech_c += cRates
    mech_cuda += cudaRates

# radua
new_defines = {}
new_defines['CPPDEFINES'] = ['RADAU2A']
new_defines['CPPPATH'] = [radau2a_dir]
new_defines['NVCCDEFINES'] = ['RADAU2A']
new_defines['NVCC_INC_PATH'] = [radau2a_dir]
radau_c, radau_cuda = builder(env_save, mech_c,
                              mech_cuda if build_cuda else None,
                              new_defines, radau2a_dir,
                              variant, 'radau2a-int', target_list)

# exp4
new_defines = {}
new_defines['CPPPATH'] = [env['fftw3_inc_dir'], exp_int_dir, exp4_int_dir]
new_defines['LIBPATH'] = [env['fftw3_lib_dir']]
new_defines['LIBS'] = ['fftw3']
new_defines['NVCC_INC_PATH'] = [exp_int_dir, exp4_int_dir]
new_defines['CPPDEFINES'] = ['EXP4']
new_defines['NVCCDEFINES'] = ['EXP4']
exp4_c, exp4_cuda = builder(env_save, mech_c, mech_cuda,
                            new_defines, exp4_int_dir,
                            variant, 'exp4-int', target_list,
                            [exp_int_dir])

# exprb43
new_defines = {}
new_defines['CPPPATH'] = [env['fftw3_inc_dir'], exp_int_dir, exprb43_int_dir]
new_defines['LIBPATH'] = [env['fftw3_lib_dir']]
new_defines['LIBS'] = ['fftw3']
new_defines['NVCC_INC_PATH'] = [exp_int_dir, exprb43_int_dir]
new_defines['CPPDEFINES'] = ['RB43']
new_defines['NVCCDEFINES'] = ['RB43']
rb43c, rb43cu = builder(env_save, mech_c,
                        mech_cuda if build_cuda else None,
                        new_defines, exprb43_int_dir,
                        variant, 'exprb43-int', target_list,
                        [exp_int_dir])

# rkc
new_defines = {}
new_defines['CPPPATH'] = [rkc_dir]
new_defines['NVCC_INC_PATH'] = [rkc_dir]
new_defines['CPPDEFINES'] = ['RKC']
new_defines['NVCCDEFINES'] = ['RKC']
rkc, rkccu = builder(env_save, mech_c,
                     mech_cuda if build_cuda else None,
                     new_defines, rkc_dir,
                     variant, 'rkc-int', target_list,
                     filter_out=['nverse'])

# cvodes
new_defines = {}
new_defines['CPPDEFINES'] = ['CVODES']
new_defines['CPPPATH'] = [cvodes_dir, env['sundials_inc_dir']]
new_defines['LIBPATH'] = [env['sundials_lib_dir']]
new_defines['LIBS'] = ['sundials_cvodes', 'sundials_nvecserial']
cvodesc, _ = builder(env_save, mech_c, None, new_defines,
                         cvodes_dir, variant, 'cvodes-int',
                         target_list, None,
                         filter_out=['solver_generic', 'nverse'])

# rk78
new_defines = {}
env_cpp = env_save.Clone()
env_cpp['CCFLAGS'] = []
new_defines['CPPDEFINES'] = ['RK78']
new_defines['CPPPATH'] = [rk78_dir, env['boost_inc_dir']]
builder(env_save, mech_c, None, new_defines,
        rk78_dir, variant, 'rk78-int', target_list,
        filter_out=['solver_generic', 'nverse'])

flat_values = []
cpu_vals = []
gpu_vals = []
for key, value in target_list.iteritems():
    flat_values.extend(value)
    if 'gpu' not in key:
        cpu_vals.extend(value)
    else:
        gpu_vals.extend(value)
Alias('build', flat_values)
Alias('cpu', cpu_vals)
Alias('gpu', gpu_vals)
Default(flat_values)
