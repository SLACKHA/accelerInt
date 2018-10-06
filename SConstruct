# A SConstruct file for accerlerInt, heavily adapted from Cantera

from __future__ import print_function

import os
import sys
import platform
import shutil
import subprocess
import textwrap
from distutils.version import LooseVersion
from distutils.spawn import find_executable

from buildutils import listify, formatOption, getCommandOutput


valid_commands = ('cpu', 'opencl', 'gpu', 'cpu-wrapper', 'opencl-wrapper', 'help')

for command in COMMAND_LINE_TARGETS:
    if command not in valid_commands:
        print('ERROR: unrecognized command line target: %r' % command)
        sys.exit(0)

print('INFO: SCons is using the following Python interpreter:', sys.executable)

home = Dir('.').path

# ******************************************************
# *** Set system-dependent defaults for some options ***
# ******************************************************

opts = Variables('accelerInt.conf')
build_cuda = True
try:
    env = Environment(tools=['default', 'cuda'])
except Exception:
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
defaults.CXXFlags = '-std=c++11'
defaults.NVCCFLAGS = '-m64 -Xptxas -v'
defaults.NVCCArch = 'sm_20'
defaults.optimizeCCFlags = '-O3 -funroll-loops'
defaults.debugCCFlags = '-O0 -g -fbounds-check -Wunused-variable -Wunused-parameter'\
    ' -Wall -ftree-vrp -fsignaling-nans'
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
        setattr(defaults, key, ' '.join(value))

default_lapack_dir = os.path.join('usr', 'local', 'lib')
try:
    default_lapack_dir = os.path.join(os.environ['MKLROOT'], 'lib', 'intel64')
except KeyError:
    pass

config_options = [
    ('blas_lapack_libs',
        'Comma separated list of blas/lapack libraries to use for the various '
        'solvers, set blas_lapack_libs to the the list of libraries that should be '
        'passed to the linker, separated by commas, e.g. "lapack,blas" or '
        '"lapack,f77blas,cblas,atlas".',
        'mkl_rt,mkl_intel_lp64,mkl_core,mkl_gnu_thread,dl,mkl_mc,mkl_def'
     ),
    PathVariable('blas_lapack_dir',
                 """Directory containing the libraries specified by
                    'blas_lapack_libs'.""",
                 default_lapack_dir, PathVariable.PathAccept),
    ('toolchain',
        'The compiler tools to use for C/C++ code',
        'gnu'),
    ('NVCCFLAGS',
     'Compiler flags passed to the CUDA compiler, regardless of optimization level.',
     defaults.NVCCFLAGS),
    ('CCFLAGS',
     'Compiler flags passed to both the C and C++ compiler, regardless of '
     'optimization level',
     defaults.CCFlags),
    ('CXXFLAGS',
     'Compiler flags passed to only the C++ compiler, regardless of optimization '
     'level',
     defaults.CXXFlags),
    ('CFLAGS',
     'Compiler flags passed to only the C compiler, regardless of optimization '
     'level',
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
    ('N_RA', 'The size of the Rational Approximant for the Exponential Integrators.',
             '10'),
    BoolVariable(
        'SAME_IC', 'Use the same initial conditions (specified during mechanism '
                   'creation) during integration.', False),
    BoolVariable(
        'SHUFFLE', 'Shuffle the PaSR initial conditions.', False),
    BoolVariable(
        'PRECONDITION', 'Precondition (via clustering) the PaSR initial '
                        'conditions.', False),
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
        'FINITE_DIFFERENCE', 'Use a finite difference Jacobian (not recommended)',
        False),
    ('DIVERGENCE_WARPS', 'If specified, measure divergence in that many warps', '0'),
    ('CV_HMAX', 'If specified, the maximum stepsize for CVode', '0'),
    ('CV_MAX_STEPS', 'If specified, the maximum stepsize for CVode', '20000'),
    ('CONST_TIME_STEP', 'If specified, adaptive timestepping will be turned off '
     '(for logging purposes)', False),
    PathVariable(
        'python_cmd',
        'The python interpreter to use to generate python wrappers for '
        'accelerInt.  If not specified, the python interpreter used by '
        'scons will be used.', sys.executable, PathVariable.PathAccept)
]

opts.AddVariables(*config_options)
opts.Update(env)
opts.Save('accelerInt.conf', env)

if 'help' in COMMAND_LINE_TARGETS:
    # Print help about configuration options and exit.
    print("""
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
interface_dir = os.path.join(home, 'interface')
linalg_dir = os.path.join(home, 'linear_algebra')
generic_dir = os.path.join(home, 'generic')
radau2a_dir = os.path.join(home, 'radau2a')
rkf45_dir = os.path.join(home, 'rkf45')
exp_int_dir = os.path.join(home, 'exponential_integrators')
exp4_int_dir = os.path.join(home, 'exponential_integrators', 'exp4')
exprb43_int_dir = os.path.join(home, 'exponential_integrators', 'exprb43')
cvodes_dir = os.path.join(home, 'cvodes')
rk78_dir = os.path.join(home, 'rk78')
rkc_dir = os.path.join(home, 'rkc')
lib_dir = os.path.join(home, 'lib')
driver_dir = os.path.join(home, 'driver')

common_dir_list = [generic_dir, mech_dir, linalg_dir]

reg_count = ''
if build_cuda:
    try:
        with open(os.path.join(mech_dir, 'regcount'), 'r') as file:
            reg_count = file.readline().strip()
        NVCCFlags.append(['-maxrregcount {}'.format(reg_count)])
    except IOError:
        print('Register count not found. Using default.')

# add openmp
NVCCFlags.append(['-Xcompiler {}'.format(env['openmp_flags'])])
header_ext = {'c': '.h',
              'c++': '.hpp',
              'cuda': '.cuh'}

# link lines
LibDirs = listify(env['blas_lapack_dir'])
Libs += listify(env['blas_lapack_libs'])
if build_cuda:
    NVCCLinkFlags.append(
        [env['openmp_flags'], '-Xlinker -rpath {}/lib64'.format(env[
            'CUDA_TOOLKIT_PATH'])])

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
    with open(os.path.join(dir, 'solver_options{}'.format(
            header_ext[lang])), 'w') as file:
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

        /*! Solver timestep (may be used to force multiple timesteps per global integration step) */
        #define t_step ({})
        /*! Global integration timestep */
        #define end_time ({})

        /** type of rational approximant (n, n) */
        #define N_RA ({})

        """.format(env['t_step'], env['t_end'], env['N_RA']))

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


def get_env(save, defines):
    env = save.Clone()

    # update defines
    for key, value in defines.items():
        if key not in env:
            env[key] = []
        env[key].extend(listify(value))

    return env


platforms = ['cpu', 'opencl']
if build_cuda:
    platforms += ['cuda']

env_save = env.Clone()


def libname(node):
    """
    Get the library name from an SCons file node
    """

    filename = os.path.basename(node.rstr())
    # drop lib
    filename = filename[filename.index('lib') + len('lib'):]
    # drop exension
    filename = filename[:filename.rindex('.')]
    return filename


def add_libs_to_defines(libs, defines):
    libs = [libname(x) for x in libs]
    if 'LIBS' not in defines:
        defines['LIBS'] = []
    defines['LIBS'] += libs
    if 'LIBPATH' not in defines:
        defines['LIBPATH'] = []
    defines['LIBPATH'] += [lib_dir]
    if 'RPATH' not in defines:
        defines['RPATH'] = []
    defines['RPATH'] += [lib_dir]
    return defines


def build_lib(save, platform, defines, src, variant, target_base,
              extra_libs=[], extra_src=[]):
    extra = extra_libs[:]

    cdef = add_libs_to_defines(extra, defines.copy())
    env = get_env(save, cdef)

    if extra_src:
        env['extra_dirs'] = extra_src[:]

    # copy sconscript into dir
    shutil.copyfile(os.path.join(home, 'SConscript_base'),
                    os.path.join(src, 'SConscript'))

    # build integrator for this lib
    intlib = env.SConscript(os.path.join(src, 'SConscript'),
                            src_dir=src,
                            exports=['env', 'platform'])
    if not intlib:
        return None

    lib = env.SharedLibrary(target=target_base, source=intlib)
    lib = env.Install(lib_dir, lib)
    return lib


def build_core(save, platform, defines, variant):
    return build_lib(save, platform, defines, 'generic', variant, 'core')


def build_multitarget(save, platform, defines, libs, variant):
    return build_lib(env, platform, defines, interface_dir, variant,
                     'accelerint', libs)


def find_current_python(env):
    our_env = env.Clone()
    our_env['PATH'] = os.environ['PATH']
    if 'python_cmd' not in our_env:
        our_env['python_cmd'] = sys.executable
    return find_executable(our_env['python_cmd'], path=our_env['PATH'])


def run_with_our_python(env, target, source, action):
    # Test to see if we can import numpy and Cython
    script = textwrap.dedent("""\
        import sys
        print('{v.major}.{v.minor}'.format(v=sys.version_info))
        err = ''
        try:
            import numpy
            print(numpy.__version__)
        except ImportError as np_err:
            print('0.0.0')
            err += str(np_err) + '\\n'
        try:
            import Cython
            print(Cython.__version__)
        except ImportError as cython_err:
            print('0.0.0')
            err += str(cython_err) + '\\n'
        if err:
            print(err)
    """)

    try:
        env['python_cmd'] = find_current_python(env)
        info = getCommandOutput(env['python_cmd'], '-c', script).splitlines()
    except OSError as err:
        print('Error checking for Python:')
        print(err)
        sys.exit(err.output)
    except subprocess.CalledProcessError as err:
        print('Error checking for Python:')
        print(err, err.output)
        sys.exit(err.output)
    else:
        numpy_version = LooseVersion(info[1])
        cython_version = LooseVersion(info[2])

    missing = [x[0] for x in [('numpy', numpy_version), ('cython', cython_version)]
               if x[1] == LooseVersion('0.0.0')]
    if missing:
        print('ERROR: Could not import required packages ({}) the Python interpreter'
              ' {!r}. Did you mean to set the "python_cmd" option?"'.format(
                ', '.join(missing), env['python_cmd']))
        sys.exit(1)

    return env.Command(target=target,
                       source=source,
                       action=action.format(python=env['python_cmd']))


def build_wrapper(save, platform, defines, libs, variant):
    # problem definition, if available
    wrapper = build_lib(save, platform, defines, mech_dir, variant,
                        'accelerint_problem')
    if not wrapper:
        raise Exception('Cannot build wrapper as problem definition not found in '
                        '{}'.format(os.path.join(mech_dir, platform)))

    # and build wrapper
    env = get_env(save, defines)
    driver = os.path.join(driver_dir, 'setup.py')
    wrapper_py = run_with_our_python(env,
                                     target='pyccelerInt_cpu',
                                     source=[driver],
                                     action='{{python}} {} build_ext --inplace'
                                     .format(driver))

    env.Depends(wrapper_py, wrapper)
    return wrapper_py


def get_includes(platform, includes, new_defines={},
                 full_includes=[]):
    # include platform in path
    includes = [os.path.join(x, platform) for x in includes]
    # include any full paths
    includes += full_includes
    if platform in ['cpu', 'opencl']:
        new_defines['CPPPATH'] = includes[:]
    elif platform == 'cuda':
        new_defines['NVCC_INC_PATH'] = includes[:]
    else:
        raise NotImplementedError
    return new_defines


for p in platforms:
    new_defines = get_includes(p, [radau2a_dir, rk78_dir, rkc_dir, exp4_int_dir,
                                   exprb43_int_dir, exp_int_dir, cvodes_dir])
    core = build_core(env_save, p, new_defines, variant)

    # linear algebra
    new_defines = get_includes(p, [linalg_dir])
    linalg = build_lib(env_save, p, new_defines, linalg_dir,
                       variant, 'linalg', extra_libs=core)

    # radua
    new_defines = get_includes(p, [generic_dir, radau2a_dir, linalg_dir])
    radau = build_lib(env_save, p, new_defines, radau2a_dir,
                      variant, 'radau2a', extra_libs=core + linalg)

    # exponentials
    shared = os.path.join(exp_int_dir, 'shared')
    new_defines = get_includes(p, [generic_dir, env['fftw3_inc_dir'], exp_int_dir,
                                   linalg_dir], full_includes=[shared])
    new_defines['LIBPATH'] = [env['fftw3_lib_dir']]
    new_defines['LIBS'] = ['fftw3']
    exp = build_lib(env_save, p, new_defines, exp_int_dir,
                    variant, 'exp', extra_libs=core + linalg, extra_src=[shared])

    # exp4
    new_defines = get_includes(p,  [generic_dir, exp_int_dir, exp4_int_dir])
    new_defines['CPPDEFINES'] = ['EXP4']
    new_defines['NVCCDEFINES'] = ['EXP4']
    exp4 = build_lib(env_save, p, new_defines, exp4_int_dir,
                     variant, 'exp4', extra_libs=exp)

    # exprb43
    new_defines = get_includes(p,  [generic_dir, exp_int_dir, exprb43_int_dir])
    new_defines['LIBS'] = ['fftw3']
    new_defines['CPPDEFINES'] = ['RB43']
    new_defines['NVCCDEFINES'] = ['RB43']
    exprb43 = build_lib(env_save, p, new_defines, exprb43_int_dir,
                        variant, 'exprb43', extra_libs=exp)

    # rkc
    new_defines = get_includes(p,  [generic_dir, rkc_dir])
    rkc = build_lib(env_save, p, new_defines, rkc_dir,
                    variant, 'rkc', extra_libs=core)
    # cvodes
    new_defines = get_includes(p,  [generic_dir, cvodes_dir,
                                    env['sundials_inc_dir']])
    new_defines['LIBPATH'] = [env['sundials_lib_dir']]
    new_defines['LIBS'] = ['sundials_cvodes', 'sundials_nvecserial']
    cvodes = build_lib(env_save, p, new_defines, cvodes_dir,
                       variant, 'cvodes', extra_libs=core)

    # rk78
    new_defines = get_includes(p,  [generic_dir, rk78_dir, env['boost_inc_dir']])
    rk78 = build_lib(env_save, p, new_defines, rk78_dir, variant,
                     'rk78', extra_libs=core)

    # rkf45
    new_defines = get_includes(p,  [generic_dir, rkf45_dir])
    rkf45 = build_lib(env_save, p, new_defines, rkf45_dir, variant,
                      'rkf45', extra_libs=core)

    # add interface / problem definition
    new_defines = get_includes(p,  [generic_dir, radau2a_dir, rk78_dir, rkc_dir,
                                    exp4_int_dir, exprb43_int_dir, exp_int_dir,
                                    cvodes_dir, rkf45_dir])
    new_defines['LIBPATH'] = [env['sundials_lib_dir'], env['fftw3_lib_dir'], lib_dir]
    new_defines['LIBS'] = ['sundials_cvodes', 'sundials_nvecserial', 'fftw3']
    new_defines['RPATH'] = [lib_dir]

    # filter out non-existant
    vals = [rkc, rk78, radau, exp4, exprb43, cvodes, exp, linalg, core, rkf45]
    vals = [x for x in vals if x]
    vals = [y for x in vals for y in x]

    # add the multitarget
    target = build_multitarget(env_save, p, new_defines, vals, variant)

    # add an alias
    Alias(p, target)

    # and finally build wrapper
    defines = {}
    defines['RPATH'] = [lib_dir]
    defines['LIBPATH'] = [lib_dir]
    defines = add_libs_to_defines(vals, defines)
    wrapper = build_wrapper(env_save, p, defines, vals, variant)
    # and wrapper
    Alias(p + '-wrapper', wrapper)
