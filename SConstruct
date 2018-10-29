#! /usr/bin/env python3
# A SConstruct file for accerlerInt, heavily adapted from Cantera

from __future__ import print_function

import os
import sys
import platform
import shutil
import subprocess
import textwrap
from string import Template
from warnings import warn
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
    PathVariable(
        'opencl_inc_dir',
        'The path to the OpenCL header directory to use.', '',
        PathVariable.PathAccept),
    PathVariable(
        'opencl_lib_dir',
        'The path to the OpenCL library.', '',
        PathVariable.PathAccept),
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
        'VERBOSE', 'More verbose debugging statements.', False),
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
        'scons will be used.', sys.executable, PathVariable.PathAccept),
    PathVariable(
        'install_dir',
        'Directory to install the compiled libraries to. If not specifed, install '
        'to ./lib/', os.path.join(home, 'lib'), PathVariable.PathAccept)
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
ros_dir = os.path.join(home, 'ros')
exp_int_dir = os.path.join(home, 'exponential_integrators')
exp4_int_dir = os.path.join(home, 'exponential_integrators', 'exp4')
exprb43_int_dir = os.path.join(home, 'exponential_integrators', 'exprb43')
cvodes_dir = os.path.join(home, 'cvodes')
rk78_dir = os.path.join(home, 'rk78')
rkc_dir = os.path.join(home, 'rkc')
lib_dir = env['install_dir']
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
env['CPPDEFINES'] = []

variant = 'release' if not env['DEBUG'] else 'debug'
env['variant'] = variant

env['CPPPATH'] = common_dir_list
if 'NVCC_INC_PATH' not in env:
    env['NVCC_INC_PATH'] = []
if build_cuda:
    env['NVCC_INC_PATH'] += common_dir_list
if not (env['opencl_inc_dir'] or env['opencl_lib_dir']):
    print('OpenCL not specified, no OpenCL integrators will be built...')
env['OCL_INC_DIR'] = listify(env['opencl_inc_dir'])
env['OCL_LIB_DIR'] = listify(env['opencl_lib_dir'])


def config_error(message):
    print('ERROR:', message)
    if env['VERBOSE']:
        print('*' * 25, 'Contents of config.log:', '*' * 25)
        print(open('config.log').read())
        print('*' * 28, 'End of config.log', '*' * 28)
    else:
        print("See 'config.log' for details.")
    sys.exit(1)


def get_expression_value(includes, expression):
    s = ['#include ' + i for i in includes]
    s.extend(('#define Q(x) #x',
              '#define QUOTE(x) Q(x)',
              '#include <iostream>',
              '#ifndef SUNDIALS_PACKAGE_VERSION',  # name change in Sundials >= 3.0
              '#define SUNDIALS_PACKAGE_VERSION SUNDIALS_VERSION',
              '#endif',
              'int main(int argc, char** argv) {',
              '    std::cout << %s << std::endl;' % expression,
              '    return 0;',
              '}\n'))
    return '\n'.join(s)


def get_env(save, defines):
    env = save.Clone()

    # update defines
    for key, value in defines.items():
        if key not in env:
            env[key] = []
        env[key].extend(listify(value))

    return env

######################
#    Configuration   #
######################

# determine sundials version

sun_env = Environment(tools=['default'])
sun_env['CCFLAGS'] = ['-I{}'.format(env['sundials_inc_dir'])]
conf = Configure(sun_env)
ret, env['SUNDIALS_VERSION'] = conf.TryRun(
    get_expression_value(['"sundials/sundials_config.h"'],
                         'QUOTE(SUNDIALS_PACKAGE_VERSION)'), '.cpp')
env['SUNDIALS_VERSION'] = env['SUNDIALS_VERSION'].strip()
if env['SUNDIALS_VERSION'].startswith('"') and env['SUNDIALS_VERSION'].endswith('"'):
    env['SUNDIALS_VERSION'] = env['SUNDIALS_VERSION'][1:-1]
if ret == 0:
    config_error('Could not determine sundials version!')
print('Using Sundials version {}'.format(env['SUNDIALS_VERSION']))

# determine version
if LooseVersion(env['SUNDIALS_VERSION']) > LooseVersion('3.0'):
    cvodes_libs = ['sundials_cvodes', 'sundials_nvecserial',
                   'sundials_sunlinsollapackdense']
    env['CPPDEFINES'] += ['NEW_SUNDIALS']
else:
    cvodes_libs = ['sundials_cvodes', 'sundials_nvecserial']


platforms = ['cpu']
if env['OCL_INC_DIR']:
    platforms += ['opencl']
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
        return []

    lib = env.SharedLibrary(target=target_base + '_' + platform, source=intlib)
    ilib = env.Install(lib_dir, lib)
    return ilib


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
    python = find_executable(our_env['python_cmd'], path=our_env['PATH'])
    if python is None:
        raise Exception('Critical error: python_cmd ({}) not found, possibly moved '
                        'or deleted.'.format(our_env['python_cmd']))
    return python


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


def build_wrapper(save, platform, defines, libs, variant, multi):
    # problem definition, if available
    wrapper = build_lib(save, platform, defines, mech_dir, variant,
                        'accelerint_problem')
    if not wrapper:
        warn('Skipping {}-wrapper as problem definition not found in '
             '{}'.format(platform, os.path.join(mech_dir, platform)))
        return []

    short_names = {'cpu': 'cpu',
                   'opencl': 'ocl'}
    # and build wrapper
    env = get_env(save, defines)
    # add dependecy to multitarget
    env.Depends(wrapper, multi)
    driver = os.path.join(driver_dir, platform, 'setup.py.in')
    with open(driver, 'r') as file:
        kwargs = {}
        if platform == 'opencl':
            kwargs['cl_path'] = env['opencl_inc_dir']
        driver = Template(file.read()).substitute(libdir=lib_dir, **kwargs)
    dfile = os.path.join(driver_dir, platform, 'setup.py')
    with open(dfile, 'w') as file:
        file.write(driver)
    wrapper_py = run_with_our_python(env,
                                     target='pyccelerInt_{}'.format(
                                        short_names[platform]),
                                     source=[dfile],
                                     action='{{python}} {} build_ext --inplace'
                                     .format(dfile))

    env.Depends(wrapper_py, wrapper)
    return wrapper_py


def get_includes(platform, includes, exact_includes=[]):
    ndef = {}
    # include platform in path
    includes = [os.path.join(x, platform) for x in includes]
    if exact_includes:
        includes.extend(exact_includes)
    if platform in ['cpu', 'opencl']:
        ndef['CPPPATH'] = includes[:]
        if platform == 'opencl':
            ndef['CPPPATH'] += env['OCL_INC_DIR']
    elif platform == 'cuda':
        ndef['NVCC_INC_PATH'] = includes[:]
    else:
        print('Platform {} not implemented'.format(platform))
        raise NotImplementedError
    return ndef


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
    new_defines = get_includes(p, [generic_dir, exp_int_dir,
                                   linalg_dir, shared], exact_includes=[
                                   env['fftw3_inc_dir']])
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
    new_defines = get_includes(p,  [generic_dir, cvodes_dir], exact_includes=[
        env['sundials_inc_dir']])
    new_defines['LIBPATH'] = [env['sundials_lib_dir']]
    new_defines['LIBS'] = cvodes_libs[:]
    cvodes = build_lib(env_save, p, new_defines, cvodes_dir,
                       variant, 'cvodes', extra_libs=core)

    # rk78
    new_defines = get_includes(p,  [generic_dir, rk78_dir],
                               exact_includes=[env['boost_inc_dir']])
    rk78 = build_lib(env_save, p, new_defines, rk78_dir, variant,
                     'rk78', extra_libs=core)

    # rkf45
    new_defines = get_includes(p,  [generic_dir, rkf45_dir])
    rkf45 = build_lib(env_save, p, new_defines, rkf45_dir, variant,
                      'rkf45', extra_libs=core)

    new_defines = get_includes(p,  [generic_dir, ros_dir])
    ros = build_lib(env_save, p, new_defines, ros_dir, variant,
                    'ros', extra_libs=core)

    # add interface / problem definition
    new_defines = get_includes(p,  [generic_dir, radau2a_dir, rk78_dir, rkc_dir,
                                    exp4_int_dir, exprb43_int_dir, exp_int_dir,
                                    cvodes_dir, rkf45_dir, ros_dir],
                               exact_includes=[env['sundials_inc_dir']])
    new_defines['LIBPATH'] = [env['sundials_lib_dir'], env['fftw3_lib_dir'], lib_dir]
    new_defines['LIBS'] = cvodes_libs[:] + ['fftw3']
    new_defines['RPATH'] = [env['sundials_lib_dir'], env['fftw3_lib_dir'], lib_dir]

    if p == 'opencl':
        new_defines['LIBPATH'] += env['OCL_LIB_DIR']
        new_defines['RPATH'] += env['OCL_LIB_DIR']
        new_defines['LIBS'] += ['OpenCL']

    # filter out non-existant
    vals = [rkc, rk78, radau, exp4, exprb43, cvodes, exp, linalg, core, rkf45, ros]
    vals = [x for x in vals if x]
    vals = [y for x in vals for y in x]

    # add the multitarget
    target = build_multitarget(env_save, p, new_defines, vals, variant)

    # add an alias
    Alias(p, target)

    # and finally build wrapper
    new_defines = get_includes(p, [generic_dir])
    new_defines['RPATH'] = [lib_dir]
    new_defines['LIBPATH'] = [lib_dir]
    new_defines = add_libs_to_defines(vals, new_defines)
    wrapper = build_wrapper(env_save, p, new_defines, vals, variant, target)
    if wrapper:
        # and wrapper
        Alias(p + '-wrapper', wrapper)
