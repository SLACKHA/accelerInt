"""
SCons.Tool.cuda

CUDA Tool for SCons

"""

import os
import SCons.Action
import SCons.Builder
import SCons.Util
import SCons.Scanner


def CUDAObjCmdDefine(env):
    # create OBJ command
    # '$NVCC $NVCCPATH $NVCCFLAGS $NVCCDEFINES -dc -o $TARGET $SOURCES'
    command = '$NVCC'
    if 'NVCCPATH' in env:
        if not isinstance(env['NVCCPATH'], list):
            env['NVCCPATH'] = [env['NVCCPATH']]
        command += ' ' + ' '.join(['-I{}'.format(path) for path in env['NVCCPATH']])
    if 'NVCC_INC_PATH' in env:
        if not isinstance(env['NVCC_INC_PATH'], list):
            env['NVCC_INC_PATH'] = [env['NVCC_INC_PATH']]
        command += ' ' + ' '.join(['-I{}'.format(path) for path in env['NVCC_INC_PATH']])
    if 'NVCCFLAGS' in env:
        if not isinstance(env['NVCCFLAGS'], list):
            env['NVCCFLAGS'] = [env['NVCCFLAGS']]
        command += ' ' + ' '.join(['{}'.format(f) for f in env['NVCCFLAGS']])
    if 'NVCCDEFINES' in env:
        if not isinstance(env['NVCCDEFINES'], list):
            env['NVCCDEFINES'] = [env['NVCCDEFINES']]
        command += ' ' + ' '.join(['-D{}'.format(f) for f in env['NVCCDEFINES']])
    command += ' -dc -o $TARGET $SOURCES'
    env['NVCC_OBJ_CMD'] = command


def CUDADLinkCmdDefine(env, dname):
    # create OBJ command
    # NVCC_DLINK_CMD = '$NVCC $SOURCES $NVCCLIBPATH $NVCCLIBS $NVCCLINKFLAGS -dlink -o dlink.o'
    # NVCC_PROG_CMD = '$NLINK $NVCCLINKFLAGS $SOURCES dlink.o $NVCCLIBPATH $NVCCLIBS -o $TARGET'
    command = '$NVCC $SOURCES'
    if 'NVCCLIBPATH' in env:
        if not isinstance(env['NVCCLIBPATH'], list):
            env['NVCCLIBPATH'] = [env['NVCCLIBPATH']]
        command += ' ' + ' '.join(['-L{}'.format(path) for path in env['NVCCLIBPATH']])
    if 'NVCCLIBS' in env:
        if not isinstance(env['NVCCLIBS'], list):
            env['NVCCLIBS'] = [env['NVCCLIBS']]
        command += ' ' + ' '.join(['-l{}'.format(f) for f in env['NVCCLIBS']])
    command += ' -dlink -o ' + dname
    env['NVCC_DLINK_CMD'] = command


def CUDAProgCmdDefine(env):
    command = '$NLINK '
    if 'NVCCLINKFLAGS' in env:
        if not isinstance(env['NVCCLINKFLAGS'], list):
            env['NVCCLINKFLAGS'] = [env['NVCCLINKFLAGS']]
        command += ' ' + ' '.join(['{}'.format(f) for f in env['NVCCLINKFLAGS']])
    command += ' $SOURCES '
    if 'NVCCLIBPATH' in env:
        if not isinstance(env['NVCCLIBPATH'], list):
            env['NVCCLIBPATH'] = [env['NVCCLIBPATH']]
        command += ' ' + ' '.join(['-L{}'.format(path) for path in env['NVCCLIBPATH']])
    if 'NVCCLIBS' in env:
        if not isinstance(env['NVCCLIBS'], list):
            env['NVCCLIBS'] = [env['NVCCLIBS']]
        command += ' ' + ' '.join(['-l{}'.format(f) for f in env['NVCCLIBS']])
    if 'LIBPATH' in env:
        if not isinstance(env['LIBPATH'], list):
            env['LIBPATH'] = [env['LIBPATH']]
        command += ' ' + ' '.join(['-L{}'.format(path) for path in env['LIBPATH']])
    if 'LIBS' in env:
        if not isinstance(env['LIBS'], list):
            env['LIBS'] = [env['LIBS']]
        command += ' ' + ' '.join(['-l{}'.format(f) for f in env['LIBS']])
    command += ' -o $TARGET'
    env['NVCC_PROG_CMD'] = command


_cuda_dlink_builder = SCons.Builder.Builder(
    action=SCons.Action.Action('$NVCC_DLINK_CMD', ''),
    suffix='',
    src_suffix='$NVCC_OBJ_SUFFIX')
_cuda_prog_builder = SCons.Builder.Builder(
    action=SCons.Action.Action('$NVCC_PROG_CMD', ''),
    suffix='',
    src_suffix='$NVCC_OBJ_SUFFIX')
_cuda_obj_builder = SCons.Builder.Builder(
      action=SCons.Action.Action('$NVCC_OBJ_CMD', ''),
      suffix='$NVCC_OBJ_SUFFIX',
      src_suffix='$NVCC_SUFFIX')


def CUDAObject(env, src, target=None, *args, **kw):
    use_env = env
    name = str(src)
    if name.endswith('.c'):
        cenv = env.Clone()
        cenv['NVCCFLAGS'] += ['-Xcompiler -std=c99']
        CUDAObjCmdDefine(cenv)
        use_env = cenv

    CUDAObjCmdDefine(use_env)
    result = []
    if not target:
        suff = use_env.subst('$NVCC_SUFFIX')
        target = name[:name.index(suff)]
    obj = _cuda_obj_builder.__call__(use_env, target, src, **kw)
    result.extend(obj)
    # Add cleanup files
    env.Clean(obj, name)

    return result


def CUDADLink(env, target, source, *args, **kw):
    name = '{}_dlink.o'.format(target)
    CUDADLinkCmdDefine(env, name)
    if not source:
        source = target[:]
    if not SCons.Util.is_List(source):
        source = [source]
    result = []
    obj = _cuda_dlink_builder.__call__(env, name, source, **kw)
    result.append(obj)
    env.Clean(target, name)
    return result


def CUDAProgram(env, target, source, *args, **kw):
    CUDAProgCmdDefine(env)
    if not source:
        source = target[:]
    if not SCons.Util.is_List(source):
        source = [source]
    result = []
    obj = _cuda_prog_builder.__call__(env, target, source, **kw)
    result.append(obj)
    env.Clean(target, [str(s) for s in source])
    return result


def generate(env):
        # default compiler
        env['NVCC'] = 'nvcc'
        env['NLINK'] = env['CC']

        env.SetDefault(
          NVCC_SUFFIX='.cu',
          NVCC_OBJ_SUFFIX='.cu.o'
          )

        # default flags for the NVCC compiler
        env['NVCCFLAGS'] = ''
        # default NVCC commands
        env['NVCC_OBJ_CMD'] = '$NVCC $NVCC_INC_PATH $NVCCPATH $NVCCFLAGS $NVCCDEFINES -dc -o $TARGET $SOURCES'
        env['NVCC_DLINK_CMD'] = '$NVCC $SOURCES $NVCCLIBPATH $NVCCLIBS $NVCCLINKFLAGS -dlink -o dlink.o'
        env['NVCC_PROG_CMD'] = '$NLINK $NVCCLINKFLAGS $SOURCES dlink.o $NVCCLIBPATH $NVCCLIBS -o $TARGET'

        # helpers
        home = os.environ.get('HOME', '')
        programfiles = os.environ.get('PROGRAMFILES', '')
        homedrive = os.environ.get('HOMEDRIVE', '')

        # find CUDA Toolkit path and set CUDA_TOOLKIT_PATH
        try:
                cudaToolkitPath = env['CUDA_TOOLKIT_PATH']
        except:
                paths = [home + '/NVIDIA_CUDA_TOOLKIT',
                         home + '/Apps/NVIDIA_CUDA_TOOLKIT',
                         home + '/Apps/NVIDIA_CUDA_TOOLKIT',
                         home + '/Apps/CudaToolkit',
                         home + '/Apps/CudaTK',
                         '/usr/local/NVIDIA_CUDA_TOOLKIT',
                         '/usr/local/CUDA_TOOLKIT',
                         '/usr/local/cuda_toolkit',
                         '/usr/local/CUDA',
                         '/usr/local/cuda',
                         '/Developer/NVIDIA CUDA TOOLKIT',
                         '/Developer/CUDA TOOLKIT',
                         '/Developer/CUDA',
                         programfiles + 'NVIDIA Corporation/NVIDIA CUDA TOOLKIT',
                         programfiles + 'NVIDIA Corporation/NVIDIA CUDA',
                         programfiles + 'NVIDIA Corporation/CUDA TOOLKIT',
                         programfiles + 'NVIDIA Corporation/CUDA',
                         programfiles + 'NVIDIA/NVIDIA CUDA TOOLKIT',
                         programfiles + 'NVIDIA/NVIDIA CUDA',
                         programfiles + 'NVIDIA/CUDA TOOLKIT',
                         programfiles + 'NVIDIA/CUDA',
                         programfiles + 'CUDA TOOLKIT',
                         programfiles + 'CUDA',
                         homedrive + '/CUDA TOOLKIT',
                         homedrive + '/CUDA']
                pathFound = False
                for path in paths:
                        if os.path.isdir(path):
                                pathFound = True
                                print 'scons: CUDA Toolkit found in ' + path
                                cudaToolkitPath = path
                                break
                if not pathFound:
                    raise Exception("Cannot find the CUDA Toolkit path. Please modify your SConscript or add the path in cudaenv.py")
        env['CUDA_TOOLKIT_PATH'] = cudaToolkitPath

        # find CUDA SDK path and set CUDA_SDK_PATH
        try:
                cudaSDKPath = env['CUDA_SDK_PATH']
        except:
                paths = [home + '/NVIDIA_CUDA_SDK',  # i am just guessing here
                         home + '/Apps/NVIDIA_CUDA_SDK',
                         home + '/Apps/CudaSDK',
                         '/usr/local/NVIDIA_CUDA_SDK',
                         '/usr/local/CUDASDK',
                         '/usr/local/cuda_sdk',
                         '/usr/local/cuda/samples',
                         '/Developer/NVIDIA CUDA SDK',
                         '/Developer/CUDA SDK',
                         '/Developer/CUDA',
                         '/Developer/GPU Computing/C',
                         programfiles + 'NVIDIA Corporation/NVIDIA CUDA SDK',
                         programfiles + 'NVIDIA/NVIDIA CUDA SDK',
                         programfiles + 'NVIDIA CUDA SDK',
                         programfiles + 'CudaSDK',
                         homedrive + '/NVIDIA CUDA SDK',
                         homedrive + '/CUDA SDK',
                         homedrive + '/CUDA/SDK']
                pathFound = False
                for path in paths:
                        if os.path.isdir(path):
                                pathFound = True
                                print 'scons: CUDA SDK found in ' + path
                                cudaSDKPath = path
                                break
                if not pathFound:
                    raise Exception("Cannot find the CUDA SDK path. Please set env['CUDA_SDK_PATH'] to point to your SDK path")
        env['CUDA_SDK_PATH'] = cudaSDKPath

        env.AddMethod(CUDAObject, 'CUDAObject')
        env.AddMethod(CUDADLink, 'CUDADLink')
        env.AddMethod(CUDAProgram, 'CUDAProgram')

        # cuda libraries
        if env['PLATFORM'] == 'posix':
                cudaSDKSubLibDir = '/linux'
        elif env['PLATFORM'] == 'darwin':
                cudaSDKSubLibDir = '/darwin'
        else:
                cudaSDKSubLibDir = ''

        # add nvcc to PATH
        env.PrependENVPath('PATH', cudaToolkitPath + '/bin')

        # add required libraries
        env.Append(NVCCPATH=[cudaSDKPath + '/common/inc', cudaToolkitPath + '/include'])
        env.Append(NVCCLIBPATH=[cudaSDKPath + '/common/lib/linux/x86_64/' +
                                cudaSDKSubLibDir, cudaToolkitPath + '/lib64'])
        if 'NVCCLIBS' not in env:
            env.Append(NVCCLIBS=['cudart'])

        cuda_scan = SCons.Scanner.ClassicCPP('CScanner',
                                             '$NVCC_SUFFIX',
                                             'NVCC_INC_PATH',
                                             '^[ \t]*#[ \t]*(?:include|import)[ \t]*(<|")([^>"]+)(>|")')
        env.Append(SCANNERS=cuda_scan)


def exists(env):
    return env.Detect('nvcc')
