##
# \file
# \brief Runs the integrators for van der Pol problem

import numpy as np
import sys
import multiprocessing
import os
import argparse

try:
    import matplotlib.pyplot as plt
except ImportError:
    print('Matplotlib not found, not plotting...')
    plt = None

# add runtime path to find the pycelerrint module
sys.path.insert(0, os.getcwd())
import pyccelerInt_ocl as pycel  # noqa

# and add the path to this directory to get the pyjac module
path = os.path.dirname(__file__)
sys.path.insert(1, path)

np.random.seed(0)


def vector_width(v):
    def __raise():
        raise argparse.ArgumentError('Specified vector-width: {} is invalid'.format(
            v))
    try:
        v = int(v)
        if v not in [1, 2, 3, 4, 8, 16]:
            __raise()
        return v
    except ValueError:
        __raise()


def block_size(b):
    def __raise():
        raise argparse.ArgumentError('Specified block size: {} is invalid'.format(
            b))
    try:
        b = int(b)
        # ensure power of 2
        if not (b & (b - 1)) == 0:
            __raise()
        return b
    except ValueError:
        __raise()


def integrator_type(it):
    int_type = next((x for x in pycel.IntegratorType if it in str(x)),
                    None)
    if int_type is None:
        raise argparse.ArgumentError(
            'Integrator type: {} is invalid, possible choices {{{}}}'.format(
                int_type, ', '.join([str(x) for x in pycel.IntegratorType])))
    return int_type


if __name__ == '__main__':
    parser = argparse.ArgumentParser('example.py - run the OpenCL examples')
    parser.add_argument('-c', '--case',
                        choices=['vdp', 'pyjac'],
                        help='The example to run, currently only the van der Pol '
                             'problem and pyJac are implemented.')
    parser.add_argument('-ni', '--num_ivp',
                        type=int,
                        default=100,
                        help='The number of IVPs to solve [default: 100].')

    parser.add_argument('-nt', '--num_threads',
                        type=int,
                        default=multiprocessing.cpu_count(),
                        help='The number of threads to use during solution '
                             '[default: # of logical cores].')

    parser.add_argument('-it', '--int_type',
                        type=integrator_type,
                        default='RKF45',
                        help='The integrator type to use [default RKF45]')

    parser.add_argument('-v', '--vectorSize',
                        type=vector_width,
                        default=0,
                        help='The SIMD vector-width to use [CPU].  '
                             'Exclusive with --blockSize.')

    parser.add_argument('-b', '--blockSize',
                        type=block_size,
                        default=0,
                        help='The implicit-SIMD work-group size to use [GPU].  '
                             'Exclusive with --vectorSize.')

    parser.add_argument('-q', '--useQueue',
                        default=True,
                        action='store_true',
                        dest='useQueue',
                        help='Use the queue-based integration drivers.')

    parser.add_argument('-s', '--useStatic',
                        default=True,
                        action='store_false',
                        dest='useQueue',
                        help='Use the static scheduling based integration drivers.')

    parser.add_argument('-o', '--order',
                        choices=['C', 'F'],
                        default='F',
                        help='The data-ordering, row-major ("C") or column-major '
                             '("F").')

    parser.add_argument('-p', '--platform',
                        type=str,
                        required=True,
                        help='The OpenCL platform to use, (e.g., Intel, NVIDIA, '
                             'etc.)')

    parser.add_argument('-d', '--device_type',
                        type=pycel.DeviceType,
                        default=pycel.DeviceType.DEFAULT,
                        help='The device type to use (e.g., CPU, GPU, ACCELERATOR).')

    parser.add_argument('-tf', '--end_time',
                        type=float,
                        default=None,  # 1ms
                        help='The simulation end-time.')

    parser.add_argument('-r', '--reuse',
                        action='store_true',
                        default=False,
                        help='Reuse the previously generated pyJac code / library.')

    args = parser.parse_args()
    assert not (args.vectorSize and args.blockSize), (
        'Cannot specify vectorSize and blockSize concurrently')

    end_time = args.end_time
    if not end_time:
        end_time = 2000 if args.case == 'vdp' else 1e-3

    # create the options
    options = pycel.PySolverOptions(args.int_type,
                                    vectorSize=args.vectorSize,
                                    blockSize=args.blockSize,
                                    use_queue=args.useQueue,
                                    order=args.order,
                                    platform=args.platform,
                                    deviceType=args.device_type,
                                    atol=1e-10, rtol=1e-6, logging=True,
                                    maxIters=1e6)

    if args.case == 'vdp':
        from examples.van_der_pol.opencl import run
    else:
        from examples.pyJac.opencl import run

    print('Integrating {} IVPs with method {}, and {} threads...'.format(
        args.num_ivp, args.int_type, args.num_threads))
    run(pycel, args.num_ivp, args.num_threads, args.int_type, end_time,
        options, reuse=args.reuse, plt=plt)
