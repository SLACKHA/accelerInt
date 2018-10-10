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

sys.path.insert(0, os.getcwd())
import pyccelerInt_ocl as pycel  # noqa
np.random.seed(0)


def run(num, num_threads, itype, tf, options):
    # number of equations
    neq = 2

    # create state vectors
    phi = 2 * np.zeros((num, 2), dtype=np.float64)
    phi[:, 0] = 2
    phi[:, 1] = 0

    # set parameters
    params = np.zeros(num, dtype=np.float64)
    params[:] = 1000

    # create ivp
    # Note: we need to pass the full paths to the PyIVP such that accelerInt
    # can find our kernel files
    path = os.path.dirname(__file__)
    ivp = pycel.PyIVP([os.path.join(path, 'dydt.cl')], 0)

    # create the integrator
    integrator = pycel.PyIntegrator(itype, neq,
                                    num_threads, ivp, options)

    # and integrate
    phi_c = phi.flatten(options.order())
    time = integrator.integrate(num, 0., tf, phi_c,
                                params.flatten(options.order()), step=1.)
    # and get final state
    phi = phi_c.reshape(phi.shape, order=options.order())

    print('Integration completed in {} (ms)'.format(time))

    # get output
    t, phip = integrator.state()
    if plt:
        plt.plot(t, phip[0, 0, :], label='y1')
        plt.plot(t, phip[0, 1, :], label='y2')
        plt.ylim(np.min(phip[0, 0, :]) * 1.05, np.max(phip[0, 0, :]) * 1.05)
        plt.legend(loc=0)
        plt.title('van der Pol equation')
        plt.show()

    # check that answers from all threads match
    assert np.allclose(phi[:, 0], phi[0, 0]), np.where(
        ~np.isclose(phi[:, 0], phi[0, 0]))
    assert np.allclose(phi[:, 1], phi[0, 1]), np.where(
        ~np.isclose(phi[:, 1], phi[0, 1]))


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser('example.py - run the van der Pol accelerInt '
                                     'example')
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
                        type=pycel.IntegratorType,
                        default=pycel.IntegratorType.RKF45,
                        help='The integrator type to use [default RKF45]')

    parser.add_argument('-v', '--vectorSize',
                        type=vector_width,
                        default=8,
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
                        default='',
                        help='The OpenCL platform to use, (e.g., Intel, NVIDIA, '
                             'etc.)')

    parser.add_argument('-d', '--device_type',
                        type=pycel.DeviceType,
                        default=pycel.DeviceType.DEFAULT,
                        help='The device type to use (e.g., CPU, GPU, ACCELERATOR).')

    parser.add_argument('-tf', '--end_time',
                        type=float,
                        default=2000.,
                        help='The simulation end-time.')

    args = parser.parse_args()
    assert not (args.vectorSize and args.blockSize), (
        'Cannot specify vectorSize and blockSize concurrently')

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

    print('Integrating {} IVPs with method {}, and {} threads...'.format(
        args.num_ivp, args.int_type, args.num_threads))
    run(args.num_ivp, args.num_threads, args.int_type, args.end_time, options)
