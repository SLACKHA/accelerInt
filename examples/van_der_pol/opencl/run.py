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
import pyccelerInt_ocl as pycel
np.random.seed(0)


def run(num, num_threads, itype):
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

    # create options
    options = pycel.PySolverOptions(itype, vectorSize=8,
                                    atol=1e-10, rtol=1e-6, logging=True,
                                    order='F')

    # create the integrator
    integrator = pycel.PyIntegrator(itype, neq,
                                    num_threads, ivp, options)

    # and integrate
    time = integrator.integrate(num, 0., 2000., phi.flatten('F'),
                                params.flatten('F'), step=1.)

    print('Integration completed in {} (ms)'.format(time))

    # get output
    t, phi = integrator.state()
    if plt:
        plt.plot(t, phi[0, 0, :], label='y1')
        plt.plot(t, phi[0, 1, :], label='y2')
        plt.legend(loc=0)
        plt.title('van der Pol equation')
        plt.show()


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
                        type=str,
                        default='RKF45',
                        help='The integrator type to use [default RKF45]')

    args = parser.parse_args()
    int_type = next(x for x in pycel.IntegratorType if args.int_type in str(x))
    print('Integrating {} IVPs with method {}, and {} threads...'.format(
        args.num_ivp, int_type, args.num_threads))
    run(args.num_ivp, args.num_threads, int_type)
