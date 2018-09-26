##
# \file
# \brief Runs the integrators for van der Pol problem

import numpy as np
import sys
import multiprocessing
import os
import argparse
sys.path.insert(0, os.getcwd())
import pyccelerInt_cpu as pycel
np.random.seed(0)


def run(num, num_threads, itype):
    # number of equations
    neq = 2

    # create state vectors
    phi = 2 * np.random.random((num, 2))

    # set parameters
    params = 5 * np.random.random(num)

    # create options
    options = pycel.PySolverOptions(itype, atol=1e-15,
                                    rtol=1e-10, logging=True)

    # create the integrator
    integrator = pycel.PyIntegrator(itype, neq,
                                    num_threads, options)

    # and integrate
    time = integrator.integrate(num, 0., 10., phi.flatten('F'), params.flatten('F'),
                                step=1.)

    print('Integration completed in {} (ms)'.format(time))

    # get output
    t, phi = integrator.state()
    print(phi)


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
                        default=pycel.IntegratorType.CVODES,
                        help='The integrator type to uses [default CVODES]')

    args = parser.parse_args()
    int_type = next(x for x in pycel.IntegratorType if args.int_type in str(x))
    print('Integrating {} IVPs with method {}, and {} threads...'.format(
        args.num_ivp, args.num_threads, int_type))
    run(args.num_ivp, args.num_threads, int_type)
