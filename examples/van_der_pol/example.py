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


def run(num, num_threads):
    # number of equations
    neq = 2

    # create state vectors
    phi = 2 * np.random.random((num, 2))

    # set parameters
    params = 5 * np.random.random(num)

    # create options
    options = pycel.PySolverOptions(pycel.IntegratorType.CVODES, atol=1e-15,
                                    rtol=1e-10, logging=True)

    # create the integrator
    integrator = pycel.PyIntegrator(pycel.IntegratorType.CVODES, neq, num_threads,
                                    options)

    # and integrate
    time = integrator.integrate(num, 0., 10., phi.flatten('F'), params.flatten('F'),
                                step=1.)

    print('Integration completed in {} (ms)'.format(time))

    # get output
    phi = integrator.state(11)
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

    args = parser.parse_args()
    run(args.num_ivp, args.num_threads)
