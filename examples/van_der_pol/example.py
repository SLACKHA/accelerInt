##
# \file
# \brief Runs the integrators for van der Pol problem

import numpy as np
import sys
import os
sys.path.insert(0, os.getcwd())
import pyccelerInt_cpu as pycel

# set number of IVPs to solve
num = 100

# number of threads
num_threads = 4

# number of equations
neq = 2

# create state vectors
phi = 2 * np.random.random((num, 2))

# set parameters
params = 5 * np.random.random(num)

# create the integrator
integrator = pycel.PyIntegrator(pycel.IntegratorType.CVODES, neq, num_threads)

# and integrate
time = integrator.integrate(num, 0., 10., phi.flatten('F'), params.flatten('F'),
                            step=1.)

print('Integration completed in {} (ms)'.format(time))
