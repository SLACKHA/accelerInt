##
# \file
# \brief Runs the integrators for van der Pol problem

import numpy as np
import sys
import os
sys.path.insert(0, os.getcwd())
import pyccelerInt_cpu as pycel

# set number of IVPs to solve
num = 5000

# create state vectors
phi = 2 * np.random.random((num, 2))

# set parameters
params = 5 * np.random.random(num)

# create the integrator
integrator = pycel.PyIntegrator(pycel.IntegratorType.CVODES, 2, 40)

# and integrate
integrator.integrate(num, 0., 100., phi.flatten('F'), params.flatten('F'))
