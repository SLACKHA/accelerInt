##
# \file
# \brief Generates initial conditions file for van der Pol problem
#
# This file can be executed to generate and write initial conditions data file
# for the van der Pol example.
# Depends on numpy

import numpy as np

# set number of IVPs to solve
num = 5000

# create data
ICs = 2 * np.random.random((num, 2))

# set parameters
params = 5 * np.random.random((num, 1))

# in order to conform to the current read_initial_conditions format, we need
# to add a dummy 'time' variable
time = np.zeros((num, 1))

# finally, we need to concatenate these variables and write to file

# again, the current read_initial_conditions format is optimized for pyJac
# and expects the state vector to be stored in the form:
#
# t, y[0], param, y[1]
#
# This is accomplished via concatentate and some reshapes to get the IC slices
# as 2-D arrays
#
# This format will be updated in future releases

out_arr = np.concatenate(
    (time, np.reshape(ICs[:, 0], (-1, 1)), params,
        np.reshape(ICs[:, 1], (-1, 1))), axis=1)

# and save to file
out_arr.tofile('ign_data.bin')
