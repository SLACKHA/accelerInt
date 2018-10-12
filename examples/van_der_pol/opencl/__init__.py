##
# \file
# \brief Runs the integrators for van der Pol problem

import numpy as np
import os


def run(pycel, num, num_threads, itype, tf, options, reuse, plt):
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
    ivp = pycel.PyIVP([os.path.join(path, 'dydt.cl')], 0, [])

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
