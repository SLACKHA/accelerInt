import os
import sys

import cantera as ct
import numpy as np
from pyjac import __version_info__, create_jacobian
from pyjac.core.enum_types import KernelType
from pyjac.pywrap import pywrap
assert(__version_info__[0] >= 2), 'Only pyJac-V2+ supports OpenCL execution.'

# and add the path to this directory to get the pyjac module
path = os.path.dirname(__file__)
sys.path.insert(1, path)


def run(pycel, num, num_threads, itype, tf, options, reuse, plt):
    """ run the pyJac ignition problem"""

    # create source-rate evaluation
    width = options.vector_size() if options.vector_size() else \
        options.block_size()
    if width <= 1:
        width = None

    src_path = os.path.join(path, 'src')
    obj_path = os.path.join(path, 'obj')
    ktype = KernelType.species_rates if itype == pycel.IntegratorType.RKF45 else \
        KernelType.jacobian
    if not reuse:
        create_jacobian('opencl',
                        mech_name=os.path.join(path, os.path.pardir, 'h2.cti'),
                        width=width, build_path=src_path, last_spec=None,
                        kernel_type=ktype,
                        platform=options.platform(),
                        data_order=options.order(),
                        explicit_simd=not options.block_size() and width)
        # and compile to get the kernel object
        pywrap('opencl', src_path, obj_path, path, obj_path,
               options.platform(), ktype=ktype)

    # create initial state
    import pyjac_ocl as pyjac
    # create a kernel
    if ktype == KernelType.species_rates:
        knl = pyjac.PySpecies_RatesKernel(num, num_threads)
    else:
        knl = pyjac.PyJacobianKernel(num, num_threads)

    # equations are T, V, N_{0}... N_{nsp-1}
    neq = knl.num_species() + 1

    # create state vectors
    P0 = 101325  # Pa
    T0 = 1000  # K
    V0 = 1  # m^3
    phi = np.zeros((num, neq), dtype=np.float64)
    phi[:, 0] = T0
    phi[:, 1] = V0

    # get species names
    species = knl.species_names()

    H2 = species.index('H2')
    O2 = species.index('O2')

    # determine species moles

    # PV = nRT
    # -> n = PV / RT
    n = P0 * V0 / (ct.gas_constant * T0)
    # moles in the stoichiometric eqn [2 H2 + O2 + 3.76 N2]
    n_stoich = 2 + 1 + 3.76

    # convert stoichometric eqn to system moles
    phi[:, H2 + 2] = (2 / n_stoich) * n
    phi[:, O2 + 2] = (1 / n_stoich) * n

    # set pressure
    pressure = np.full(num, 101325, dtype=np.float64)

    # get the working memory
    rwk_size = knl.required_working_memory()

    if options.vector_size() or options.block_size():
        # pyJac returns the complete (vectorized) memory req's, but we need
        # to pass in the unvectorized size
        vw = options.vector_size() if options.vector_size() else options.block_size()
        rwk_size /= vw

    # create a simple wrapper
    with open(os.path.join(src_path, 'dydt.cl'), 'w') as file:
        file.write(
            """
            #include "species_rates.oclh"
            void dydt (__private __ValueType const t,
                       __global __ValueType const * __restrict__ param,
                       __global __ValueType const * __restrict__ y,
                       __global __ValueType * __restrict__ dy,
                       __global __ValueType * __restrict__ rwk)
           {
              species_rates(0, param, y, dy, rwk);
           }
           """)
    with open(os.path.join(src_path, 'jac.cl'), 'w') as file:
        file.write(
            """
            #include "jacobian.oclh"
            void jacob (__private __ValueType const t,
                        __global __ValueType const * __restrict__ param,
                        __global __ValueType const * __restrict__ y,
                        __global __ValueType * __restrict__ jac,
                        __global __ValueType * __restrict__ rwk)
           {
              jacobian(0, param, y, jac, rwk);
           }
           """)

    files = [os.path.join(src_path, 'species_rates.ocl'),
             os.path.join(src_path, 'chem_utils.ocl'),
             os.path.join(src_path, 'dydt.cl')]

    if ktype == KernelType.jacobian:
        files += [os.path.join(src_path, 'jac.cl'),
                  os.path.join(src_path, 'jacobian.ocl')]

    # create ivp
    # Note: we need to pass the full paths to the PyIVP such that accelerInt
    # can find our kernel files
    ivp = pycel.PyIVP(files,
                      rwk_size,
                      include_paths=[src_path])

    # create the integrator
    integrator = pycel.PyIntegrator(itype, neq, num_threads, ivp, options)

    # and integrate
    phi_c = phi.flatten(options.order())
    time = integrator.integrate(num, 0., tf, phi_c,
                                pressure.flatten(options.order()), step=1e-6)
    # and get final state
    phi = phi_c.reshape(phi.shape, order=options.order())

    print('Integration completed in {} (ms)'.format(time))

    # get output
    t, phip = integrator.state()
    if plt:
        plt.plot(t, phip[0, 0, :], label='pyJac')
        plt.ylim(np.min(phip[0, 0, :]) * 0.95, np.max(phip[0, 0, :]) * 1.05)
        plt.title('H2 Ignition')

        # plot same problem in CT for comparison
        gas = ct.Solution(os.path.join(path, os.path.pardir, 'h2.cti'))
        gas.TPX = 1000, 101325, 'H2:2, O2:1, N2:3.76'
        reac = ct.IdealGasConstPressureReactor(gas)
        net = ct.ReactorNet([reac])
        t = [net.time]
        T = [reac.thermo.T]
        while net.time < tf:
            net.advance(net.time + 1e-6)
            t.append(net.time)
            T.append(reac.thermo.T)

        plt.plot(t, T, '--', label='Cantera')

        plt.legend(loc=0)
        plt.show()

    # check that answers from all threads match
    assert np.allclose(phi[:, 0], phi[0, 0]), np.where(
        ~np.isclose(phi[:, 0], phi[0, 0]))
    assert np.allclose(phi[:, 1], phi[0, 1]), np.where(
        ~np.isclose(phi[:, 1], phi[0, 1]))
