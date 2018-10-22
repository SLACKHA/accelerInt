##
# \file
# \brief Runs the integrators for van der Pol problem

import os

import cantera as ct
import numpy as np
from pyjac import __version_info__, create_jacobian
from pyjac.core.enum_types import KernelType, JacobianFormat
from pyjac.pywrap import pywrap

from pyccelerInt import Problem, get_plotter

assert(__version_info__[0] >= 2), 'Only pyJac-V2+ supports OpenCL execution.'


def import_pyjac(platform):
    try:
        if platform == 'opencl':
            import pyjac_ocl as pj  # noqa
            return pj
        elif platform == 'c':
            import pyjac_cpu as pj  # noqa
            return pj
        else:
            raise Exception('Language {} not recognized!'.format(platform))
    except ImportError:
        raise Exception('pyccelerInt wrapper for platform: {} could not be '
                        'imported (using path {})'.format(
                            platform, os.getcwd()))


class Ignition(Problem):
    """
    An implementation of a constant pressure / volume homogenous ignition problem
    for pyccelerInt, utilizing source rates and Jacobian's from pyJac.
    """

    def __init__(self, platform, options, reuse=False, conp=True):
        """
        Initialize the problem.

        Parameters
        ----------
        platform: ['opencl', 'c']
            The runtime platform to use for the problem
        options: :class:`pyccelerint/PySolverOptions`
            The solver options to use
        reuse: bool [False]
            If true, reuse any previously generated code / modules
        conp: bool [True]
            If true, use the constant pressure assumption, else constant volume
        """

        self.conp = conp
        path = os.path.abspath(os.path.dirname(__file__))
        super(Ignition, self).__init__(platform, options, path, reuse=reuse)

        # build pyjac module
        # create source-rate evaluation
        width = options.vector_size() if options.vector_size() else \
            options.block_size()
        if width <= 1:
            width = None

        obj_path = os.path.join(self.dir, 'obj')
        ktype = KernelType.jacobian
        if not self.reuse:
            create_jacobian(self.platform,
                            mech_name=os.path.join(self.dir, 'h2.cti'),
                            width=width, build_path=self.src_path, last_spec=None,
                            kernel_type=ktype,
                            platform=options.platform(),
                            data_order=options.order(),
                            jac_format=JacobianFormat.full,
                            explicit_simd=not options.block_size() and width)
            # and compile to get the kernel object
            pywrap(self.platform, self.src_path, obj_path, os.getcwd(), obj_path,
                   options.platform(), ktype=ktype)

            # create a simple wrapper to join
            if platform == 'opencl':
                with open(os.path.join(self.src_path, 'dydt.cl'), 'w') as file:
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
                with open(os.path.join(self.src_path, 'jac.cl'), 'w') as file:
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

        # create initial state
        pj = import_pyjac(self.platform)
        # create a kernel -- note that we supply 1 IVP / 1 thread here such that we
        # can properly scale the required memory
        self.knl = pj.PyJacobianKernel(1, 1)

        # equations are T, V, N_{0}... N_{nsp-1}
        self.neq = self.knl.num_species() + 1

        # get the working memory
        self.rwk_size = self.knl.required_working_memory()
        if options.vector_size() or options.block_size():
            # pyJac returns the complete (vectorized) memory req's, but we need
            # to pass in the unvectorized size
            vw = options.vector_size() if options.vector_size() else \
                options.block_size()
            self.rwk_size /= vw

    @property
    def src_path(self):
        return os.path.join(os.getcwd(), 'examples_working')

    def setup(self, num, options):
        """
        Do any setup work required for this problem, initialize input arrays,
        generate code, etc.

        Parameters
        ----------
        num: int
            The number of individual IVPs to integrate
        options: :class:`pyccelerInt.PySolverOptions`
            The integration options to use
        """

        # create state vectors
        P0 = 101325  # Pa
        T0 = 1000  # K
        V0 = 1  # m^3
        self.phi = np.zeros((num, self.neq), dtype=np.float64)
        self.phi[:, 0] = T0
        self.phi[:, 1] = V0

        # get species names
        species = self.knl.species_names()

        H2 = species.index('H2')
        O2 = species.index('O2')

        # determine species moles

        # PV = nRT
        # -> n = PV / RT
        n = P0 * V0 / (ct.gas_constant * T0)
        # moles in the stoichiometric eqn [2 H2 + O2 + 3.76 N2]
        n_stoich = 2 + 1 + 3.76

        # convert stoichometric eqn to system moles
        self.phi[:, H2 + 2] = (2 / n_stoich) * n
        self.phi[:, O2 + 2] = (1 / n_stoich) * n

        # set pressure
        self.pressure = np.full(num, 101325, dtype=np.float64)
        self.init = True

    def get_ivp(self):
        """
        Return the IVP for the VDP problem
        """

        if self.platform == 'opencl':
            # create ivp
            # Note: we need to pass the full paths to the PyIVP such that accelerInt
            # can find our kernel files
            files = [os.path.join(self.src_path, 'species_rates.ocl'),
                     os.path.join(self.src_path, 'chem_utils.ocl'),
                     os.path.join(self.src_path, 'dydt.cl'),
                     os.path.join(self.src_path, 'jac.cl'),
                     os.path.join(self.src_path, 'jacobian.ocl')]
            return self.get_wrapper().PyIVP(files, self.rwk_size,
                                            include_paths=[self.src_path])
        elif self.platform == 'c':
            return self.get_wrapper().PyIVP(self.rwk_size, True)

    def get_initial_conditions(self):
        """
        Returns
        -------
        phi: :class:`np.ndarray`
            A copy of this problem's initial state-vector
        user_data: :class:`np.ndarray`
            A copy of this problem's user data
        """
        return self.phi.copy(), self.pressure.copy()

    @property
    def num_equations(self):
        return self.neq

    def plot(self, ivp_index, times, solution):
        """
        Plot the solution of this problem for the specified IVP index

        Parameters
        ----------
        ivp_index: int
            The index in the state-vector to plot the IVP solution of
        times: :class:`numpy.ndarray`
            An array of times corresponding to the solution array
        solution: :class:`numpy.ndarray`
            An array shaped (num, neq, times) that contains the integrated solution,
            e.g., via :func:`integrator.state()`
        """

        plt = get_plotter()
        plt.plot(times, solution[ivp_index, 0, :], label='pyJac')
        plt.ylim(np.min(solution[ivp_index, 0, :]) * 0.95,
                 np.max(solution[ivp_index, 0, :]) * 1.05)
        plt.title('H2 Ignition')

        # plot same problem in CT for comparison
        gas = ct.Solution(os.path.join(self.dir, 'h2.cti'))
        gas.TPX = 1000, 101325, 'H2:2, O2:1, N2:3.76'
        reac = ct.IdealGasConstPressureReactor(gas)
        net = ct.ReactorNet([reac])
        t = [net.time]
        T = [reac.thermo.T]
        while net.time <= times[-1]:
            net.advance(net.time + self.get_default_stepsize())
            t.append(net.time)
            T.append(reac.thermo.T)

        plt.plot(t, T, '--', label='Cantera')

        plt.legend(loc=0)
        plt.show()

    def get_default_stepsize(self):
        """
        Return the default time-step size for this Problem
        """
        return 1e-6

    def get_default_endtime(self):
        """
        Return the default end-time for this Problem
        """
        return 1e-3
