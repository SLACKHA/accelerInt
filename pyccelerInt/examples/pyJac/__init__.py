##
# \file
# \brief Runs the integrators for van der Pol problem

import os

import cantera as ct
import numpy as np
from pyjac import __version_info__, create_jacobian
from pyjac.core.enum_types import KernelType, JacobianFormat
from pyjac.pywrap import pywrap

from pyccelerInt import Problem, get_plotter, lang_map

assert(__version_info__[0] >= 2), 'Only pyJac-V2+ supports OpenCL execution.'


def import_pyjac(lang):
    try:
        if lang == 'opencl':
            import pyjac_ocl as pj  # noqa
            return pj
        elif lang == 'c':
            import pyjac_c as pj  # noqa
            return pj
        else:
            raise Exception('Language {} not recognized!'.format(lang))
    except ImportError:
        raise Exception('pyccelerInt wrapper for language: {} could not be '
                        'imported (using path {})'.format(
                            lang, os.getcwd()))


class Ignition(Problem):
    """
    An implementation of a constant pressure / volume homogenous ignition problem
    for pyccelerInt, utilizing source rates and Jacobian's from pyJac.
    """

    @classmethod
    def path(cls):
        """
        Returns the path
        """
        return cls._src_path()

    @classmethod
    def data_path(cls):
        return os.path.abspath(os.path.dirname(__file__))

    @classmethod
    def generate(cls, lang, vector_size=None, block_size=None, platform='',
                 order='C', reuse=False):
        """
        Generate any code that must be run _before_ building
        """

        # build pyjac module
        # set OpenCL specific options
        platform = None
        width = None
        explicit_simd = False
        if lang == 'opencl':
            width = vector_size if vector_size else block_size
            if width <= 1:
                width = None
            platform = platform
            explicit_simd = not block_size and width

        out_path = cls._src_path(lang)

        # create source-rate evaluation
        obj_path = os.path.join(out_path, 'obj')
        ktype = KernelType.jacobian
        if not reuse:
            create_jacobian(lang,
                            mech_name=os.path.join(cls.data_path(), 'h2.cti'),
                            width=width, build_path=out_path, last_spec=None,
                            kernel_type=ktype,
                            platform=platform,
                            data_order=order,
                            jac_format=JacobianFormat.full,
                            explicit_simd=explicit_simd,
                            unique_pointers=lang == 'c')

            # and compile to get the kernel object
            pywrap(lang, out_path, obj_path, os.getcwd(), obj_path,
                   platform, ktype=ktype)

            # create a simple wrapper to join
            if lang == 'opencl':
                with open(os.path.join(out_path, 'dydt.cl'), 'w') as file:
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
                with open(os.path.join(out_path, 'jac.cl'), 'w') as file:
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
            elif lang == 'c':
                with open(os.path.join(out_path, 'dydt.cpp'), 'w') as file:
                    file.write(
                        """
                        #include "species_rates.hpp"
                        extern "C" {
                            void dydt (const double t,
                                       const double param,
                                       const double * __restrict__ y,
                                       double * __restrict__ dy,
                                       double* __restrict__ rwk)
                           {
                              species_rates(0, &param, y, dy, rwk);
                           }
                       }
                       """)
                with open(os.path.join(out_path, 'jac.cpp'), 'w') as file:
                    file.write(
                        """
                        #include "jacobian.hpp"
                        extern "C" {
                            void eval_jacob (const double t,
                                             const double param,
                                             const double * __restrict__ y,
                                             double * __restrict__ jac,
                                             double * __restrict__ rwk)
                           {
                              jacobian(0, &param, y, jac, rwk);
                           }
                        }
                       """)

    def __init__(self, platform, options, conp=True):
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
        super(Ignition, self).__init__(platform, options)

        # create initial state
        pj = import_pyjac(self.lang)
        # create a kernel -- note that we supply 1 IVP / 1 thread here such that we
        # can properly scale the required memory
        self.knl = pj.PyJacobianKernel(1, 1)

        # equations are T, V, N_{0}... N_{nsp-1}
        self.neq = self.knl.num_species() + 1

        # get the working memory
        self.rwk_size = self.knl.required_working_memory()
        if self.lang == 'opencl' and (options.vector_size()
                                      or options.block_size()):
            # pyJac returns the complete (vectorized) memory req's, but we need
            # to pass in the unvectorized size
            vw = options.vector_size() if options.vector_size() else \
                options.block_size()
            self.rwk_size /= vw

    @classmethod
    def _src_path(cls, lang=''):
        path = os.path.join(os.getcwd(), 'examples_working')
        if lang:
            path = os.path.join(path, lang_map[lang])
        return path

    @property
    def src_path(self):
        return self._src_path(self.lang)

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

        # store tolerances
        self.rtol = options.rtol()
        self.atol = options.atol()

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

        if self.lang == 'opencl':
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
        elif self.lang == 'c':
            return self.get_wrapper().PyIVP(self.rwk_size)

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
        gas = ct.Solution(os.path.join(self.data_path(), 'h2.cti'))
        gas.TPX = 1000, 101325, 'H2:2, O2:1, N2:3.76'
        reac = ct.IdealGasConstPressureReactor(gas)
        net = ct.ReactorNet([reac])
        net.rtol, net.atol = self.rtol, self.atol
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
