##
# \file
# \brief Runs the integrators for van der Pol problem

import numpy as np
import os

from pyccelerInt import Problem, get_plotter


class VDP(Problem):
    """
    An implementation of the van der Pol problem for pyccelerInt
    """

    @classmethod
    def path(cls):
        """
        Returns the path
        """
        return os.path.abspath(os.path.dirname(__file__))

    @classmethod
    def generate(cls, reuse=False, **kwargs):
        """
        Generate any code that must be run _before_ building
        """
        pass

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

        # number of equations
        neq = self.num_equations

        # create state vectors
        self.phi = 2 * np.zeros((num, neq), dtype=np.float64)
        self.phi[:, 0] = 2
        self.phi[:, 1] = 0

        # set parameters
        self.params = np.zeros(num, dtype=np.float64)
        self.params[:] = 1000
        self.init = True

    def get_ivp(self):
        """
        Return the IVP for the VDP problem
        """
        if self.lang == 'opencl':
            # create ivp
            # Note: we need to pass the full paths to the PyIVP such that accelerInt
            # can find our kernel files
            return self.get_wrapper().PyIVP([
                os.path.join(self.path(), 'opencl', 'dydt.cl'),
                os.path.join(self.path(), 'opencl', 'jacob.cl')], 0)
        elif self.lang == 'c':
            return self.get_wrapper().PyIVP(0)

    def get_initial_conditions(self):
        """
        Returns
        -------
        phi: :class:`np.ndarray`
            A copy of this problem's initial state-vector
        user_data: :class:`np.ndarray`
            A copy of this problem's user data
        """
        return self.phi.copy(), self.params.copy()

    @property
    def num_equations(self):
        return 2

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
        plt.plot(times, solution[ivp_index, 0, :], label='y1')
        plt.plot(times, solution[ivp_index, 1, :], label='y2')
        plt.ylim(np.min(solution[ivp_index, 0, :]) * 1.05,
                 np.max(solution[ivp_index, 0, :]) * 1.05)
        plt.legend(loc=0)
        plt.title('van der Pol equation')
        plt.show()

    def get_default_stepsize(self):
        """
        Return the default time-step size for this Problem
        """
        return 1.

    def get_default_endtime(self):
        """
        Return the default end-time for this Problem
        """
        return 2000.
