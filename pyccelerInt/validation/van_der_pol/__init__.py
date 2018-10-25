import numpy as np

from pyccelerInt import get_plotter
from pyccelerInt.examples.van_der_pol import VDP
from pyccelerInt.validation import ValidationProblem


def vdp_init(num):
    """
    A simple initializer for the van der Pol problem

    Returns
    -------

    """

    params = np.arange(1, num + 1, dtype=np.float64)
    phi = np.zeros((num, 2), dtype=np.float64)
    phi[:, 0] = np.arange(1, num + 1, dtype=np.float64)

    return phi, params


class VDP_valid(VDP, ValidationProblem):
    def __init__(self, lang, options, reuse=False):
        """
        Initialize the problem.

        Parameters
        ----------
        lang: ['opencl', 'c']
            The runtime language to use for the problem
        options: :class:`pyccelerint/PySolverOptions`
            The solver options to use
        reuse: bool [False]
            If true, reuse any previously generated code / modules
        """

        VDP.__init__(self, lang, options, reuse=reuse)

    def setup(self, num, options, initializer=vdp_init):
        """
        Do any setup work required for this problem, initialize input arrays,
        generate code, etc.

        Parameters
        ----------
        num: int
            The number of individual IVPs to integrate
        options: :class:`pyccelerInt.PySolverOptions`
            The integration options to use
        initializer: :class:`six.Callable`
            A function that takes as an argument the :param:`num` of IVPs to generate
            initial conditions, and returns a tuple of (phi, params) -- the initial
            conditions
        """

        self.phi, self.params = initializer(num)
        self.init = True

    @property
    def step_start(self):
        return 1

    @property
    def step_end(self):
        return 1e-6

    def plot(self, step_sizes, errors, end_time, label=''):
        """
        Plot the validation curve for this problem

        Parameters
        ----------
        step_sizes: :class:`numpy.ndarray`
            The array of step sizes
        errors: :class:`numpy.ndarray`
            The array of normalized errors to plot
        """

        plt = get_plotter()
        # convert stepsizes to steps taken
        st = end_time / step_sizes
        plt.loglog(st, errors, label=label, linestyle='', marker='o')
        plt.ylim(np.min(errors) * 0.8,
                 np.max(errors) * 1.2)
        plt.legend(loc=0)
        plt.xlabel('Steps taken')
        plt.ylabel('|E|')
        plt.title('van der Pol equation')
        plt.show()
