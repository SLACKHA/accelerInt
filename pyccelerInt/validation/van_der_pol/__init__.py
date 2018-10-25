import numpy as np

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
    def __init__(self, lang, options):
        """
        Initialize the problem.

        Parameters
        ----------
        lang: ['opencl', 'c']
            The runtime language to use for the problem
        options: :class:`pyccelerint/PySolverOptions`
            The solver options to use
        """

        VDP.__init__(self, lang, options)

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
    def plot_name(self):
        return "van der Pol equation"

    @property
    def step_start(self):
        return 1

    @property
    def step_end(self):
        return 1e-6
