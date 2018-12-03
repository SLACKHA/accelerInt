import os
import numpy as np

from pyccelerInt.examples.pyJac import Ignition
from pyccelerInt.validation import ValidationProblem

from pyjac.tests.test_utils.data_bin_writer import load as dload


def pyjac_valid(num, conp=True):
    """
    Loads partially stirred reactor states for this model
    """

    # load data
    num_conditions, data = dload([], os.path.dirname(__file__))

    # data is in the form, T, P, concentrations
    T = data[:num, 0].flatten()
    P = data[:num, 1].flatten()
    # set V -> 1, such that concentrations = moles
    V = np.ones_like(P)
    # set moles
    moles = data[:num, 2:].copy()

    if conp:
        param = P
        extra = V
    else:
        param = V
        extra = P

    phi = np.concatenate((np.reshape(T, (-1, 1)),
                          np.reshape(extra, (-1, 1)),
                          moles[:, :-1]), axis=1)

    return phi, param


class Ignition_valid(ValidationProblem, Ignition):
    @classmethod
    def data_path(cls):
        return ''

    @classmethod
    def model(cls):
        return 'gri30.cti'

    def __init__(self, lang, options, conp=True):
        """
        Initialize the problem.

        Parameters
        ----------
        lang: ['opencl', 'c']
            The runtime language to use for the problem
        options: :class:`pyccelerint/PySolverOptions`
            The solver options to use
        conp: bool [False]
            If true, constant pressure ignition problem, else constant volume
        """

        Ignition.__init__(self, lang, options, conp=conp)
        self.regions = {}

    def setup(self, num, options, initializer=pyjac_valid):
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

        if not initializer:
            self.phi, self.pressure = super(Ignition_valid, self).setup(num, options)
        else:
            self.phi, self.pressure = initializer(num, self.conp)
        self.init = True

    def get_default_endtime(self):
        """
        Return the default end-time for this Problem
        """
        return 1e-6

    @property
    def plot_name(self):
        return "{} Ignition".format('CONP' if self.conp else 'CONV')

    def plot_specialize(self, solver, runtimes, errors, tols):
        switch_1 = np.where(tols == 1e-07)[0][0]
        switch_2 = np.where(tols == 1e-11)[0][0]
        if solver not in self.regions:
            self.regions[solver] = {}
        self.regions[solver][1e-07] = np.mean(runtimes[switch_1])
        self.regions[solver][1e-11] = np.mean(runtimes[switch_2])

    def finalize_specializaton(self, plt):
        # draw regions
        region_1 = np.min([self.regions[sol][1e-07] for sol in self.regions])
        region_2 = np.max([self.regions[sol][1e-11] for sol in self.regions])
        plt.axvspan(region_1, region_2, facecolor='lightgrey', zorder=0)
