"""pyccelerInt - python bindings for accelerInt

Defines common problem interfaces for examples, validation and performance studies
"""

from shutil import which
import subprocess
import multiprocessing
import os
import logging
import sys
scons = which('scons').strip()
# add runtime path to find the pyccelerint module
sys.path.insert(0, os.getcwd())


def import_wrapper(platform):
    try:
        if platform == 'opencl':
            import pyccelerInt_ocl as pycel  # noqa
            return pycel
        elif platform == 'c':
            import pyccelerInt_cpu as pycel  # noqa
            return pycel
        else:
            raise Exception('Language {} not recognized!'.format(platform))
    except ImportError:
        raise Exception('pyccelerInt wrapper for platform: {} could not be '
                        'imported (using path {})'.format(
                            platform, os.getcwd()))


def get_plotter(self):
    """
    Returns the matplotlib plotting module, if available
    """

    try:
        from matplotlib.pyplot import plt
        return plt
    except ImportError:
        raise Exception('Plotting not available!')


def have_plotter(self):
    """
    Return True if we have matplotlib available for plotting
    """

    try:
        return get_plotter()
    except Exception:
        return False


class Problem(object):
    """
    An abstract base class to define problems for pyccelerInt
    """

    available_platforms = ['c', 'opencl']

    def __init__(self, platform, code_directory):
        """
        Initialize the problem.

        Parameters
        ----------
        platform: ['opencl', 'c']
            The runtime platform to use for the problem
        code_directory: str
            The path to the user implementation of the source term
            and jacobian files
        """

        if platform not in Problem.available_platforms:
            raise Exception('Unknown platform: {}!'.format(platform))
        self.platform = platform
        self.dir = os.path.abspath(code_directory)
        self.init = False
        self.built = False

    def build(self):
        """
        Compile / construct the problem files
        """

        try:
            out = subprocess.check_call([scons, self.platform + '-wrapper',
                                         'mechanism_dir={}'.format(self.dir),
                                         '-j', multiprocessing.cpu_count()])
        except subprocess.ProcessCalledError as e:
            logging.getLogger(__name__).error(out)
            raise Exception('Error building {}-wrapper for problem in '
                            'directory {}'.format(self.platform, self.dir))

    def setup(self, num):
        """
        Do any setup work required for this problem, initialize input arrays,
        generate code, etc.

        Parameters
        ----------
        num: int
            The number of individual IVPs to integrate
        """
        raise NotImplementedError

    def get_initial_conditions(self):
        """
        Returns
        -------
        phi: :class:`np.ndarray`
            A copy of this problem's initial state-vector
        user_data: :class:`numpy.ndarray`
            A copy of this problem's user data
        """
        raise NotImplementedError

    def get_ivp(self):
        """
        Returns
        -------
        pyivp: :class:`PyIVP`
            A python wrapped IVP class for integrator initialization
        """
        raise NotImplementedError

    def get_default_stepsize(self):
        """
        Return the default time-step size for this Problem
        """
        raise NotImplementedError

    def get_default_endtime(self):
        """
        Return the default end-time for this Problem
        """
        raise NotImplementedError

    def get_wrapper(self):
        """
        Returns
        -------
        wrapper: module
            The imported pyccelerInt wrapper for this :attr:`platform`, used for
            creation of the integrator / IVP / solver options / etc.
        """
        return import_wrapper(self.platform)

    @property
    def num_equations(self):
        """
        Return the number of equations to solve for this problem
        """
        raise NotImplementedError

    def run(self, integrator, num, t_end, t_start=0., t_step=-1.,
            return_state=False):
        """
        Integrate the IVPs for this problem definition

        Parameters
        ----------
        integrator: :class:`pyccelerint.PyIntegrator`
            The integrator object to use
        num: int
            The number of individual IVPs to integrate
        t_end: float
            The simulation end time
        t_start: float [0]
            Optional, the simulation start time.  If not specified, defaults to zero.
        t_step: float [-1]
            Optional, if specified the global integration time-step.
            Useful for logging, as this controls the times at which the state is
            output.
        return_state: bool [False]
            Optional, if True return the final integration state in addition to
            the wall-clock duration

        Returns
        -------
        time: float
            The wall-clock duration of IVP integration measured in milliseconds
        state: :class:`numpy.ndarray`
            If :param:`return_state` is True, the final state vector is returned
        """

        if not self.init:
            self.setup(num)
            self.init = True
        if not self.built:
            self.build()
            self.built = True

        phi_i, ud_i = self.get_initial_conditions()
        # flatten
        phi = phi_i.flatten(integrator.order())
        ud = ud_i.flatten(integrator.order())

        ret = integrator.integrate(num, t_start, t_end, phi, ud, step=t_step)

        if return_state:
            # reshape
            phi = phi.reshape(phi_i.shape, integrator.order())
            ret = (ret, phi)

        return ret

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
        raise NotImplementedError


def get_solver_options(lang, integrator_type,
                       atol=1e-10, rtol=1e-6, logging=False,
                       maximum_steps=1e6, h_init=1e-6,
                       vector_size=None, block_size=None,
                       use_queue=True, order='C',
                       platform='', device_type=None,
                       num_ra=10, max_krylov=-1):
    """
    Return the constructed solver options for the given pyccelerInt runtime
    """

    pycel = import_wrapper(lang)
    # create the options
    if lang == 'c':
        return pycel.PySolverOptions(integrator_type,
                                     atol=atol, rtol=rtol, logging=logging,
                                     h_init=h_init,
                                     num_rational_approximants=num_ra,
                                     max_krylov_subspace_dimension=max_krylov)

    elif lang == 'opencl':
        return pycel.PySolverOptions(integrator_type,
                                     vectorSize=vector_size,
                                     blockSize=block_size,
                                     use_queue=use_queue,
                                     order=order,
                                     platform=platform,
                                     deviceType=device_type,
                                     atol=atol, rtol=rtol, logging=logging,
                                     maxIters=maximum_steps)


def create_integrator(problem, integrator_type, options, num_threads):
    """
    Returns
    -------
    ivp: :class:`IVP`
        The initialized IVP
    integrator: :class:`PyIntegrator`
        The initialized integrator class
    """
    pycel = import_wrapper(problem.platform)
    # get ivp
    ivp = problem.get_ivp()

    return pycel.PyIntegrator(integrator_type, int(problem.num_equations),
                              int(num_threads), ivp, options)
