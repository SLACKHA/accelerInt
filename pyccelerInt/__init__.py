"""pyccelerInt - python bindings for accelerInt

Defines common problem interfaces for examples, validation and performance studies
"""

from shutil import which
import subprocess
import multiprocessing
import os
import logging
import sys
import argparse
scons = which('scons').strip()
# add runtime path to find the pyccelerint module
sys.path.insert(0, os.getcwd())

lang_map = {'opencl': 'opencl',
            'c': 'cpu'}


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


def get_plotter():
    """
    Returns the matplotlib plotting module, if available
    """

    try:
        from matplotlib import pyplot as plt
        return plt
    except ImportError:
        raise Exception('Plotting not available!')


def have_plotter():
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

    available_languages = ['c', 'opencl']

    @classmethod
    def path(cls):
        """
        Returns the path
        """
        raise NotImplementedError

    @classmethod
    def build(cls, lang):
        """
        Compile / construct the problem files for this problem
        """

        try:
            path = cls.path()
            subprocess.check_output([scons,
                                     lang_map[lang] + '-wrapper',
                                     'mechanism_dir={}'.format(path),
                                     '-j', str(multiprocessing.cpu_count())])
        except subprocess.CalledProcessError as e:
            logging.getLogger(__name__).error(e.output.decode())
            raise Exception('Error building {}-wrapper for problem in '
                            'directory {}'.format(lang, path))

    @classmethod
    def generate(cls, reuse=False, **kwargs):
        """
        Generate any code that must be run _before_ building
        """
        raise NotImplementedError

    def __init__(self, lang, options):
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

        if lang not in Problem.available_languages:
            raise Exception('Unknown platform: {}!'.format(lang))
        self.lang = lang
        self.dir = os.path.abspath(self.path())
        self.options = options

        # mark not initialized
        self.init = False

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
        reuse: bool [False]
            If true, reuse any previously generated code / modules
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
        return import_wrapper(self.lang)

    def num_equations(self):
        """
        Return the number of equations to solve for this problem
        """
        return NotImplementedError

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
            self.setup(num, self.options)

        phi_i, ud_i = self.get_initial_conditions()
        # flatten
        phi = phi_i.flatten(integrator.order())
        ud = ud_i.flatten(integrator.order())

        ret = integrator.integrate(num, t_start, t_end, phi, ud, step=t_step)

        if return_state:
            # reshape
            phi = phi.reshape(phi_i.shape, order=integrator.order())
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

############################
# Input validation section #
############################


def check_vector_width(v):
    def __raise():
        raise Exception('Specified vector-width: {} is invalid'.format(v))
    try:
        v = int(v)
        if v not in [1, 2, 3, 4, 8, 16]:
            __raise()
        return v
    except ValueError:
        __raise()


def check_block_size(b):
    def __raise():
        raise Exception('Specified block size: {} is invalid'.format(b))
    try:
        b = int(b)
        # ensure power of 2
        if not (b & (b - 1)) == 0:
            __raise()
        return b
    except ValueError:
        __raise()


def enum_str(enum_val):
    s = str(enum_val)
    return s[s.index('.') + 1:]


def check_enum(choice, enum, help_str):
    selected = next((x for x in enum if choice == enum_str(x)), None)
    if selected is None:
        err_str = ('{help}: {choice} is invalid, possible choices:'
                   ' {{{choices}}}'.format(
                                        help=help_str,
                                        choice=choice,
                                        choices=', '.join([
                                            enum_str(x) for x in enum])))
        raise Exception(err_str)
    return selected


def check_integrator_type(pycel, choice):
    return check_enum(choice, pycel.IntegratorType, 'Integrator type')


def check_device_type(pycel, choice):
    return check_enum(choice, pycel.DeviceType, 'Device type')


###############################
# Build Options / Integrators #
###############################


def get_solver_options(lang, integrator_type,
                       atol=1e-10, rtol=1e-6, logging=False,
                       maximum_steps=int(1e6), vector_size=None,
                       block_size=None, use_queue=True, order='C',
                       platform='', device_type=None,
                       num_ra=10, max_krylov=-1,
                       constant_timestep=None):
    """
    Return the constructed solver options for the given pyccelerInt runtime
    """

    pycel = import_wrapper(lang)

    # language specific options
    kwargs = {}
    if constant_timestep:
        if lang == 'c':
            raise NotImplementedError
        elif lang != 'c':
            kwargs['stepper_type'] = pycel.StepperType.CONSTANT
            kwargs['h_const'] = constant_timestep

    # create the options
    if lang == 'c':
        return pycel.PySolverOptions(integrator_type,
                                     atol=atol, rtol=rtol, logging=logging,
                                     max_iters=maximum_steps,
                                     num_rational_approximants=num_ra,
                                     max_krylov_subspace_dimension=max_krylov,
                                     **kwargs)

    elif lang == 'opencl':
        return pycel.PySolverOptions(integrator_type,
                                     vectorSize=vector_size,
                                     blockSize=block_size,
                                     use_queue=use_queue,
                                     order=order,
                                     platform=platform,
                                     deviceType=device_type,
                                     atol=atol, rtol=rtol, logging=logging,
                                     maxIters=maximum_steps,
                                     **kwargs)


def build_problem(problem_type, lang, integrator_type,
                  reuse=True, vector_size=0, block_size=0, platform='',
                  order='C', logging=True, use_queue=True,
                  device_type=None, rtol=1e-06, atol=1e-10,
                  constant_timestep=None,
                  maximum_steps=int(1e6)):
    """
    Build and return the :class:`Problem` and :class:`SolverOptions`

    Parameters
    ----------
    problem_type: :class:`Problem`
        The type (note: literal python type) of problem to build.
    lang: ['c', 'opencl']
        The language to run the problem in
    integrator_type: :class:`IntegratorType`
        The integrator type to use for this problem
    reuse: bool [False]
        If true, re-use a previously generated problem
    vector_size: int [0]
        The vector size to use, if zero do not use explicit-SIMD vectorization
        [OpenCL only, exclusive with :param:`block_size`]
    block_size: int [0]
        The block size to use, if zero do not use implicit-SIMD vectorization
        [OpenCL only, exclusive with :param:`vector_size`]
    platform: str ['']
        The OpenCL platform to use
    order: ['C', 'F']
        The data-ordering pattern to use
    logging: bool [True]
        If true, enable solution logging
    use_queue: bool [True]
        If true, use the queue-scheduler based solvers [OpenCL only]
    device_type: :class:`DeviceType` [None]
        The OpenCL device type to use.
    rtol: float [1e-06]
        The relative integration tolerance
    atol: float [1e-10]
        The absolute integration tolerance
    constant_timestep: float [None]
        If specified, use this as a constant integration time-step
    maximum_steps: int [1e6]
        The maximum number of integration steps allowed per-IVP (per-global
        time-step)


    Returns
    -------
    problem: :class:`Problem`
        The constructed problem
    solver_options: :class:`SolverOptions`
        The constructed solver options
    """

    # check vector / block sizes
    if vector_size:
        check_vector_width(vector_size)
    if block_size:
        check_block_size(block_size)

    # setup / build wrapper
    problem_type.generate(lang=lang,
                          reuse=reuse,
                          vector_size=vector_size,
                          block_size=block_size,
                          platform=platform,
                          order=order)
    problem_type.build(lang)

    # get the wrapper to check the integrator / device type
    wrapper = import_wrapper(lang)
    integrator_type = check_integrator_type(wrapper, integrator_type)
    if lang == 'opencl':
        device_type = check_device_type(wrapper, device_type)
    if lang == 'c' and order == 'C':
        raise Exception('Currently only F-ordering is implemented '
                        'for the cpu-solvers.')

    options = get_solver_options(lang, integrator_type,
                                 logging=logging,
                                 vector_size=vector_size,
                                 block_size=block_size,
                                 use_queue=use_queue,
                                 order=order,
                                 platform=platform,
                                 device_type=device_type,
                                 rtol=rtol,
                                 atol=atol,
                                 maximum_steps=maximum_steps,
                                 constant_timestep=constant_timestep)

    # create problem
    problem = problem_type(lang, options)

    return problem, options


def create_integrator(problem, integrator_type, options, num_threads):
    """
    Returns
    -------
    ivp: :class:`IVP`
        The initialized IVP
    integrator: :class:`PyIntegrator`
        The initialized integrator class
    """
    pycel = import_wrapper(problem.lang)
    integrator_type = check_integrator_type(pycel, integrator_type)
    # get ivp
    ivp = problem.get_ivp()

    return ivp, pycel.PyIntegrator(integrator_type, problem.num_equations,
                                   num_threads, ivp, options)


################
# Build Parser #
################

def build_parser(helptext='Run the pyccelerInt examples.', get_parser=False):
    parser = argparse.ArgumentParser(helptext, conflict_handler='resolve')
    parser.add_argument('-c', '--case',
                        choices=['vdp', 'pyjac'],
                        required=True,
                        help='The example to run, currently only the van der Pol '
                             'problem and pyJac are implemented.')
    parser.add_argument('-l', '--language',
                        choices=['c', 'opencl'],
                        required=True,
                        help='The pyccelerInt platform to use')
    parser.add_argument('-ni', '--num_ivp',
                        type=int,
                        default=100,
                        help='The number of IVPs to solve [default: 100].')

    parser.add_argument('-nt', '--num_threads',
                        type=int,
                        default=multiprocessing.cpu_count(),
                        help='The number of threads to use during solution '
                             '[default: # of logical cores].')

    parser.add_argument('-it', '--int_type',
                        type=str,
                        required=True,
                        help='The integrator type to use.')

    parser.add_argument('-v', '--vector_size',
                        type=check_vector_width,
                        default=0,
                        help='The SIMD vector-width to use [CPU]. '
                             'Exclusive with --blockSize. Only used for OpenCL.')

    parser.add_argument('-b', '--block_size',
                        type=check_block_size,
                        default=0,
                        help='The implicit-SIMD work-group size to use [GPU].  '
                             'Exclusive with --vectorSize. Only used for OpenCL.')

    parser.add_argument('-qu', '--use_queue',
                        default=True,
                        action='store_true',
                        dest='use_queue',
                        help='Use the queue-based integration drivers. '
                             'Only used for OpenCL.')

    parser.add_argument('-st', '--use_static',
                        default=True,
                        action='store_false',
                        dest='use_queue',
                        help='Use the static scheduling based integration drivers. '
                             'Only used for OpenCL.')

    parser.add_argument('-o', '--order',
                        choices=['C', 'F'],
                        default='F',
                        help='The data-ordering, row-major ("C") or column-major '
                             '("F").')

    parser.add_argument('-p', '--platform',
                        type=str,
                        help='The OpenCL platform to use, (e.g., Intel, NVIDIA, '
                             'etc.)')

    parser.add_argument('-d', '--device_type',
                        type=str,
                        default='DEFAULT',
                        help='The device type to use (e.g., CPU, GPU, ACCELERATOR). '
                             'Only used for OpenCL.')

    parser.add_argument('-tf', '--end_time',
                        type=float,
                        default=None,  # 1ms
                        help='The simulation end-time.')

    parser.add_argument('-ru', '--reuse',
                        action='store_true',
                        default=False,
                        help='Reuse the previously generated pyJac code / library.')

    parser.add_argument('-rtol', '--relative_tolerance',
                        type=float,
                        default=1e-06,
                        help='The relative tolerance for the solvers.')

    parser.add_argument('-atol', '--absolute_tolerance',
                        type=float,
                        default=1e-10,
                        help='The absolute tolerance for the solvers.')

    parser.add_argument('-m', '--max_steps',
                        type=float,
                        default=1e6,
                        help='The maximum number of steps allowed per global '
                             'integration time-step.')

    if get_parser:
        return parser

    return parser.parse_args()
