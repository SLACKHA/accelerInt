import logging
import argparse

import numpy as np
from pyccelerInt import create_integrator, build_problem, have_plotter, get_plotter


class ValidationProblem(object):
    """
    Abstract base class for validation problems
    """

    def __init__(self):
        pass

    @property
    def step_start(self):
        """
        Return the largest validation constant step-size
        """
        raise NotImplementedError

    @property
    def step_end(self):
        """
        Return the smalles validation constant step-size
        """
        raise NotImplementedError

    @property
    def plot_name(self):
        raise NotImplementedError

    def plot(self, step_sizes, errors, end_time, label='', order=None,
             plot_filename=''):
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

        if order is not None:
            # plot expected order
            expected = np.zeros_like(errors)
            expected[0] = errors[0]
            for i in range(1, expected.size):
                expected[i] = expected[i - 1] * np.power(
                    step_sizes[i] / step_sizes[i - 1], order)

            plt.loglog(st, expected, label='order ({})'.format(order))

        plt.ylim(np.min(errors) * 0.8,
                 np.max(errors) * 1.2)
        plt.legend(loc=0)
        plt.xlabel('Steps taken')
        plt.ylabel('|E|')
        plt.title(self.plot_name)
        if not plot_filename:
            plt.show()
        else:
            plt.savefig(plot_filename)


def build_case(problem, lang, is_reference=False,
               integrator_type='', reuse=False,
               vector_size=0, block_size=0,
               platform='', order='F', use_queue=True,
               device_type='DEFAULT', max_steps=int(1e6),
               reference_rtol=1e-15, reference_atol=1e-20,
               num_threads=1, constant_timestep=None):
    """
    Run validation for the given reference and test problems

    Parameters
    ----------
    problem: :class:`pyccelerInt.Problem`
        The problem to build
    lang: ['c', 'opencl']
        The language to run the problem in
    is_reference: bool [False]
        If true, this integrator will be used as the reference answer
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
    use_queue: bool [True]
        If true, use the queue-scheduler based solvers [OpenCL only]
    device_type: :class:`DeviceType` [None]
        The OpenCL device type to use.
    rtol: float [1e-06]
        The relative integration tolerance
    atol: float [1e-10]
        The absolute integration tolerance
    maximum_steps: int [1e6]
        The maximum number of integration steps allowed per-IVP (per-global
        time-step)
    num_threads: int [1]
        The number of threads to use for integration


    Returns
    -------
    ivp: :class:`PyIVP`
        The constructed IVP
    problem: :class:`Problem`
        The constructed problem
    solver_options: :class:`SolverOptions`
        The constructed solver options
    """

    kwargs = {}
    if is_reference:
        kwargs['rtol'] = reference_rtol
        kwargs['atol'] = reference_atol
    else:
        assert constant_timestep
        kwargs['constant_timestep'] = constant_timestep

    problem, options = build_problem(problem, lang, integrator_type,
                                     reuse=reuse,
                                     vector_size=vector_size,
                                     block_size=block_size,
                                     platform=platform,
                                     order=order,
                                     use_queue=use_queue,
                                     device_type=device_type,
                                     maximum_steps=max_steps,
                                     logging=False,
                                     **kwargs)

    # get IVP and integrator
    ivp, integrator = create_integrator(problem, integrator_type, options,
                                        num_threads)

    return ivp, problem, integrator


class NanException(Exception):
    pass


def run_case(num, phir, test, test_problem,
             t_end, norm_rtol=1e-6, norm_atol=1e-10,
             condition_norm=2,
             ivp_norm=np.inf, results={}):
    """
    Run the :param:`test` integrator over :param:`num`
    problems, to end time :param:`t_end`, using constant integration timesteps

    Parameters
    ----------
    num: int
        The number of individual IVPs to execute
    test: :class:`PyIntegrator`
        The integrator to validate
    test_problem: :class:`Problem`
        The constructed problem for the test integrator
    t_end: float
        The integration end time
    norm_atol: float [1e-10]
        The absolute tolerance used in error normalization
    norm_rtol: float [1e-06]
        The absolute tolerance used in error normalization
    condition_norm: str or int
        The linear algebra norm used for determination of error in each IVP,
        see CN in the equation below
    ivp_norm: str or int
        The linear algebra norm used over the :param:`condition_norm` of all IVPs,
        see IN in the equation below
    results: dict
        If 'true', store the test results here

    Returns
    -------
    norm_error: :class:`np.ndarray`
        The weighted root-mean-squared error, calculate as:

        .. math::

            E_i\left(t\right) &= \left\lVert \frac{y_i\left(t\right) -
                \hat{y_i}\left(t\right)}{\text{atol} + \hat{y_i}\left(t\right)
                \text{rtol}}\right\rVert_{\text{CN}}

            \left\lvert E\right\rvert &= \left\lVert E_{i}\left(t\right)
                \right\rVert_{\text{IN}}
    """

    def __store_check(key, value):
        if not results:
            return
        if key not in results:
            results[key] = np.array([value])
        else:
            if not np.array_equal(results[key], np.array([value])):
                raise NanException('Mismatch in results for value {}, stored: {}, ',
                                   'current: {}'.format(value, results[key], value))

    if results:
        # store rtol / atol / etc. if not already there
        __store_check('norm_atol', norm_atol)
        __store_check('norm_rtol', norm_rtol)
        __store_check('condition_norm', condition_norm)
        __store_check('ivp_norm', ivp_norm)

    # run test
    _, phit = test_problem.run(test, num, t_end, t_start=0, return_state=True)
    if results:
        __store_check('test_{}'.format(test.constant_timestep()), phit)

    if np.any(phit != phit):
        logger = logging.getLogger(__name__)
        logger.warn('NaN / Solution detected for step-size! '
                    'Try reducing starting step-size.')
        raise NanException()

    # calculate condition norm
    phir_limited = np.take(phir, np.arange(num), axis=0)
    err = (np.abs(phit - phir_limited) / (
        norm_atol + np.abs(phir_limited) * norm_rtol)).squeeze()
    err = np.linalg.norm(err, ord=condition_norm, axis=1)

    # calculate ivp norm
    err = np.linalg.norm(err, ord=ivp_norm, axis=0)

    if results:
        __store_check('err_{}'.format(test.constant_timestep()), err)

    return err


def run_validation(num, reference, ref_problem,
                   t_end, test_builder, step_start=1e-3, step_end=1e-8,
                   norm_rtol=1e-6, norm_atol=1e-10, condition_norm=2,
                   ivp_norm=np.inf, label='', plot_filename='',
                   error_filename='', reuse=False, num_points=10):
    """
    Run the validation case for the test integrator using constant time-steps ranging
    from :param:`step_start` to `step_end`

    Parameters
    ----------
    num: int
        The number of individual IVPs to execute
    reference: :class:`PyIntegrator`
        The reference integrator
    ref_problem: :class:`Problem`
        The constructed problem for the reference integrator
    t_end: float
        The integration end time to use for all validation case
    step_start: float
        The largest constant integration step-size to use for the test integrator
        validation
    step_end: float
        the smallest constant integration step-size to use for the test integrator
        validation
    norm_atol: float [1e-10]
        The absolute tolerance used in error normalization
    norm_rtol: float [1e-06]
        The absolute tolerance used in error normalization
    condition_norm: str or int
        The linear algebra norm used for determination of error in each IVP,
        see CN in the equation below
    ivp_norm: str or int
        The linear algebra norm used over the :param:`condition_norm` of all IVPs,
        see IN in the equation below
    test_builder: :class:`Callable`
        A Callable that takes as an argument the step-size (and an optional argument
        of the iteration to avoid re-building code if possible)
        and returns the result of :func:`build_case`
    plot_filename: str
        If supplied, save plot to this file
    error_filename: str
        If supplied, save validation / test results to this .npz file for
        post-procesisng
    reuse: bool [False]
        If true, attempt to reuse validation data from :param:`error_filename`.
        Requires that :param:`error_filename` be specified.

    Returns
    -------
    step_sizes: :class:`np.ndarray`
        The constant integration step-sizes used for the test-case
    err: :class:`np.ndarray`
        The error norms calculated for the problem (:see:`run_case`) for each
        step-size
    """

    results = {}
    if reuse:
        if not error_filename:
            raise Exception('Error filename must be supplied to reuse past '
                            'validation!')

        # load from npz
        phir = np.load(error_filename)['phi_valid']
    else:
        # run reference problem once
        # run validation
        _, phir = ref_problem.run(reference, num, t_end, t_start=0,
                                  return_state=True)

    if error_filename:
        results['phi_valid'] = phir

        # save results to file as intermediate results
        np.savez(error_filename, **results)

    start = np.rint(np.log10(step_start))
    end = np.rint(np.log10(step_end))
    # determine direction of progression, and ensure that the final step-size is
    # included
    steps = np.logspace(start, end, num=num_points)
    errs = np.zeros(steps.size)
    test_order = None

    for i, size in enumerate(steps):
        testivp, test_problem, test = test_builder(steps[i], iteration=i)

        if test_order is None:
            test_order = test.solver_order()

        try:
            errs[i] = run_case(num, phir,
                               test, test_problem,
                               t_end, norm_rtol=norm_rtol, norm_atol=norm_atol,
                               condition_norm=condition_norm, ivp_norm=ivp_norm,
                               results=results if error_filename else False)
        except NanException:
            pass

        # save results to file as intermediate results
        if error_filename:
            np.savez(error_filename, **results)

        del testivp
        del test_problem
        del test

        print(steps[i], errs[i])

    if results:
        # save to file
        np.savez(error_filename, **results)

    # filter
    good = np.where(errs != 0)
    errs = errs[good]
    steps = steps[good]

    if have_plotter():
        ref_problem.plot(steps, errs, t_end, label=label,
                         order=test_order, plot_filename=plot_filename)

    return steps, errs


def filetype(value, ext):
    if not isinstance(value, str):
        raise argparse.ArgumentError('{} must be a string!'.format(value))
    if not value.endswith(ext):
        raise argparse.ArgumentError('{} must be a {} file!'.format(value, ext))
    return value


def npz_file(value):
    return filetype(value, '.npz')


def pdf_file(value):
    return filetype(value, '.pdf')


def build_parser(helptext='Run pyccelerInt validation examples.', get_parser=False):
    from pyccelerInt import build_parser
    parser = build_parser(helptext=helptext, get_parser=True)

    # add validation specific options
    parser.add_argument('-fp', '--plot_filename',
                        default=None,
                        type=pdf_file,
                        help='If specified, save the resulting plot to a file '
                             'instead of showing it to the screen')

    parser.add_argument('-fe', '--error_filename',
                        default=None,
                        type=npz_file,
                        help='If specified, save the error results to a file for '
                             'later processing.')

    parser.add_argument('-rv', '--reuse_validation',
                        default=False,
                        action='store_true',
                        help='If supplied, attempt to reuse the validation data '
                             'from the `error_filename`.')

    parser.add_argument('-nv', '--num_validation',
                        default=10,
                        type=int,
                        help='The number of timesteps to use for validation between '
                             'the start and end times.')

    parser.add_argument('-ss', '--starting_stepsize',
                        default=None,
                        type=float,
                        help='If supplied, the starting validation step-size.')

    parser.add_argument('-se', '--ending_stepsize',
                        default=None,
                        type=float,
                        help='If supplied, the final validation step-size.')

    # and change default max steps
    parser.set_defaults(max_steps=float(1e9))

    if get_parser:
        return parser

    return parser.parse_args()
