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
    def tol_start(self):
        """
        Return the largest validation tolerance
        """
        return 1e-4

    @property
    def tol_end(self):
        """
        Return the smallest validation tolerance
        """
        return 1e-15

    @property
    def plot_name(self):
        raise NotImplementedError

    def plot(self, runtimes, errors, label='', order=None,
             plot_filename='', final=False):
        """
        Plot the validation curve for this problem

        Parameters
        ----------
        runtimes: list of list of float
            The array of (arrays) of runtimes for each solver / tolerance combination
        errors: :class:`numpy.ndarray`
            The array of normalized errors to plot
        """

        plt = get_plotter(not plot_filename)
        rt_dev = np.zeros(len(runtimes))
        rt = np.zeros(len(runtimes))
        for i in range(len(runtimes)):
            rt_dev[i] = np.std(runtimes[i])
            rt[i] = np.mean(runtimes[i])
        # convert stepsizes to steps taken
        plt.errorbar(rt, errors, xerr=rt_dev, marker='.', linestyle='',
                     label=label)

        if final:
            plt.xscale('log', basex=10)
            plt.yscale('log', basey=10)
            plt.legend(loc=0)
            plt.ylabel('|E|')
            plt.xlabel('CPU Time (ms)')
            plt.title(self.plot_name)
            if not plot_filename:
                plt.show()
            else:
                plt.savefig(plot_filename)


def build_case(problem, lang, rtol, atol,
               integrator_type='', reuse=False,
               vector_size=0, block_size=0,
               platform='', order='F', use_queue=True,
               device_type='DEFAULT', max_steps=int(1e6),
               num_threads=1, constant_timestep=None):
    """
    Run validation for the given reference and test problems

    Parameters
    ----------
    problem: :class:`pyccelerInt.Problem`
        The problem to build
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
                                     rtol=rtol,
                                     atol=atol)

    # get IVP and integrator
    ivp, integrator = create_integrator(problem, integrator_type, options,
                                        num_threads)

    return ivp, problem, integrator


class NanException(Exception):
    pass


def run_case(num, phir, test, test_problem,
             t_end, name, norm_rtol=1e-6, norm_atol=1e-10,
             condition_norm=2, ivp_norm=np.inf, results={},
             num_repeats=5):
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
    num_repeats: int
        The number of times to run the case, in order to get more reliable runtime
        estimations

    Returns
    -------
    runtimes: list of float
        The runtimes for the solver in milliseconds
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
        __store_check('condition_norm', condition_norm)
        __store_check('ivp_norm', ivp_norm)

    # run test
    runtimes = []
    print('{}/{}'.format(1, num_repeats))
    rt, phit = test_problem.run(test, num, t_end, t_start=0, return_state=True)
    runtimes.append(rt)
    for i in range(1, num_repeats):
        print('{}/{}'.format(i + 1, num_repeats))
        rt, _ = test_problem.run(test, num, t_end, t_start=0, return_state=True)
        runtimes.append(rt)

    if results:
        __store_check('test_{}'.format(name), phit)

    if np.any(phit != phit):
        logger = logging.getLogger(__name__)
        logger.warn('NaN / Solution detected for step-size! '
                    'Try reducing starting step-size.')
        raise NanException()

    # calculate condition norm
    phir_limited = np.take(phir, np.arange(num), axis=0)
    err = (np.abs(phit - phir_limited) / (
        norm_atol / norm_rtol + np.abs(phir_limited))).squeeze()
    err = np.linalg.norm(err, ord=condition_norm, axis=1)

    # calculate ivp norm
    err = np.linalg.norm(err, ord=ivp_norm, axis=0)

    if results:
        __store_check('err_{}'.format(name), err)

    return runtimes, err


def run_validation(num, reference, ref_problem,
                   t_end, test_builder, solvers, tol_start=1e-4, tol_end=1e-15,
                   norm_rtol=1e-6, norm_atol=1e-10, condition_norm=2,
                   ivp_norm=np.inf, plot_filename='',
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
    tol_start: float
        The largest adaptive tolerance (used for RTOL & ATOL) to use in validation
        testing
    tol_end: float
        The smallest adaptive tolerance (used for RTOL & ATOL) to use in validation
        testing
    solvers: list of solver_type's
        If specified, run the validation case over all the solvers on the list.
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

    # determine direction of progression, and ensure that the final step-size is
    # included
    tols = np.arange(np.rint(np.log10(tol_start)), np.rint(np.log10(tol_end)) - 1,
                     step=-1)
    tols = np.power(10., tols)
    errs = np.zeros(tols.size)
    runtimes = [None for i in range(tols.size)]
    test_order = None

    for j, solver in enumerate(solvers):
        for i, tol in enumerate(tols):
            testivp, test_problem, test = test_builder(
                tol, tol, iteration=i, solver=solver)

            if test_order is None:
                test_order = test.solver_order()

            try:
                runtimes[i], errs[i] = run_case(
                    num, phir, test, test_problem, t_end,
                    '{}_{}'.format(solver, tol),
                    norm_rtol=norm_rtol, norm_atol=norm_atol,
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

            print(tols[i], errs[i])

        if results:
            # save to file
            np.savez(error_filename, **results)

        # filter
        good = np.where(errs != 0)
        errs = errs[good]
        steps = tols[good]

        if have_plotter():
            ref_problem.plot(runtimes, errs, label=solver,
                             order=test_order, plot_filename=plot_filename,
                             final=j == len(solvers) - 1)

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

    parser.add_argument('-ss', '--starting_tolerance',
                        default=None,
                        type=float,
                        help='If supplied, the starting validation tolerance.')

    parser.add_argument('-se', '--ending_tolerance',
                        default=None,
                        type=float,
                        help='If supplied, the final validation tolerance.')

    # and change default max steps
    parser.set_defaults(max_steps=float(1e9))

    # and help text on the integrator type
    parser.add_argument('-it', '--int_type',
                        type=str,
                        required=True,
                        help='A comma separated list of integrators to validate.')
    if get_parser:
        return parser

    return parser.parse_args()
