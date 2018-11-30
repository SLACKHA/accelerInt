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

    def linestyle(self, index):
        ls = ['-.', ':', '--', '-']
        return ls[index % len(ls)]

    def markerstyle(self, index):
        ms = ['.', 'v', 's', 'p']
        return ms[index % len(ms)]

    def color(self, plt, num_solvers, index):
        ns = num_solvers + 1
        return plt.get_cmap('inferno', ns)(index)

    def use_agg(self, plot_filename):
        return not plot_filename

    def plot(self, runtimes, errors, label='', order=None,
             plot_filename='', index=None, num_solvers=None,
             use_latex=True):
        """
        Plot the validation curve for this problem

        Parameters
        ----------
        runtimes: list of list of float
            The array of (arrays) of runtimes for each solver / tolerance combination
        errors: :class:`numpy.ndarray`
            The array of normalized errors to plot
        """

        plt = get_plotter(use_agg=self.use_agg(plot_filename), use_latex=use_latex)
        rt_dev = np.zeros(len(runtimes))
        rt = np.zeros(len(runtimes))
        for i in range(len(runtimes)):
            rt_dev[i] = np.std(runtimes[i])
            rt[i] = np.mean(runtimes[i])
        # convert stepsizes to steps taken
        markersize = 10
        plt.errorbar(rt, errors, xerr=rt_dev, marker=self.markerstyle(index),
                     linestyle=self.linestyle(index), label=label,
                     color=self.color(plt, num_solvers, index),
                     markerfacecolor='none', markersize=markersize)

        if index == num_solvers - 1:
            plt.xscale('log', basex=10)
            plt.yscale('log', basey=10)
            ylabel = '|E|'
            legend_fontsize = 20
            label_fontsize = 24
            tick_size = 20
            minor_size = 16
            if use_latex:
                ylabel = r'$\left\lVert E\right\rVert$'
            plt.ylabel(ylabel, fontsize=label_fontsize)
            plt.xlabel('CPU Time (ms)', fontsize=label_fontsize)
            plt.legend(**{'loc': 0,
                          'fontsize': legend_fontsize,
                          'numpoints': 1,
                          'shadow': True,
                          'fancybox': True})
            plt.tick_params(axis='both', which='major', labelsize=tick_size)
            plt.tick_params(axis='both', which='minor', labelsize=minor_size)
            plt.tight_layout()
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


def __store_check(results, keylist, value, allow_overwrite=False):
    if not results:
        return
    if not isinstance(keylist, list):
        keylist = [keylist]

    whereat = results
    for i, key in enumerate(keylist):
        if key not in whereat:
            # store either a new dictionary if we're not at the last key, or
            # the value itself
            whereat[key] = {} if (i < len(keylist) - 1) else np.array([value])
            whereat = whereat[key]
        else:
            if (i < len(keylist) - 1):
                # go one level deeper
                whereat = whereat[key]
                continue
            elif not np.array_equal(whereat[key], np.array([value])):
                msg = ('Mismatch in results for value {}, stored: {}, '
                       'current: {}.'.format(value, whereat[key], value))
                if allow_overwrite:
                    msg += '\nOverwriting...'
                    print(msg)
                    whereat[key] = np.array([value])
                else:
                    raise NanException(msg)


def calc_error(solver, tolerance, phir, phit, norm_rtol, norm_atol,
               condition_norm, ivp_norm, results={},
               use_fatode_err=False, use_rel_err=False, recalc=False):
    # calculate condition norm
    err = (np.abs(phit - phir) / (
        norm_atol + norm_rtol * np.abs(phir))).squeeze()
    err = np.linalg.norm(err, ord=condition_norm, axis=1)
    # calculate ivp norm
    err = np.linalg.norm(err, ord=ivp_norm, axis=0)

    # calculate fatode error
    err_fatode = (np.linalg.norm(np.abs(
        phit - phir), ord=condition_norm, axis=1) / np.linalg.norm(
        phir, ord=condition_norm, axis=1)).squeeze()
    err_fatode = np.linalg.norm(err_fatode, ord=ivp_norm, axis=0)

    # calculate simple relative error
    err_rel = (np.abs(phit - phir) / (
        1e-30 + np.abs(phir))).squeeze()
    err_rel = np.linalg.norm(err_rel, ord=condition_norm, axis=1)
    # calculate ivp norm
    err_rel = np.linalg.norm(err_rel, ord=ivp_norm, axis=0)

    ret_err = err
    if use_fatode_err:
        ret_err = err_fatode
    if use_rel_err:
        print('using relative error: ', err_rel)
        ret_err = err_rel

    if results:
        __store_check(results, ['err', solver, tolerance], err, recalc)
        __store_check(results, ['err_fatode', solver, tolerance], err_fatode, recalc)
        __store_check(results, ['err_rel', solver, tolerance], err_rel, recalc)

    return ret_err


def run_case(num, phir, test, test_problem,
             t_end, solver, tolerance, norm_rtol, norm_atol,
             condition_norm, ivp_norm, results={},
             num_repeats=5, use_fatode_err=False):
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

    if results:
        # store rtol / atol / etc. if not already there
        __store_check(results, 'condition_norm', condition_norm,
                      allow_overwrite=True)
        __store_check(results, 'ivp_norm', ivp_norm, allow_overwrite=True)
        __store_check(results, 'num', num, allow_overwrite=True)

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
        __store_check(results, ['test', solver, tolerance], phit)
        __store_check(results, ['runtimes', solver, tolerance], runtimes)

    if np.any(phit != phit):
        logger = logging.getLogger(__name__)
        logger.warn('NaN / Solution detected for step-size! '
                    'Try reducing starting step-size.')
        raise NanException()

    # calculate condition norm
    phir_limited = np.take(phir, np.arange(num), axis=0)
    err = calc_error(solver, tolerance, phir_limited, phit, norm_rtol=norm_rtol,
                     norm_atol=norm_atol, condition_norm=condition_norm,
                     ivp_norm=ivp_norm, results=results,
                     use_fatode_err=use_fatode_err)

    return runtimes, err


def run_validation(num, reference, ref_problem,
                   t_end, test_builder, solvers, tol_start=1e-4, tol_end=1e-15,
                   norm_rtol=1e-06, norm_atol=1e-10, condition_norm=2,
                   ivp_norm=np.inf, plot_filename='',
                   error_filename='', reuse=False, recalculate_error=False):
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

        def dictify(npz):
            npz = dict(npz)
            for key in npz:
                if isinstance(npz[key], np.ndarray) and \
                        npz[key].dtype == np.dtype(object):
                    assert npz[key].size == 1
                    # nested dict
                    npz[key] = dictify(npz[key].item())
                elif isinstance(npz[key], dict):
                    # nested dict
                    npz[key] = dictify(npz[key])
                else:
                    continue
            return npz

        # load from npz
        results = dictify(np.load(error_filename))
        phir = results['phi_valid']
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

    def __check(keylist, test):
        whereat = results
        for i, key in enumerate(keylist):
            if key not in whereat:
                return False
            if i == len(keylist) - 1:
                # run test
                return test(whereat[key])
            else:
                whereat = whereat[key]

    def __delete(keylist):
        whereat = results
        for i, key in enumerate(keylist):
            if key not in whereat:
                return
            if i == len(keylist) - 1:
                del whereat[key]
            else:
                whereat = whereat[key]

    def __load(keylist):
        whereat = results
        for i, key in enumerate(keylist):
            if key not in whereat:
                raise Exception(key + ' not in results!')
            if i == len(keylist) - 1:
                return whereat[key]
            else:
                whereat = whereat[key]

    built_any = False
    for j, solver in enumerate(solvers):
        for i, tol in enumerate(tols):
            if reuse and __check(['test', solver, tol], lambda x: x.shape[1] == num):
                runtimes[i] = __load(['runtimes', solver, tol])
                # we have data for this solution
                if recalculate_error:
                    phir_limited = np.take(phir, np.arange(num), axis=0)
                    phit = __load(['test', solver, tol]).squeeze()
                    errs[i] = calc_error(solver, tol, phir_limited, phit,
                                         norm_rtol=norm_rtol, norm_atol=norm_atol,
                                         condition_norm=condition_norm,
                                         ivp_norm=ivp_norm, results=results,
                                         recalc=True)
                continue
            elif reuse:
                # the stored data doesn't match
                __delete(['test', solver, tol])
                __delete(['err', solver, tol])
                __delete(['err_fatode', solver, tol])

            testivp, test_problem, test = test_builder(
                tol, tol, iteration=i and built_any, solver=solver)
            built_any = True

            if test_order is None:
                test_order = test.solver_order()

            try:
                runtimes[i], errs[i] = run_case(
                    num, phir, test, test_problem, t_end,
                    solver, tol, norm_rtol=norm_rtol, norm_atol=norm_atol,
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
                             num_solvers=len(solvers),
                             index=j)

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
                        help='If supplied, attempt to reuse the completed runs '
                             'from the `error_filename`.')

    parser.add_argument('-rce', '--recalculate_error',
                        default=False,
                        action='store_true',
                        help='If supplied, recalculate the error stored in the '
                             '`error_filename`, using the (potentially) new norms.')

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
