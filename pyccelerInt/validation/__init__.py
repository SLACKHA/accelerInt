import numpy as np

from pyccelerInt import create_integrator, build_problem


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

    def plot(self, step_sizes, errors, end_time=None, label=''):
        """
        Plot the validation curve for this problem

        Parameters
        ----------
        step_sizes: :class:`numpy.ndarray`
            The array of step sizes
        errors: :class:`numpy.ndarray`
            The array of normalized errors to plot
        """

        raise NotImplementedError


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


def run_case(num, phir, test, test_problem,
             t_end, step_size,
             norm_rtol=1e-6, norm_atol=1e-10,
             condition_norm=2,
             ivp_norm=np.inf):
    """
    Run the :param:`validation` and :param:`test` integrators over :param:`num`
    problems, to end time :param:`t_end`, using constant integration
    :param:`step_size`'s (for the test integrator)

    Parameters
    ----------
    num: int
        The number of individual IVPs to execute
    reference: :class:`PyIntegrator`
        The reference integrator
    ref_problem: :class:`Problem`
        The constructed problem for the reference integrator
    test: :class:`PyIntegrator`
        The integrator to validate
    test_problem: :class:`Problem`
        The constructed problem for the test integrator
    t_end: float
        The integration end time
    step_size: float
        The constant integration step-size to use for the test-integrator
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

    # run test
    _, phit = test_problem.run(test, num, t_end, t_start=0,
                               t_step=step_size, return_state=True)

    # calculate condition norm
    err = np.abs(phit - phir) / (norm_atol + phir * norm_rtol)
    err = np.linalg.norm(err, ord=condition_norm, axis=1)

    # calculate ivp norm
    err = np.linalg.norm(err, ord=ivp_norm, axis=0)

    return err


def run_validation(num, reference, ref_problem,
                   t_end, test_builder, step_start=1e-3, step_end=1e-8,
                   norm_rtol=1e-6, norm_atol=1e-10, condition_norm=2,
                   ivp_norm=np.inf):
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
        A Callable that takes as an argument and returns the result of
        :func:`build_case`

    Returns
    -------
    step_sizes: :class:`np.ndarray`
        The constant integration step-sizes used for the test-case
    err: :class:`np.ndarray`
        The error norms calculated for the problem (:see:`run_case`) for each
        step-size
    """

    # run reference problem once
    # run validation
    _, phir = ref_problem.run(reference, num, t_end, t_start=0, return_state=True)

    start = np.rint(np.log10(step_start))
    end = np.rint(np.log10(step_end))
    # determine direction of progression, and ensure that the final step-size is
    # included
    step = -1 if end < start else 1
    end += step
    steps = np.arange(start, end, step)
    errs = np.zeros(steps.size)

    for i, size in enumerate(steps):
        steps[i] = np.power(10., size)

        testivp, test_problem, test = test_builder(steps[i])

        errs[i] = run_case(num, phir,
                           test, test_problem,
                           t_end, steps[i],
                           norm_rtol=norm_rtol, norm_atol=norm_atol,
                           condition_norm=condition_norm, ivp_norm=ivp_norm)

        del testivp
        del test_problem
        del test

        print(steps[i], errs[i])

    return steps, errs
