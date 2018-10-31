"""The main interface for the pyccelerInt examples"""


from pyccelerInt import create_integrator, have_plotter, build_problem
from pyccelerInt.validation import build_parser


def get_case(case):
    if case == 'vdp':
        from pyccelerInt.examples.van_der_pol import VDP as case
    elif case == 'pyjac':
        from pyccelerInt.examples.pyJac import Ignition as case
    else:
        raise NotImplementedError

    return case


if __name__ == '__main__':

    args = build_parser()

    problem, options = build_problem(get_case(args.case), args.language,
                                     args.int_type,
                                     reuse=args.reuse,
                                     vector_size=args.vector_size,
                                     block_size=args.block_size,
                                     platform=args.platform,
                                     order=args.order,
                                     logging=True,
                                     use_queue=args.use_queue,
                                     device_type=args.device_type,
                                     rtol=args.relative_tolerance,
                                     atol=args.absolute_tolerance,
                                     maximum_steps=int(args.max_steps))

    end_time = args.end_time
    if not end_time:
        end_time = problem.get_default_endtime()

    # get IVP and integrator
    ivp, integrator = create_integrator(problem, args.int_type, options,
                                        args.num_threads)

    print('Integrating {} IVPs with method {}, and {} threads...'.format(
        args.num_ivp, str(args.int_type), args.num_threads))

    # run problem
    time, phi = problem.run(integrator, args.num_ivp, end_time, t_start=0,
                            t_step=problem.get_default_stepsize(), return_state=True)

    print('Integration finished in {} (s)'.format(time / 1000.))

    if have_plotter():
        t, phi = integrator.state()
        problem.plot(0, t, phi)

    # and cleanup
    del ivp
    del options
    del integrator
