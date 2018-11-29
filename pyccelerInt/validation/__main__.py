from pyccelerInt.validation import build_case, run_validation, build_parser


def get_case(case):
    if case == 'vdp':
        from pyccelerInt.validation.van_der_pol import VDP_valid as case
    elif case == 'pyjac':
        from pyccelerInt.validation.pyJac import Ignition_valid as case
    else:
        raise NotImplementedError

    return case


if __name__ == '__main__':
    args = build_parser()

    case = get_case(args.case)

    # build reference solver
    refivp, refp, refi = build_case(
        case, rtol=1e-15, atol=1e-20, lang='c',
        integrator_type='CVODES', reuse=args.reuse,
        order='F', max_steps=int(args.max_steps),
        num_threads=args.num_threads)

    # check the integrator types
    solvers = [x.strip() for x in args.int_type.split(',')]

    end_time = args.end_time
    if not end_time:
        end_time = refp.get_default_endtime()

    def builder(rtol, atol, solver, iteration=0):
        return build_case(
            case, args.language,
            integrator_type=solver,
            reuse=args.reuse or iteration > 0,
            vector_size=args.vector_size,
            block_size=args.block_size,
            platform=args.platform, order=args.order,
            use_queue=args.use_queue,
            device_type=args.device_type,
            max_steps=args.max_steps,
            num_threads=args.num_threads,
            rtol=rtol,
            atol=atol)

    ts = refp.tol_start if args.starting_tolerance is None else \
        args.starting_tolerance
    te = refp.tol_end if args.ending_tolerance is None else \
        args.ending_tolerance
    run_validation(
        args.num_ivp, refi, refp,
        end_time, builder, solvers,
        tol_start=ts,
        tol_end=te,
        plot_filename=args.plot_filename,
        error_filename=args.error_filename,
        reuse=args.reuse_validation,
        recalculate_error=args.recalculate_error)

    # and cleanup
    del refivp
    del refp
    del refi
