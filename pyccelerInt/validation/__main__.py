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
    refivp, refp, refi = build_case(case, 'c', is_reference=True,
                                    integrator_type='CVODES', reuse=args.reuse,
                                    order='F', max_steps=int(args.max_steps),
                                    reference_rtol=1e-15, reference_atol=1e-20,
                                    num_threads=args.num_threads)

    end_time = args.end_time
    if not end_time:
        end_time = refp.get_default_endtime()

    def builder(stepsize, iteration=0):
        return build_case(case, args.language,
                          is_reference=False,
                          integrator_type=args.int_type,
                          reuse=args.reuse or iteration > 0,
                          vector_size=args.vector_size,
                          block_size=args.block_size,
                          platform=args.platform, order=args.order,
                          use_queue=args.use_queue,
                          device_type=args.device_type,
                          max_steps=args.max_steps,
                          num_threads=args.num_threads,
                          constant_timestep=stepsize)

    run_validation(args.num_ivp, refi, refp,
                   end_time, builder, step_start=refp.step_start,
                   step_end=refp.step_end, label=args.int_type,
                   plot_filename=args.plot_filename,
                   error_filename=args.error_filename,
                   reuse=args.reuse_validation)

    # and cleanup
    del refivp
    del refp
    del refi
