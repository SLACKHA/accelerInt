"""The main interface for the pyccelerInt examples"""

import argparse
import multiprocessing

from pyccelerInt import create_integrator, \
    have_plotter, build_problem, check_vector_width, check_block_size


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Run the pyccelerInt examples.')
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

    parser.add_argument('-q', '--use_queue',
                        default=True,
                        action='store_true',
                        dest='use_queue',
                        help='Use the queue-based integration drivers. '
                             'Only used for OpenCL.')

    parser.add_argument('-s', '--use_static',
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

    args = parser.parse_args()

    if args.case == 'vdp':
        from pyccelerInt.examples.van_der_pol import VDP as case
    else:
        from pyccelerInt.examples.pyJac import Ignition as case

    problem, options = build_problem(case, args.language, args.int_type,
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
                                     constant_timesteps=False,
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
