# test our indexing macros using pyopencl
from collections import OrderedDict
from string import Template
import os
import numpy as np
import pyopencl as cl
from optionloop import OptionLoop

path = os.path.abspath(os.path.dirname(__file__))
ocl_path = os.path.join(path, os.path.pardir, os.path.pardir, 'generic', 'opencl')

# create testing loop
loop = OptionLoop(OrderedDict(
    order=['C', 'F'],
    vector_width=[0, 4],
    block_width=[0, 4],
    num_work_groups=[1, 4],
    neq=[5]
    ))

with open(os.path.join(ocl_path, 'solver.h'), 'r') as file:
    base_src = file.read()

for state in loop:
    if state['vector_width'] and state['block_width']:
        continue

    platforms = cl.get_platforms()
    cpud = None
    alld = None
    for p in platforms:
        devices = p.get_devices(device_type=cl.device_type.CPU)
        if devices:
            cpud = devices[0]
        devices = p.get_devices(device_type=cl.device_type.ALL)
        if devices:
            alld = devices[0]

    if cpud:
        # create subdevice w/ correct # of threads
        device = cpud.create_sub_devices([
            cl.device_partition_property.BY_COUNTS, state['num_work_groups']])[0]
    elif alld:
        device = alld
    else:
        raise Exception('No OpenCL devices found!')

    ctx = cl.Context(devices=[device])
    queue = cl.CommandQueue(ctx)

    # define source
    src = Template(
        """
            #define __order '${order}'
            #define use_explicit_simd ${vector_width}
            #if use_explicit_simd
                #define __ValueSize use_explicit_simd
            #endif
            #define use_block_width ${block_width}
            #if use_block_width
                #define __blockSize use_block_width
            #endif
            #define neq ${neq}
        """).safe_substitute(**state)

    full_src = src + base_src + \
        """
            void __kernel
            __attribute__((reqd_work_group_size(__blockSize, 1, 1)))
            tester (__global double * __restrict__ zero_d,
                    __global double * __restrict__ one_d)
            {
                #define paramIndex (__getIndex1D(1, 0))
                #define phiIndex(idx) (__getIndex1D(neq, (idx)))
                zero_d[paramIndex] = paramIndex;
                for (int i = 0; i < neq; ++i)
                {
                    one_d[phiIndex(i)] = phiIndex(i);
                }
            }
        """

    # print(full_src)
    # build
    prg = cl.Program(ctx, full_src).build()

    gsize = state['num_work_groups']
    lsize = 1
    if state['block_width']:
        gsize *= state['block_width']
        lsize = state['block_width']

    # build memory
    zero_d_np = np.zeros(gsize, dtype=np.float64)
    one_d_np = np.zeros((gsize, state['neq']),
                        order=state['order'], dtype=np.float64)
    mf = cl.mem_flags
    zero_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=zero_d_np)
    one_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=one_d_np)

    # run and copy-back
    prg.tester(queue, (gsize,), (lsize,), zero_g, one_g)

    r_zerod = np.empty_like(zero_d_np)
    r_oned = np.empty_like(one_d_np)
    cl.enqueue_copy(queue, r_zerod, zero_g)
    cl.enqueue_copy(queue, r_oned, one_g)

    # test
    assert np.array_equal(r_zerod, np.arange(gsize, dtype=np.float64))
    assert np.array_equal(r_oned.flatten(state['order']),
                          np.arange(gsize * state['neq'], dtype=np.float64))
