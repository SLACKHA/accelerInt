/*solver_generic.cu
 * the generic integration driver for all GPU solvers
 * \file solver_generic.cu
 *
 * \author Nicholas Curtis
 * \date 03/10/2015
 *
 */

#include "header.h"
#include "solver.cuh"
#include "gpu_memory.cuh"
#include "launch_bounds.cuh"

#define T_ID (threadIdx.x + (blockDim.x * blockIdx.x))
#define GRID_SIZE (blockDim.x * gridDim.x)

 __global__ void
 __launch_bounds__(TARGET_BLOCK_SIZE, TARGET_BLOCKS)
intDriver (const int NUM, const double t, const double t_end,
                const double *pr_global, double *y_global)
{
    if (T_ID < NUM)
    {

#ifndef GLOBAL_MEM
        // local array with initial values
        double y_local[NN];
        double pr_local = pr_global[T_ID];

        // load local array with initial values from global array
#pragma unroll
        for (int i = 0; i < NN; i++)
        {
            y_local[i] = y_global[T_ID + i * GRID_SIZE];
        }
#endif
        // call integrator for one time step
        integrate (t, t_end, pr_local, y_local);

#ifndef GLOBAL_MEM
        // update global array with integrated values
#pragma unroll
        for (int i = 0; i < NN; i++)
        {
            y_global[T_ID + i * GRID_SIZE] = y_local[i];
        }
#endif

    }

} // end intDriver