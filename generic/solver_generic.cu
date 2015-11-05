/*solver_generic.cu
 * the generic integration driver for all GPU solvers
 * \file solver_generic.cu
 *
 * \author Nicholas Curtis
 * \date 03/10/2015
 *
 */

#include "header.cuh"
#include "solver.cuh"
#include "gpu_memory.cuh"

#define T_ID (threadIdx.x + (blockDim.x * blockIdx.x))
#define GRID_SIZE (blockDim.x * gridDim.x)

 __global__ void
intDriver (const int NUM, const double t, const double t_end,
                const double *pr_global, double *y_global)
{
    if (T_ID < NUM)
    {

        // local array with initial values
        double y_local[NSP];
        double pr_local = pr_global[T_ID];

        // load local array with initial values from global array
#pragma unroll
        for (int i = 0; i < NSP; i++)
        {
            y_local[i] = y_global[T_ID + i * GRID_SIZE];
        }
        // call integrator for one time step
        integrate (t, t_end, pr_local, y_local);

        // update global array with integrated values
#pragma unroll
        for (int i = 0; i < NSP; i++)
        {
            y_global[T_ID + i * GRID_SIZE] = y_local[i];
        }

    }

} // end intDriver