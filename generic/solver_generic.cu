/*solver_generic.cu
 * the generic integration driver for all GPU solvers
 * \file solver_generic.cu
 *
 * \author Nicholas Curtis
 * \date 03/10/2015
 *
 */

#include "solver.cuh"

#define T_ID (threadIdx.x + (blockDim.x * blockIdx.x))

 __global__
void intDriver (const int NUM,
                const double t,
                const double t_end,
                const double * __restrict__ pr_global,
                double * __restrict__ y_global,
                const mechanism_memory * __restrict__ d_mem,
                const solver_memory * __restrict__ s_mem)
{
    if (T_ID < NUM)
    {
        // call integrator for one time step
        integrate (t, t_end, pr_global[T_ID], d_mem->y, d_mem, s_mem);
    }
} // end intDriver