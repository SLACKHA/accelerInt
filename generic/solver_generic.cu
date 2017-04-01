/**
 * \file
 * \brief the generic integration driver for the GPU solvers
 *
 * \author Nicholas Curtis
 * \date 03/10/2015
 *
 */

#include "solver.cuh"
#include "header.cuh"
#include "gpu_macros.cuh"

#ifdef GENERATE_DOCS
 namespace genericcu {
#endif

/**
 * \brief Generic driver for the GPU integrators
 * \param[in]       NUM             The (non-padded) number of IVPs to integrate
 * \param[in]       t               The current system time
 * \param[in]       t_end           The IVP integration end time
 * \param[in]       pr_global       The system constant variable (pressures / densities)
 * \param[in,out]   y_global        The system state vectors at time t.  Returns system state vectors at time t_end
 * \param[in]       d_mem           The mechanism_memory struct that contains the pre-allocated memory for the RHS \ Jacobian evaluation
 * \param[in]       s_mem           The solver_memory struct that contains the pre-allocated memory for the solver
 */
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

#ifdef GENERATE_DOCS
 }
#endif