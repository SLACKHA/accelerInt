/**
 * \file
 * \brief the generic main file for all GPU solvers
 *
 * \author Nicholas Curtis
 * \date 03/10/2015
 *
 * Contains skeleton of all methods that need to be defined on a per solver basis.
 */

#ifndef SOLVER_CUH
#define SOLVER_CUH

 #include "solver_options.cuh"
 #include "solver_init.cuh"
 #include "solver_props.cuh"

#ifdef GENERATE_DOCS
 namespace genericcu {
#endif

 __global__
void intDriver (const int NUM,
                const double t,
                const double t_end,
                const double * __restrict__ pr_global,
                double * __restrict__ y_global,
                const mechanism_memory * __restrict__ d_mem,
                const solver_memory * __restrict__ s_mem);

__device__ void integrate (const double,
						   const double,
						   const double,
						   double * const __restrict__,
						   mechanism_memory const * const __restrict__,
						   solver_memory const * const __restrict__);

__host__
void check_error(int num_conditions, int* code_arr);

#ifdef GENERATE_DOCS
 }
#endif

#endif