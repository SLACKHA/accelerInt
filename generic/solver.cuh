/*solver.cuh
 * the generic main file for all GPU solvers
 * \file solver.cuh
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

 __global__ void 
 intDriver (const int,
 			const double,
 			const double,
            double const * const __restrict__,
            double * const __restrict__, 
            mechanism_memory const * const __restrict__, 
            solver_memory const * const __restrict__);

__device__ void integrate (const double,
						   const double,
						   const double,
						   double * const __restrict__,
						   mechanism_memory const * const __restrict__,
						   solver_memory const * const __restrict__);

__host__
void check_error(int num_conditions, int* code_arr);

 #endif