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

 #include "solver_options.h"
 #include "solver_props.h"
 #include "solver_init.cuh"

 __global__ void 
 intDriver (const int, const double, const double, 
                const double*, double*);

 __device__
 void integrate(const double, const double, const double, double*);

 #endif