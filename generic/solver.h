/*solver.cuh
 * the generic main file for all GPU solvers
 * \file solver.cuh
 *
 * \author Nicholas Curtis
 * \date 03/10/2015
 *
 * Contains skeleton of all methods that need to be defined on a per solver basis.
 */

 #ifndef SOLVER_H
 #define SOLVER_H

 #include "solver_options.h"
 #include "solver_init.h"
 #include "solver_props.h"

 void intDriver (const int NUM, const double t, const double t_end, 
                const double* pr_global, double* y_global);

 int integrate(const double t_start, const double t_end, const double pr, double* y);

 #endif