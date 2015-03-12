/* solver_init.cuh
 * the generic include file for all exponential solvers
 * \file solver_init
 *
 * \author Nicholas Curtis
 * \date 03/09/2015
 *
 * Contains declarations that all individual solvers must implement
 */

 #ifndef SOLVER_INIT_H
 #define SOLVER_INIT_H

 void initialize_solver(int num_threads);
 void cleanup_solver(int num_threads);
 const char* solver_name();

 #endif
