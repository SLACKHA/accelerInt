/* solver_init.cuh
 * the generic include file for all exponential solvers
 * \file solver_init
 *
 * \author Nicholas Curtis
 * \date 03/09/2015
 *
 * Contains declarations that all individual solvers must implement
 */

 #ifndef SOLVER_INIT_CUH
 #define SOLVER_INIT_CUH

 void init_solver_log();
 void solver_log();
 void initialize_solver();
 void cleanup_solver();
 const char* solver_name();

 #endif
