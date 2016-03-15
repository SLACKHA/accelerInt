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

 #include "solver_props.cuh"
 #include "header.cuh"

 size_t required_solver_size();
 void init_solver_log();
 void solver_log();
 void initialize_solver(const int, solver_memory**, solver_memory**);
 void cleanup_solver(solver_memory**, solver_memory**);
 const char* solver_name();

 #endif
