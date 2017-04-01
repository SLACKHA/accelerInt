/**
 * \file
 * \brief Header definitions for solver initialization routins
 *
 * \author Nicholas Curtis
 * \date 03/09/2015
 *
 * Contains declarations for initialization routines that all individual solvers must implement
 */

 #ifndef SOLVER_INIT_CUH
 #define SOLVER_INIT_CUH

 #include "solver_props.cuh"
 #include "header.cuh"

 #ifdef GENERATE_DOCS
 namespace genericcu{
 #endif

 size_t required_solver_size();
 void init_solver_log();
 void solver_log();
 void initialize_solver(const int, solver_memory**, solver_memory**);
 void cleanup_solver(solver_memory**, solver_memory**);
 const char* solver_name();

 #ifdef GENERATE_DOCS
 }
 #endif

 #endif
