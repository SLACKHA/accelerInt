/**
 * \file
 * \brief Header definitions for solver initialization routins
 *
 * \author Nicholas Curtis
 * \date 03/09/2015
 *
 * Contains declarations for initialization routines that all individual solvers must implement
 */

 #ifndef SOLVER_INIT_H
 #define SOLVER_INIT_H

 #ifdef GENERATE_DOCS
 namespace generic{
 #endif

 void init_solver_log();
 void solver_log();
 void initialize_solver(int num_threads);
 void cleanup_solver(int num_threads);
 const char* solver_name();

 #ifdef GENERATE_DOCS
 }
 #endif

 #endif
