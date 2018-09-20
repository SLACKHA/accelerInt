/**
 * \file
 * \brief Implementation of the necessary initialization for Boost's RK78-Felhberg solver
 *
 * \author Nicholas Curtis
 * \date 04/15/2016
 *
 */

//wrapper code
#include "rk78_typedefs.hpp"

#ifdef GENERATE_DOCS
namespace rk78 {
#endif

//! State vector containers for boost
std::vector<state_type*> state_vectors;
//! RHS wrappers for boost
std::vector<rhs_eval*> evaluators;
//! Addaptive timesteppers
std::vector<stepper*> steppers;
//! ODE controllers
std::vector<controller> controllers;
#ifdef STIFFNESS_MEASURE
std::vector<double> max_stepsize;
#include <stdio.h>
FILE* stepsizes;
#endif

extern "C" void initialize_solver(int);
extern "C" void cleanup_solver(int);
extern "C" const char* solver_name();
extern "C" void init_solver_log();
extern "C" void solver_log();

/*! \fn void initialize_solver(int num_threads)
   \brief Initializes the solver
   \param num_threads The number of OpenMP threads to use
*/
void initialize_solver(int num_threads) {
	//create the necessary state vectors and evaluators
	for (int i = 0; i < num_threads; ++i)
	{
		state_vectors.push_back(new state_type(NSP, 0.0));
		evaluators.push_back(new rhs_eval());
		steppers.push_back(new stepper());
		controllers.push_back(make_controlled<stepper>(ATOL, RTOL, *steppers[i]));
	}
}

/*!
   \brief Cleans up the created solvers
   \param num_threads The number of OpenMP threads used

   Frees and cleans up allocated RK78 memory.
*/
void cleanup_solver(int num_threads) {
	for (int i = 0; i < state_vectors.size(); ++i)
	{
		delete state_vectors[i];
		delete evaluators[i];
		delete steppers[i];
		controllers.pop_back();
	}
#ifdef STIFFNESS_MEASURE
	fclose(stepsizes);
#endif
}

/*!
   \fn char* solver_name()
   \brief Returns a descriptive solver name
*/
const char* solver_name() {
	const char* name = "rk78-int";
	return name;
}

/*!
   \fn init_solver_log()
   \brief Initializes solver specific items for logging

   Initializes stepsize logging for stiffness measurement
*/
void init_solver_log() {
#ifdef STIFFNESS_MEASURE
	stepsizes = fopen("stepsize_log.txt", "w");
#endif
}

/*!
   \fn solver_log()
   \brief Executes solver specific logging tasks
*/
void solver_log() {
#ifdef STIFFNESS_MEASURE
	for (int i = 0; i < max_stepsize.size(); ++i){
		fprintf(stepsizes, "%d\t%.16e\n", i, max_stepsize[i]);
	}
#endif
}

#ifdef GENERATE_DOCS
}
#endif
