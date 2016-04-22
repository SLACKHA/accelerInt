/* rk78_init.cpp
*  Implementation of the necessary initialization for Boost's RK78-Felhberg solver
 * \file rk78_init.cpp
 *
 * \author Nicholas Curtis
 * \date 04/15/2016
 *
 */

//wrapper code
#include "rk78_typedefs.hpp"

std::vector<state_type*> state_vectors;
std::vector<rhs_eval*> evaluators;
std::vector<stepper*> steppers;
std::vector<controller*> controllers;
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

void initialize_solver(int num_threads) {
	//create the necessary state vectors and evaluators
	for (int i = 0; i < num_threads; ++i)
	{
		state_vectors.push_back(new state_type(NSP, 0.0));
		evaluators.push_back(new rhs_eval());
		steppers.push_back(new stepper());
		controllers.push_back(new controller(ATOL, RTOL, *steppers[i]));
	}
	
}

void cleanup_solver(int num_threads) {
	for (int i = 0; i < state_vectors.size(); ++i)
	{
		delete state_vectors[i];
		delete evaluators[i];
		delete steppers[i];
		delete controllers[i];
	}
#ifdef STIFFNESS_MEASURE
	fclose(stepsizes);
#endif
}

const char* solver_name() {
	const char* name = "rk78-int";
	return name;
}

void init_solver_log() {
#ifdef STIFFNESS_MEASURE
	stepsizes = fopen("stepsize_log.txt", "w");
#endif
}
void solver_log() {
#ifdef STIFFNESS_MEASURE
	for (int i = 0; i < max_stepsize.size(); ++i){
		fprintf(stepsizes, "%d\t%.16e\n", i, max_stepsize[i]);
	}
#endif
}