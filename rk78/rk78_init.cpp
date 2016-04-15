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
	}
	
}

void cleanup_solver(int num_threads) {
	for (int i = 0; i < state_vectors.size(); ++i)
	{
		delete state_vectors[i];
		delete evaluators[i];
		delete steppers[i];
	}
}

const char* solver_name() {
	const char* name = "rk78-int";
	return name;
}

void init_solver_log() {
	
}
void solver_log() {
	
}