/* rb43_init.cu
*  Implementation of the necessary initialization for the 4th order (3rd order embedded) Rosenbrock Solver
 * \file rb43_init.cu
 *
 * \author Nicholas Curtis
 * \date 03/09/2015
 *
 */

#include "rational_approximant.h"

 void initialize_solver(int num_threads) {
 	find_poles_and_residuals();
 }

 const char* solver_name() {
 	const char* name = "exp4-int";
 	return name;
 }

 void cleanup_solver(int num_threads) {
 	//nothing to do
 }