/* rb43_init.cu
*  Implementation of the necessary initialization for the 4th order (3rd order embedded) Rosenbrock Solver
 * \file rb43_init.cu
 *
 * \author Nicholas Curtis
 * \date 03/09/2015
 *
 */

#include "rational_approximant.h"

 void init_solver_log() {
 #ifdef LOG_OUTPUT
	//file for krylov logging
	FILE* logFile;
	//open and clear
	const char* f_name = solver_name();
	int len = strlen(f_name);
	char out_name[len + 17];
	sprintf(out_name, "log/%s-kry-log.txt", f_name);
	logFile = fopen(out_name, "w");

	char out_reject_name[len + 23];
	sprintf(out_reject_name, "log/%s-kry-reject.txt", f_name);    
	//file for krylov logging
	FILE* rFile;
	//open and clear
	rFile = fopen(out_reject_name, "w");
	fclose(logFile);
	fclose(rFile);
 #endif
 }

 void solver_log() {
 	
 }

 void initialize_solver(int num_threads) {
 	find_poles_and_residuals();
 }

 const char* solver_name() {
 	const char* name = "exp4-int";
 	return name;
 }

void cleanup_solver(int num_threads) {
 }