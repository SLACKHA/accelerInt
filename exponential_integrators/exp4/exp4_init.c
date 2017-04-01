/*!
 * \file
 * \brief Implementation of the necessary initialization for the EXP4 method
 *
 * \author Nicholas Curtis
 * \date 03/09/2015
 *
 */

#include "rational_approximant.h"

#ifdef GENERATE_DOCS
namespace exp4 {
#endif

/*!
   \fn init_solver_log()
   \brief Initializes the Krylov subspace logging files (if LOG_OUTPUT is defined)
   @see solver_options.cuh
*/
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

/*! \fn void initialize_solver(int num_threads)
   \brief Initializes the solver
   \param num_threads The number of OpenMP threads to use
*/
void initialize_solver(int num_threads) {
    //Solves for the poles and residuals used for the Rational Approximants in the Krylov subspace methods
 	find_poles_and_residuals();
}

/*!
   \fn char* solver_name()
   \brief Returns a descriptive solver name
*/
const char* solver_name() {
 	const char* name = "exp4-int";
 	return name;
}

void cleanup_solver(int num_threads) {
}

#ifdef GENERATE_DOCS
}
#endif
