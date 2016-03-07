/*radau2a_props.cuh
 *Various macros controlling behaviour of RADAU2A algorithm
 * \file RADAU2A_props.cuh
 * \author Nicholas Curtis
 * \date 03/10/2015
 */

#ifndef RADAU2A_PROPS_CUH
#define RADAU2A_PROPS_CUH

#include "header.cuh"
#include <cuComplex.h>

#define STRIDE (NSP)

struct solver_memory
{
	double* E1;
	cuDoubleComplex* E2;
	int* ipiv1;
	int* ipiv2;
	double* Z1;
	double* Z2;
	double* Z3;
	double* DZ1;
	double* DZ2;
	double* DZ3;
	double* CONT;
	double* y0;
	double* F0;
	double* work1;
	double* work2;
	double* work3;
	cuDoubleComplex* work4;
	int* result;
};

enum errorCodes {
	success = 0,
	err_consecutive_steps = 1,
	max_steps_exceeded = 2,
	h_plus_t_equals_h = 3,
	newton_max_iterations_exceeded = 4
};


#endif