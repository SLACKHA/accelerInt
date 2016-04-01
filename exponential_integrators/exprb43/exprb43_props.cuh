/*rb43_props.cuh
 *Various macros controlling behaviour of RB43 algorithm
 * \file rb43_props.cuh
 * \author Nicholas Curtis
 * \date 03/10/2015
 */

#ifndef RB43_PROPS_CUH
#define RB43_PROPS_CUH

#include "header.cuh"
#include <cuComplex.h>
#include <stdio.h>

//if defined, uses (I - h * Hm)^-1 to smooth the krylov error vector
//#define USE_SMOOTHED_ERROR
//max order of the phi functions (i.e. for error estimation)
#define P 4
//order of embedded methods
#define ORD 3.0
#define M_MAX NSP
#define STRIDE (M_MAX + P)
#define MAX_STEPS (10000)

struct solver_memory
{
	double* sc;
	double* work1;
	double* work2;
	double* work3;
	double* gy;
	double* Hm;
	double* phiHm;
	double* Vm;
	double* savedActions;
	int* ipiv;
	cuDoubleComplex* invA;
	cuDoubleComplex* work4;
	int* result;
};

#define EC_success (0)
#define EC_consecutive_steps (1)
#define EC_max_steps_exceeded (2)
#define EC_h_plus_t_equals_h (3)

#endif