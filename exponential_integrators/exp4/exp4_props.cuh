/*exp4_props.cuh
 *Various macros controlling behaviour of EXP4 algorithm
 * \file exp4_props.cuh
 * \author Nicholas Curtis
 * \date 03/10/2015
 */

#ifndef EXP4_PROPS_CUH
#define EXP4_PROPS_CUH

#include "header.cuh"
 
//if defined, uses (I - h * Hm)^-1 to smooth the krylov error vector
//#define USE_SMOOTHED_ERROR
//max order of the phi functions (i.e. for error estimation)
#define P 1
//order of embedded methods
#define ORD 3.0
#define M_MAX NSP
#define STRIDE (M_MAX + P)

struct solver_memory
{
	double* sc;
	double* work1;
	double* work2;
	double* work3;
	double* Hm;
	double* phiHm;
	double* Vm;
	double* k1;
	double* k2;
	double* k3;
	double* k4;
	double* k5;
	double* k6;
	double* k7;
	int* ipiv;
	cuDoubleComplex* invA;
	int* result;
};

enum errorCodes {
	success = 0,
	err_consecutive_steps = 1,
	max_steps_exceeded = 2,
	h_plus_t_equals_h = 3
};

#endif