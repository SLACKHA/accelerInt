/**
 * \file
 * \brief Various macros controlling behaviour of RADAU2A algorithm
 * \author Nicholas Curtis
 * \date 03/10/2015
 */

#ifndef RADAU2A_PROPS_CUH
#define RADAU2A_PROPS_CUH

#include "header.cuh"
#include <cuComplex.h>
#include <stdio.h>

#ifdef GENERATE_DOCS
namespace radau2acu {
#endif

//! the matrix dimensions
#define STRIDE (NSP)

//! Memory required for Radau-IIa GPU solver
struct solver_memory
{
	//! The matrix for the non-complex system solve
	double* E1;
	//! The matrix for the complex system solve
	cuDoubleComplex* E2;
	//! The error weight scaling vector
	double* scale;
	//! Pivot indicies for E1
	int* ipiv1;
	//! Pivot indicies for E2
	int* ipiv2;
	//! Stage 1 values
	double* Z1;
	//! Stage 2 values
	double* Z2;
	//! Stage 3 values
	double* Z3;
	//! Change in stage 1 values
	double* DZ1;
	//! Change in stage 2 values
	double* DZ2;
	//! Change in stage 3 values
	double* DZ3;
	//! Quadratic interpolate
	double* CONT;
	//! Initial state vectors
	double* y0;
	//! work vector
	double* work1;
	//! work vector
	double* work2;
	//! work vector
	double* work3;
	//! complex work vector
	cuDoubleComplex* work4;
	//! array of return codes @see RKCU_ErrCodes
	int* result;
};

/**
 * \addtogroup CUErrorCodes Return codes of Integrators
 * @{
 */
/**
 * \defgroup RKCU_ErrCodes Return codes of GPU Radau-IIa Integrator
 * @{
 */

//! Successful time step
#define EC_success (0)
//! Maximum number of consecutive internal timesteps with error reached @see #Max_consecutive_errs
#define EC_consecutive_steps (1)
//! Maximum number of internal timesteps exceeded @see #Max_no_steps
#define EC_max_steps_exceeded (2)
//! Timescale reduced such that t + h == t in floating point math
#define EC_h_plus_t_equals_h (3)
//! Maximum allowed Newton Iteration steps exceeded @see #NewtonMaxit
#define EC_newton_max_iterations_exceeded (4)

/**
 * @}
 */
/**
 * @}
 */

#ifdef GENERATE_DOCS
}
#endif

#endif