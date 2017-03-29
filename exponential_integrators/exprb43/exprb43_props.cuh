/*!
 * \file exprb43_props.cuh
 * \brief Various macros controlling behaviour of RB43 algorithm
 * \author Nicholas Curtis
 * \date 03/10/2015
 */

#ifndef RB43_PROPS_CUH
#define RB43_PROPS_CUH

#include "header.cuh"
#include <cuComplex.h>
#include <stdio.h>

#ifdef GENERATE_DOCS
namespace exprb43cu {
#endif


//if defined, uses (I - h * Hm)^-1 to smooth the krylov error vector
//#define USE_SMOOTHED_ERROR
 //! max order of the phi functions (for error estimation)
#define P 4
//! order of embedded methods
#define ORD 3.0
//! maximum Krylov dimension (without phi order)
#define M_MAX NSP
//! Krylov matrix stride
#define STRIDE (M_MAX + P)
//! Maximum allowed internal timesteps per integration step
#define MAX_STEPS (100000)
//! Number of consecutive errors on internal integration steps allowed before exit
#define MAX_CONSECUTIVE_ERRORS (5)

struct solver_memory
{
	//! the scaled error coefficients
	double* sc;
	//! a work array
	double* work1;
	//! a work array
	double* work2;
	//! a work array
	double* work3;
	//! The difference between RHS function and the Jacobian state vector product
	double* gy;
	//! The Hessenberg Kyrlov subspace array for EXP4, to take the exponential action on
	double* Hm;
	//! the exponential Krylov subspace array for EXP4
	double* phiHm;
	//! the Arnoldi basis array
	double* Vm;
	//! Saved stage results
	double* savedActions;
	//! the pivot indicies
	int* ipiv;
	//! the inverse of the Hessenberg Krylov subspace
	cuDoubleComplex* invA;
	//! a (complex) work array
	cuDoubleComplex* work4;
	//! an array of integration results for the various threads @see exprb43cu_ErrCodes
	int* result;
};

/**
 * \addtogroup CUErrorCodes Return codes of Integrators
 * @{
 */
/**
 * \defgroup exprb43cu_ErrCodes Return codes of GPU EXP4 integrator
 * @{
 */

//! Successful integration step
#define EC_success (0)
//! Maximum consecutive errors on internal integration steps reached
#define EC_consecutive_steps (1)
//! Maximum number of internal integration steps reached
#define EC_max_steps_exceeded (2)
//! Timestep reduced such that update would have no effect on simulation time
#define EC_h_plus_t_equals_h (3)

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