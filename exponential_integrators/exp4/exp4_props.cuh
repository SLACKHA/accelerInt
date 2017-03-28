/*!
 * \file exp4_props.cuh
 * \brief Various macros controlling behaviour of EXP4 algorithm
 * \author Nicholas Curtis
 * \date 03/10/2015
 */

#ifndef EXP4_PROPS_CUH
#define EXP4_PROPS_CUH

#include "header.cuh"
#include <cuComplex.h>
#include <stdio.h>

#ifdef GENERATE_DOCS
namespace exp4cu {
#endif

//#define USE_SMOOTHED_ERROR ///if defined, uses (I - h * Hm)^-1 to smooth the krylov error vector

//! max order of the phi functions (for error estimation)
#define P 1
//! order of embedded methods
#define ORD 3.0
//! maximum Krylov dimension (without phi order)
#define M_MAX NSP
//! Krylov matrix stride
#define STRIDE (M_MAX + P)
//! Maximum allowed internal timesteps per integration step
#define MAX_STEPS (10000)
//! Number of consecutive errors on internal integration steps allowed before exit
#define MAX_CONSECUTIVE_ERRORS (5)

/*!
 * \brief Structure containing memory needed for EXP4 algorithm
 */
struct solver_memory
{
	double* sc; /// the scaled error coefficients
	double* work1; /// a work array
	double* work2; /// a work array
	double* work3; /// a work array
	cuDoubleComplex* work4; /// a (complex) work array
	double* Hm; /// The Hessenberg Kyrlov subspace array for EXP4, to take the exponential action on
	double* phiHm; /// the exponential Krylov subspace array for EXP4
	double* Vm; /// the Arnoldi basis array
	double* k1; /// the stage 1 results
	double* k2; /// the stage 2 results
	double* k3; /// the stage 3 results
	double* k4; /// the stage 4 results
	double* k5; /// the stage 5 results
	double* k6; /// the stage 6 results
	double* k7; /// the stage 7 results
	int* ipiv; /// the pivot indicies
	cuDoubleComplex* invA; /// the inverse of the Hessenberg Krylov subspace
	int* result; /// an array of integration results for the various threads @see exp4cu_ErrCodes
};

/**
 * \defgroup exp4cu_ErrCodes Return codes of GPU EXPRB43 integrator
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
#ifdef GENERATE_DOCS
}
#endif

#endif