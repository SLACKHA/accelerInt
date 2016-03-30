/* exponential_linear_algebra.c
 * Implementation of various linear algebra functions needed in the exponential integrators
 * \file exponential_linear_algebra.c
 *
 * \author Nicholas Curtis
 * \date 03/09/2015
 *
 */

#ifndef EXPONENTIAL_LINEAR_ALGEBRA_CUH
#define EXPONENTIAL_LINEAR_ALGEBRA_CUH

#include "header.cuh"
#include "solver_options.h"
#include "solver_props.cuh"

///////////////////////////////////////////////////////////////////////////////

/** Matrix-vector multiplication of a matrix sized MxM and a vector Mx1
 * 
 * \param[in]		m 		size of the matrix
 * \param[in]		A		matrix of size MxM
 * \param[in]		V		vector of size Mx1
 * \param[out]		Av		vector that is A * v
 */
__device__
void matvec_m_by_m (const int,
	const double * const __restrict__,
	const double * const __restrict__, double * const __restrict__);

///////////////////////////////////////////////////////////////////////////////

/** Matrix-vector plus equals for a matrix of size MxM and vector of size Mx1
 * 
 *  That is, it returns (A + I) * v
 * 
 * \param[in]		m 		size of the matrix
 * \param[in]		A		matrix of size MxM
 * \param[in]		V		vector of size Mx1
 * \param[out]		Av		vector that is (A + I) * v
 */
__device__ void matvec_m_by_m_plusequal (const int,
	const double * const __restrict__, 
	const double * const __restrict__, double * const __restrict__);

/** Matrix-vector multiplication of a matrix sized NSPxM and a vector of size Mx1 scaled by a specified factor
 * 
 *  That is, it returns A * v * scale
 * 
 * \param[in]		m 		size of the matrix
 * \param[in]		scale 	a number to scale the multplication by
 * \param[in]		A		matrix
 * \param[in]		V		the vector
 * \param[out]		Av		vector that is A * V * scale
 */
__device__
void matvec_n_by_m_scale (const int,
	const double,
	const double * const __restrict__,
	const double * const __restrict__, double * const __restrict__ );


/** Matrix-vector multiplication of a matrix sized NSPxM and a vector of size Mx1 scaled by a specified factor
 *
 *  Computes the following:
 *  Av1 = A * V1 * scale[0]
 *  Av2 = A * V2 * scale[1]
 *  Av3 = A * V3 * scale[2] + V4 + V5
 * 
 * Performs inline matrix-vector multiplication (with unrolled loops)
 * 
 * \param[in]		m 		size of the matrix
 * \param[in]		scale 	a list of numbers to scale the multplication by
 * \param[in]		A		matrix
 * \param[in]		V		a list of 5 pointers corresponding to V1, V2, V3, V4, V5
 * \param[out]		Av		a list of 3 pointers corresponding to Av1, Av2, Av3
 */
__device__
void matvec_n_by_m_scale_special (const int,
	const double * __restrict__,
	const double * __restrict__,
	double * const __restrict__ *, double * __restrict__*);

/** Matrix-vector multiplication of a matrix sized NSPxM and a vector of size Mx1 scaled by a specified factor
 *
 *  Computes the following:
 *  Av1 = A * V1 * scale[0]
 *  Av2 = A * V2 * scale[1]
 * 
 * Performs inline matrix-vector multiplication (with unrolled loops)
 * 
 * \param[in]		m 		size of the matrix
 * \param[in]		scale 	a list of numbers to scale the multplication by
 * \param[in]		A		matrix
 * \param[in]		V		a list of 2 pointers corresponding to V1, V2
 * \param[out]		Av		a list of 2 pointers corresponding to Av1, Av2
 */
__device__
void matvec_n_by_m_scale_special2 (const int,
	const double* __restrict__,
	const double* __restrict__,
	double* const __restrict__ *, double* __restrict__ *);

/** Matrix-vector multiplication of a matrix sized NSPxM and a vector of size Mx1 scaled by a specified factor and added to another vector
 * 
 * Computes A * V * scale + add
 * 
 * \param[in]		m 		size of the matrix
 * \param[in]		scale 	a number to scale the multplication by
 * \param[in]		add 	the vector to add to the result
 * \param[in]		A		matrix
 * \param[in]		V		the vector
 * \param[out]		Av		vector that is A * V * scale + add
 */
__device__
void matvec_n_by_m_scale_add (const int,
	const double,
	const double* __restrict__,
	const double* __restrict__,
	double* __restrict__, const double* __restrict__);

///////////////////////////////////////////////////////////////////////////////

/** Matrix-vector multiplication of a matrix sized NSPxM and a vector of size Mx1 scaled by a specified factor and adds and subtracts the specified vectors
 *  note, the addition is twice the specified vector
 * 
 *  Computes scale * A * V + 2 * add - sub
 * 
 * \param[in]		m 		size of the matrix
 * \param[in]		scale 	a number to scale the multplication by
 * \param[in]		A		matrix
 * \param[in]		V		the vector
 * \param[out]		Av		vector that is scale * A * V + 2 * add - sub
 * \param[in]		add 	the vector to add to the result
 * \param[in]		sub 	the vector to subtract from the result
 */
__device__
void matvec_n_by_m_scale_add_subtract (const int,
	const double,
	const double* __restrict__,
	const double*,
	double* __restrict__,
	const double* __restrict__, const double* __restrict__);

///////////////////////////////////////////////////////////////////////////////

/** Get scaling for weighted norm
 * 
 *	Computes 1.0 / (ATOL + MAX(|y0|, |y1|) * RTOL)
 *
 * \param[in]		y0		values at current timestep
 * \param[in]		y1		values at next timestep
 * \param[out]		sc	array of scaling values
 */
__device__
void scale (const double* __restrict__, 
	const double* __restrict__, double* __restrict__);

///////////////////////////////////////////////////////////////////////////////

/** Get scaling for weighted norm for the initial timestep (used in krylov process)
 * 
 *  Computes 1.0 / (ATOL + |y1| * RTOL)
 *
 * \param[in]		y0		values at current timestep
 * \param[out]		sc	array of scaling values
 */
__device__
void scale_init (const double* __restrict__, double* __restrict__);

///////////////////////////////////////////////////////////////////////////////

/** Perform weighted norm
 *
 *  Computes sqrt(sum((nums^2) * sc) / NSP)
 * 
 * \param[in]		nums	values to be normed
 * \param[in]		sc		scaling array for norm
 * \return			norm	weighted norm
 */
__device__
double sc_norm (const double* __restrict__, const double* __restrict__);

/** Computes and returns the two norm of a vector
 *
 *  sqrt(sum(v^2))
 *
 *	\param[in]		v 		the vector
 */
__device__
double two_norm(const double* __restrict__);

/** Normalize the input vector using a 2-norm
 * 
 *  v_out = v / |v|_2
 *
 * \param[in]		v		vector to be normalized
 * \param[out]		v_out	where to stick the normalized part of v (in a column)
 */
__device__
double normalize (const double* __restrict__, double* __restrict__);

/** Performs the dot product of the w vector with the given vector
 * 
 *	returns Vm \dot w
 *
 * \param[in]		w   	the vector with with to dot
 * \param[in]		Vm		the subspace vector
 * \out 			sum		the dot product of the specified vectors
 */
__device__
double dotproduct(const double* __restrict__, const double* __restrict__);

/** Subtracts Vm scaled by s from w
 * 
 *  w -= Vm * s
 *
 * \param[in]		s   	the scale multiplier to use
 * \param[in]		Vm		the subspace matrix
 * \param[out]		w 		the vector to subtract from
 */
__device__
void scale_subtract(const double, const double* __restrict__, double* __restrict__);

/** Sets Vm to s * w
 * 
 *	Vm = s * w
 *
 * \param[in]		stride 	number of columns in Vm
 * \param[in]		s   	the scale multiplier to use
 * \param[in]		w 		the vector to use as a base
 * \param[out]		Vm		the subspace matrix to set
 */
__device__
void scale_mult(const double, const double* __restrict__, double* __restrict__);

#endif