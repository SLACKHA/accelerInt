/**
 * \file exponential_linear_algebra.cuh
 * \brief Definitions of various linear algebra functions needed in the exponential integrators
 *
 * \author Nicholas Curtis
 * \date 03/09/2015
 *
 */

#ifndef EXPONENTIAL_LINEAR_ALGEBRA_CUH
#define EXPONENTIAL_LINEAR_ALGEBRA_CUH

#include "header.cuh"
#include "solver_options.cuh"
#include "solver_props.cuh"

///////////////////////////////////////////////////////////////////////////////

/*!
 * \brief Matrix-vector multiplication of a matrix sized MxM and a vector Mx1
 * \param[in]		m 		size of the matrix
 * \param[in]		A		matrix of size MxM
 * \param[in]		V		vector of size Mx1
 * \param[out]		Av		vector that is A * v
 */
__device__
void matvec_m_by_m (const int m,
	const double * const __restrict__ A,
	const double * const __restrict__ V, double * const __restrict__ Av);

///////////////////////////////////////////////////////////////////////////////

/*!
 *
 * \brief Matrix-vector plus equals for a matrix of size MxM and vector of size Mx1.
 *  	  That is, it returns (A + I) * v
 *
 * \param[in]		m 		size of the matrix
 * \param[in]		A		matrix of size MxM
 * \param[in]		V		vector of size Mx1
 * \param[out]		Av		vector that is (A + I) * v
 */
__device__ void matvec_m_by_m_plusequal (const int m,
	const double * const __restrict__ A,
	const double * const __restrict__ V, double * const __restrict__ Av);

/*!
 *
 * \brief Matrix-vector multiplication of a matrix sized NSPxM and a vector of size Mx1 scaled by a specified factor
 *  	   That is, it returns A * v * scale
 *
 * \param[in]		m 		size of the matrix
 * \param[in]		scale 	a number to scale the multplication by
 * \param[in]		A		matrix
 * \param[in]		V		the vector
 * \param[out]		Av		vector that is A * V * scale
 */
__device__
void matvec_n_by_m_scale (const int m,
	const double scale,
	const double * const __restrict__ A,
	const double * const __restrict__ V, double * const __restrict__ Av);


/*!
 *  \brief Matrix-vector multiplication of a matrix sized NSPxM and a vector of size Mx1 scaled by a specified factor
 *
 *  Computes the following:
 *  \f$Av1 = A * V1 * scale[0]\f$,
 *  \f$Av2 = A * V2 * scale[1]\f$, and
 *  \f$Av3 = A * V3 * scale[2] + V4 + V5\f$
 *
 * \param[in]		m 		size of the matrix
 * \param[in]		scale 	a list of numbers to scale the multplication by
 * \param[in]		A		matrix
 * \param[in]		V		a list of 5 pointers corresponding to V1, V2, V3, V4, V5
 * \param[out]		Av		a list of 3 pointers corresponding to Av1, Av2, Av3
 */
__device__
void matvec_n_by_m_scale_special (const int m,
	const double * __restrict__ scale,
	const double * __restrict__ A,
	double * const __restrict__ * V, double * __restrict__* Av);

/*!
 *  \brief Matrix-vector multiplication of a matrix sized NSPxM and a vector of size Mx1 scaled by a specified factor
 *
 *  Computes the following:
 *  	\f$Av1 = A * V1 * scale[0]\f$
 *  and:
 *  	\f$Av2 = A * V2 * scale[1]\f$
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
void matvec_n_by_m_scale_special2 (const int m,
	const double* __restrict__ scale,
	const double* __restrict__ A,
	double* const __restrict__ * V, double* __restrict__ * Av);

/*!
 * \brief Matrix-vector multiplication of a matrix sized NSPxM and a vector of size Mx1 scaled by a specified factor and added to another vector
 *
 * Computes \f$A * V * scale + add\f$
 *
 * \param[in]		m 		size of the matrix
 * \param[in]		scale 	a number to scale the multplication by
 * \param[in]		add 	the vector to add to the result
 * \param[in]		A		matrix
 * \param[in]		V		the vector
 * \param[out]		Av		vector that is A * V * scale + add
 */
__device__
void matvec_n_by_m_scale_add (const int m,
	const double scale,
	const double* __restrict__ add,
	const double* __restrict__ A,
	double* __restrict__ V, const double* __restrict__ Av);

///////////////////////////////////////////////////////////////////////////////

/*!
 *  \brief Matrix-vector multiplication of a matrix sized NSPxM and a vector of size Mx1 scaled by a specified factor and adds and subtracts the specified vectors
 * 		   note, the addition is twice the specified vector
 *
 *  Computes \f$scale * A * V + 2 * add - sub\f$
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
void matvec_n_by_m_scale_add_subtract (const int m,
	const double scale,
	const double* __restrict__ A,
	const double* V,
	double* __restrict__ Av,
	const double* __restrict__ add, const double* __restrict__ sub);

///////////////////////////////////////////////////////////////////////////////

/*!
 *  \brief Get scaling for weighted norm
 *
 *	Computes \f$\frac{1.0}{ATOL + \max\left(\left|y0\right|, \left|y1\right|) * RTOL\right)}\f$
 *
 * \param[in]		y0		values at current timestep
 * \param[in]		y1		values at next timestep
 * \param[out]		sc	array of scaling values
 */
__device__
void scale (const double* __restrict__ y0,
	const double* __restrict__ y1, double* __restrict__ sc);

///////////////////////////////////////////////////////////////////////////////

/*!
 * \brief Get scaling for weighted norm for the initial timestep (used in krylov process)
 *
 * \param[in]		y0		values at current timestep
 * \param[out]		sc	array of scaling values
 */
__device__
void scale_init (const double* __restrict__ y0, double* __restrict__ sc);

///////////////////////////////////////////////////////////////////////////////

/*!
 *  \brief Perform weighted norm
 *
 *  Computes \f$\left| nums * sc\right|_2\f$
 *
 * \param[in]		nums	values to be normed
 * \param[in]		sc		scaling array for norm
 * \return			norm	weighted norm
 */
__device__
double sc_norm(const double* __restrict__ nums, const double* __restrict__ sc);

/*!
 * \brief Computes and returns the two norm of a vector
 *
 *  Computes \f$\sqrt{\sum{v^2}}\f$
 *
 *	\param[in]		v 		the vector
 */
__device__
double two_norm(const double* __restrict__ v);

/*!
 *  \brief Normalize the input vector using a 2-norm
 *
 *  \f$v_{out} = \frac{v}{\left| v \right|}_2\f$
 *
 * \param[in]		v		vector to be normalized
 * \param[out]		v_out	where to stick the normalized part of v (in a column)
 */
__device__
double normalize (const double* __restrict__ v, double* __restrict__ v_out);

/*!
 *  \brief Performs the dot product of the w (NSP x 1) vector with the given subspace vector (NSP x 1)
 *
 *	returns \f$Vm \dot w\f$
 *
 * \param[in]		w   	the vector with with to dot
 * \param[in]		Vm		the subspace vector
 * \returns 		sum - the dot product of the specified vectors
 */
__device__
double dotproduct(const double* __restrict__ w, const double* __restrict__ Vm);

/*!
 * \brief Subtracts Vm scaled by s from w
 *
 *  \f$ w -= Vm * s\f$
 *
 * \param[in]		s   	the scale multiplier to use
 * \param[in]		Vm		the subspace matrix
 * \param[out]		w 		the vector to subtract from
 */
__device__
void scale_subtract(const double s, const double* __restrict__ Vm, double* __restrict__ w);

/*!
 *  \brief Sets Vm to s * w
 *
 *	\f$Vm = s * w\f$
 *
 * \param[in]		s   	the scale multiplier to use
 * \param[in]		w 		the vector to use as a base
 * \param[out]		Vm		the subspace matrix to set
 */
__device__
void scale_mult(const double s, const double* __restrict__ w, double* __restrict__ Vm);

#endif