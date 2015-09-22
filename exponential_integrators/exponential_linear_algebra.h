/* exponential_linear_algebra.c
 * Implementation of various linear algebra functions needed in the exponential integrators
 * \file exponential_linear_algebra.c
 *
 * \author Nicholas Curtis
 * \date 03/09/2015
 *
 */

#ifndef EXPONENTIAL_LINEAR_ALGEBRA_H
#define EXPONENTIAL_LINEAR_ALGEBRA_H

#include "header.h"
#include "solver_options.h"
#include "solver_props.h"

/** Computes and returns the two norm of a vector
 *
 *	\param[in]		v 		the vector
 */
static inline
double two_norm(const double* v)
{
	double norm = 0.0;
	#pragma unroll
	for (int i = 0; i < NN; ++i) {
		norm += v[i] * v[i];
	}
	return sqrt(norm);
}

/** Normalize the input vector using a 2-norm
 * 
 * \param[in]		v		vector to be normalized
 * \param[out]		v_out	where to stick the normalized part of v (in a column)
 */
static inline
double normalize (const double * v, double* v_out) {
	
	double norm = two_norm(v);

	if (norm == 0)
		norm = 1;

	#pragma unroll
	for (int i = 0; i < NN; ++i) {
		v_out[i] = v[i] / norm;
	}
	return norm;
}

/** Performs the dot product of the w vector with the given Matrix
 * 
 * \param[in]		w   	the vector with with to dot
 * \param[in]		Vm		the subspace matrix
 * \out						the dot product of the specified vectors
 */
static inline
double dotproduct(const double* w, const double* Vm)
{
	double sum = 0;
	#pragma unroll
	for(int i = 0; i < NN; i++)
	{
		sum += w[i] * Vm[i];
	}
	return sum;
}

/** Sets column c of Vm to s * w
 * 
 * \param[in]		c 		the column of matrix Vm to use
 * \param[in]		stride 	number of columns in Vm
 * \param[in]		s   	the scale multiplier to use
 * \param[in]		w 		the vector to use as a base
 * \param[out]		Vm		the subspace matrix to set
 */
static inline void scale_mult(const double s, const double* w, double* Vm)
{
	#pragma unroll
	for (int i = 0; i < NN; i++)
	{
		Vm[i] = w[i] * s;
	}
}

///////////////////////////////////////////////////////////////////////////////

/** Perform weighted norm
 * 
 * \param[in]		nums	values to be normed
 * \param[in]		sc		scaling array for norm
 * \return			norm	weighted norm
 */
static inline
double sc_norm (const double * nums, const double * sc) {
	double norm = 0.0;
	
	#pragma unroll
	for (int i = 0; i < NN; ++i) {
		norm += nums[i] * nums[i] / (sc[i] * sc[i]);
	}
	
	norm = sqrt(norm / NN);
	
	return norm;
}

/** Subtracts column c of Vm scaled by s from w
 * 
 * \param[in]		s   	the scale multiplier to use
 * \param[in]		Vm		the subspace matrix
 * \param[out]		w 		the vector to subtract from
 */
static inline void scale_subtract(const double s, const double* Vm, double* w)
{
	#pragma unroll
	for (int i = 0; i < NN; i++)
	{
		w[i] -= s * Vm[i];
	}
}

///////////////////////////////////////////////////////////////////////////////

/** Matrix-vector multiplication of a matrix sized MxM and a vector Mx1
 * 
 * Performs  matrix-vector multiplication (with unrolled loops) 
 * 
 * \param[in]		m 		size of the matrix
 * \param[in]		A		matrix of size MxM
 * \param[in]		V		vector of size Mx1
 * \param[out]		Av		vector that is A * v
 */
static inline
void matvec_m_by_m (const int m, const double * A, const double * V, double * Av) {
	//for each row
	#pragma unroll
	for (int i = 0; i < m; ++i) {
		Av[i] = 0.0;
		
		//go across a row of A, multiplying by a column of phiHm
		#pragma unroll
		for (int j = 0; j < m; ++j) {
			Av[i] += A[j * STRIDE + i] * V[j];
		}
	}
}

///////////////////////////////////////////////////////////////////////////////

/** Matrix-vector plus equals for a matrix of size MxM and vector of size Mx1
 * 
 *  That is, it returns (A + I) * v
 *
 * Performs  matrix-vector multiplication (with unrolled loops) 
 * 
 * \param[in]		m 		size of the matrix
 * \param[in]		A		matrix of size MxM
 * \param[in]		V		vector of size Mx1
 * \param[out]		Av		vector that is (A + I) * v
 */
static inline void matvec_m_by_m_plusequal (const int m, const double * A, const double * V, double * Av)
{
	//for each row
	#pragma unroll
	for (int i = 0; i < m; ++i) {
		Av[i] = 0.0;
		
		//go across a row of A, multiplying by a column of phiHm
		#pragma unroll
		for (int j = 0; j < m; ++j) {
			Av[i] += A[j * STRIDE + i] * V[j];
		}

		Av[i] += V[i];
	}
}

/** Matrix-vector multiplication of a matrix sized NNxM and a vector of size Mx1 scaled by a specified factor
 * 
 * Performs  matrix-vector multiplication (with unrolled loops)
 * 
 * \param[in]		m 		size of the matrix
 * \param[in]		scale 	a number to scale the multplication by
 * \param[in]		A		matrix
 * \param[in]		V		the vector
 * \param[out]		Av		vector that is A * V
 */
static inline
void matvec_n_by_m_scale (const int m, const double scale, const double * A, const double * V, double * Av) {
	//for each row
	#pragma unroll
	for (int i = 0; i < NN; ++i) {
		Av[i] = 0.0;
		
		//go across a row of A, multiplying by a column of phiHm
		#pragma unroll
		for (int j = 0; j < m; ++j) {
			Av[i] += A[j * NN + i] * V[j];
		}

		Av[i] *= scale;
	}
}

///////////////////////////////////////////////////////////////////////////////

/** Matrix-vector multiplication of a matrix sized NNxM and a vector of size Mx1 scaled by a specified factor and added to another vector
 * 
 * Performs  matrix-vector multiplication (with unrolled loops)
 * 
 * \param[in]		m 		size of the matrix
 * \param[in]		scale 	a number to scale the multplication by
 * \param[in]		add 	the vector to add to the result
 * \param[in]		A		matrix
 * \param[in]		V		the vector
 * \param[out]		Av		vector that is A * V
 */
static inline
void matvec_n_by_m_scale_add (const int m, const double scale, const double * A, const double * V, double * Av, const double* add) {
	//for each row
	#pragma unroll
	for (int i = 0; i < NN; ++i) {
		Av[i] = 0.0;
		
		//go across a row of A, multiplying by a column of phiHm
		#pragma unroll
		for (int j = 0; j < m; ++j) {
			Av[i] += A[j * NN + i] * V[j];
		}

		Av[i] = Av[i] * scale + add[i];
	}
}

///////////////////////////////////////////////////////////////////////////////

/** Matrix-vector multiplication of a matrix sized NNxM and a vector of size Mx1 scaled by a specified factor and adds and subtracts the specified vectors
 *  note, the addition is twice the specified vector
 * 
 * Performs  matrix-vector multiplication (with unrolled loops)
 * 
 * \param[in]		m 		size of the matrix
 * \param[in]		scale 	a number to scale the multplication by
 * \param[in]		add 	the vector to add to the result
 * \param[]
 * \param[in]		A		matrix
 * \param[in]		V		the vector
 * \param[out]		Av		vector that is A * V
 */
static inline
void matvec_n_by_m_scale_add_subtract (const int m, const double scale, const double * A, const double * V, double * Av, const double* add, const double * sub) {
	//for each row
	#pragma unroll
	for (int i = 0; i < NN; ++i) {
		Av[i] = 0.0;
		
		//go across a row of A, multiplying by a column of phiHm
		#pragma unroll
		for (int j = 0; j < m; ++j) {
			Av[i] += A[j * NN + i] * V[j];
		}

		Av[i] = Av[i] * scale + 2.0 * add[i] - sub[i];
	}
}

/** Matrix-vector multiplication of a matrix sized NNxM and a vector of size Mx1 scaled by a specified factor
 *
 *  Computes the following:
 *  Av1 = A * V1 * scale[0]
 *  Av2 = A * V2 * scale[1]
 *  Av3 = A * V3 * scale[2] + V4 + V5
 * 
 * Performs  matrix-vector multiplication (with unrolled loops)
 * 
 * \param[in]		m 		size of the matrix
 * \param[in]		scale 	a list of numbers to scale the multplication by
 * \param[in]		A		matrix
 * \param[in]		V		a list of 5 pointers corresponding to V1, V2, V3, V4, V5
 * \param[out]		Av		a list of 3 pointers corresponding to Av1, Av2, Av3
 */
static inline
void matvec_n_by_m_scale_special (const int m, const double scale[], const double * A, const double* V[], double* Av[]) {
	//for each row
	#pragma unroll
	for (int i = 0; i < NN; ++i) {
		#pragma unroll
		for (int k = 0; k < 3; k++)
		{
			Av[k][i] = 0.0;
		}
		
		//go across a row of A, multiplying by a column of phiHm
		#pragma unroll
		for (int j = 0; j < m; ++j) {
			#pragma unroll
			for (int k = 0; k < 3; k++)
			{
				Av[k][i] += A[j * NN + i] * V[k][j];
			}
		}

		#pragma unroll
		for (int k = 0; k < 3; k++)
		{
			Av[k][i] *= scale[k];
		}
		Av[2][i] += V[3][i];
		Av[2][i] += V[4][i];
	}
}

/** Matrix-vector multiplication of a matrix sized NNxM and a vector of size Mx1 scaled by a specified factor
 *
 *  Computes the following:
 *  Av1 = A * V1 * scale[0]
 *  Av2 = A * V2 * scale[1]
 * 
 * Performs  matrix-vector multiplication (with unrolled loops)
 * 
 * \param[in]		m 		size of the matrix
 * \param[in]		scale 	a list of numbers to scale the multplication by
 * \param[in]		A		matrix
 * \param[in]		V		a list of 2 pointers corresponding to V1, V2
 * \param[out]		Av		a list of 2 pointers corresponding to Av1, Av2
 */
static inline
void matvec_n_by_m_scale_special2 (const int m, const double scale[], const double * A, const double* V[], double* Av[]) {
	//for each row
	#pragma unroll
	for (int i = 0; i < NN; ++i) {
		#pragma unroll
		for (int k = 0; k < 2; k++)
		{
			Av[k][i] = 0.0;
		}
		
		//go across a row of A, multiplying by a column of phiHm
		#pragma unroll
		for (int j = 0; j < m; ++j) {
			#pragma unroll
			for (int k = 0; k < 2; k++)
			{
				Av[k][i] += A[j * NN + i] * V[k][j];
			}
		}

		#pragma unroll
		for (int k = 0; k < 2; k++)
		{
			Av[k][i] *= scale[k];
		}
	}
}

///////////////////////////////////////////////////////////////////////////////

/** Get scaling for weighted norm
 * 
 * \param[in]		y0		values at current timestep
 * \param[in]		y1		values at next timestep
 * \param[out]	sc	array of scaling values
 */
static inline
void scale (const double * y0, const double * y1, double * sc) {
	#pragma unroll
	for (int i = 0; i < NN; ++i) {
		sc[i] = ATOL + fmax(fabs(y0[i]), fabs(y1[i])) * RTOL;
	}
}

///////////////////////////////////////////////////////////////////////////////

/** Get scaling for weighted norm for the initial timestep (used in krylov process)
 * 
 * \param[in]		y0		values at current timestep
 * \param[out]	sc	array of scaling values
 */
static inline
void scale_init (const double * y0, double * sc) {
	#pragma unroll
	for (int i = 0; i < NN; ++i) {
		sc[i] = ATOL + fabs(y0[i]) * RTOL;
	}
}

#endif