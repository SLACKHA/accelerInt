/*!
 * \file exponential_linear_algebra.cu
 * \brief Implementation of various linear algebra functions needed in the exponential integrators
 *
 * \author Nicholas Curtis
 * \date 03/09/2015
 *
 */

#include "exponential_linear_algebra.cuh"

///////////////////////////////////////////////////////////////////////////////

__device__
void matvec_m_by_m (const int m, const double * const __restrict__ A,
						const double * const __restrict__ V,
						double * const __restrict__ Av) {
	//for each row
	for (int i = 0; i < m; ++i) {
		Av[INDEX(i)] = A[INDEX(i)] * V[INDEX(0)];

		//go across a row of A, multiplying by a column of phiHm
		for (int j = 1; j < m; ++j) {
			Av[INDEX(i)] += A[INDEX(j * STRIDE + i)] * V[INDEX(j)];
		}
	}
}

///////////////////////////////////////////////////////////////////////////////

__device__ void matvec_m_by_m_plusequal (const int m, const double * const __restrict__ A,
										 const double * const __restrict__ V, double * const __restrict__ Av)
{
	//for each row
	for (int i = 0; i < m; ++i) {
		Av[INDEX(i)] = A[INDEX(i)] * V[INDEX(0)];

		//go across a row of A, multiplying by a column of phiHm
		for (int j = 1; j < m; ++j) {
			Av[INDEX(i)] += A[INDEX(j * STRIDE + i)] * V[INDEX(j)];
		}

		Av[INDEX(i)] += V[INDEX(i)];
	}
}

__device__
void matvec_n_by_m_scale (const int m, const double scale,
						  const double * const __restrict__ A,
						  const double * const __restrict__ V,
						  double * const __restrict__ Av) {
	//for each row
	#pragma unroll
	for (int i = 0; i < NSP; ++i) {
		Av[INDEX(i)] = A[INDEX(i)] * V[INDEX(0)];

		//go across a row of A, multiplying by a column of phiHm
		for (int j = 1; j < m; ++j) {
			Av[INDEX(i)] += A[INDEX(j * NSP + i)] * V[INDEX(j)];
		}

		Av[INDEX(i)] *= scale;
	}
}

__device__
void matvec_n_by_m_scale_special (const int m, const double * __restrict__ scale,
								  const double * __restrict__ A,
								  double * const __restrict__ * V,
								  double * __restrict__ * Av) {
	//for each row
	#pragma unroll
	for (int i = 0; i < NSP; ++i) {
		Av[0][INDEX(i)] = A[INDEX(i)] * V[0][INDEX(0)];
		Av[1][INDEX(i)] = A[INDEX(i)] * V[1][INDEX(0)];
		Av[2][INDEX(i)] = A[INDEX(i)] * V[2][INDEX(0)];

		//go across a row of A, multiplying by a column of phiHm
		for (int j = 1; j < m; ++j) {
			Av[0][INDEX(i)] += A[INDEX(j * NSP + i)] * V[0][INDEX(j)];
			Av[1][INDEX(i)] += A[INDEX(j * NSP + i)] * V[1][INDEX(j)];
			Av[2][INDEX(i)] += A[INDEX(j * NSP + i)] * V[2][INDEX(j)];
		}

		Av[0][INDEX(i)] *= scale[0];
		Av[1][INDEX(i)] *= scale[1];
		Av[2][INDEX(i)]  = scale[2] * Av[2][INDEX(i)] + V[3][INDEX(i)] + V[4][INDEX(i)];
	}
}

__device__
void matvec_n_by_m_scale_special2 (const int m, const double* __restrict__ scale, const double* __restrict__ A,
										double* const __restrict__ * V, double* __restrict__ * Av) {
	//for each row
	#pragma unroll
	for (int i = 0; i < NSP; ++i) {
		Av[0][INDEX(i)] = A[INDEX(i)] * V[0][INDEX(0)];
		Av[1][INDEX(i)] = A[INDEX(i)] * V[1][INDEX(0)];

		//go across a row of A, multiplying by a column of phiHm
		for (int j = 1; j < m; ++j) {
			Av[0][INDEX(i)] += A[INDEX(j * NSP + i)] * V[0][INDEX(j)];
			Av[1][INDEX(i)] += A[INDEX(j * NSP + i)] * V[1][INDEX(j)];
		}

		Av[0][INDEX(i)] *= scale[0];
		Av[1][INDEX(i)] *= scale[1];
	}
}

///////////////////////////////////////////////////////////////////////////////

__device__
void matvec_n_by_m_scale_add (const int m, const double scale,
								const double* __restrict__ A, const double* __restrict__ V,
								double* __restrict__ Av, const double* __restrict__ add) {
	//for each row
	#pragma unroll
	for (int i = 0; i < NSP; ++i) {
		Av[INDEX(i)] = A[INDEX(i)] * V[INDEX(0)];

		//go across a row of A, multiplying by a column of phiHm
		for (int j = 1; j < m; ++j) {
			Av[INDEX(i)] += A[INDEX(j * NSP + i)] * V[INDEX(j)];
		}

		Av[INDEX(i)] = Av[INDEX(i)] * scale + add[INDEX(i)];
	}
}

///////////////////////////////////////////////////////////////////////////////

__device__
void matvec_n_by_m_scale_add_subtract (const int m, const double scale,
										const double* __restrict__ A, const double* V,
										double* __restrict__ Av, const double* __restrict__ add,
										const double* __restrict__ sub) {
	//for each row
	#pragma unroll
	for (int i = 0; i < NSP; ++i) {
		Av[INDEX(i)] = A[INDEX(i)] * V[INDEX(0)];

		//go across a row of A, multiplying by a column of phiHm
		#pragma unroll
		for (int j = 1; j < m; ++j) {
			Av[INDEX(i)] += A[INDEX(j * NSP + i)] * V[INDEX(j)];
		}

		Av[INDEX(i)] = Av[INDEX(i)] * scale + 2.0 * add[INDEX(i)] - sub[INDEX(i)];
	}
}

///////////////////////////////////////////////////////////////////////////////

__device__
void scale (const double* __restrict__ y0, const double* __restrict__ y1, double* __restrict__ sc) {
	#pragma unroll
	for (int i = 0; i < NSP; ++i) {
		sc[INDEX(i)] = 1.0 / (ATOL + fmax(fabs(y0[INDEX(i)]), fabs(y1[INDEX(i)])) * RTOL);
	}
}

///////////////////////////////////////////////////////////////////////////////

__device__
void scale_init (const double* __restrict__ y0, double* __restrict__ sc) {
	#pragma unroll
	for (int i = 0; i < NSP; ++i) {
		sc[INDEX(i)] = 1.0 / (ATOL + fabs(y0[INDEX(i)]) * RTOL);
	}
}

///////////////////////////////////////////////////////////////////////////////

__device__
double sc_norm (const double* __restrict__ nums, const double* __restrict__ sc) {
	double norm = 0.0;

	#pragma unroll
	for (int i = 0; i < NSP; ++i) {
		norm += nums[INDEX(i)] * nums[INDEX(i)] * (sc[INDEX(i)] * sc[INDEX(i)]);
	}

	return sqrt(norm / ((double)NSP));
}

__device__
double two_norm(const double* __restrict__ v)
{
	double norm = 0.0;
	#pragma unroll
	for (int i = 0; i < NSP; ++i) {
		norm += v[INDEX(i)] * v[INDEX(i)];
	}
	return sqrt(norm);
}

__device__
double normalize (const double* __restrict__ v, double* __restrict__ v_out) {

	double norm = two_norm(v);

	//unlikely to happen, if so, we still need to copy
	if (norm == 0.0)
		norm = 1.0;

	double m_norm = 1.0 / norm;

	#pragma unroll
	for (int i = 0; i < NSP; ++i) {
		v_out[INDEX(i)] = v[INDEX(i)] * m_norm;
	}
	return norm;
}

__device__
double dotproduct(const double* __restrict__ w, const double* __restrict__ Vm)
{
	double sum = 0;
	#pragma unroll
	for(int i = 0; i < NSP; i++)
	{
		sum += w[INDEX(i)] * Vm[INDEX(i)];
	}
	return sum;
}

__device__ void scale_subtract(const double s, const double* __restrict__ Vm, double* __restrict__ w)
{
	#pragma unroll
	for (int i = 0; i < NSP; i++)
	{
		w[INDEX(i)] -= s * Vm[INDEX(i)];
	}
}

__device__ void scale_mult(const double s, const double* __restrict__ w, double* __restrict__ Vm)
{
	#pragma unroll
	for (int i = 0; i < NSP; i++)
	{
		Vm[INDEX(i)] = w[INDEX(i)] * s;
	}
}