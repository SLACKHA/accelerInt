#include <stdlib.h>
#include "header.cuh"
#include "solver_options.cuh"
#include "solver_props.cuh"
//#include "linear-algebra.h"
#include "complexInverse.cuh"

extern __device__ __constant__ cuDoubleComplex poles[N_RA];
extern __device__ __constant__ cuDoubleComplex res[N_RA];

__device__
int phi2Ac_variable(const int m, const double* __restrict__ A, const double c,
						double* __restrict__ phiA, const solver_memory* __restrict__ solver,
						cuDoubleComplex* __restrict__ work) {
	
	cuDoubleComplex * const __restrict__ invA = solver->invA;
	int * const __restrict__ ipiv = solver->ipiv;
	int info = 0;
	
	#pragma unroll
	for (int i = 0; i < m; ++i) {
		#pragma unroll
		for (int j = 0; j < m; ++j) {
			phiA[INDEX(i + j*STRIDE)] = 0.0;
		}
	}

	#pragma unroll
	for (int q = 0; q < N_RA; q += 2) {
		
		// compute transpose and multiply with constant
		for (int i = 0; i < m; ++i) {
			for (int j = 0; j < m; ++j) {
				// A - theta * I
				if (i == j) {
					invA[INDEX(i + j*STRIDE)] = cuCsub(make_cuDoubleComplex(c * A[INDEX(i + j*STRIDE)], 0.0), poles[q]);
				} else {
					invA[INDEX(i + j*STRIDE)] = make_cuDoubleComplex(c * A[INDEX(i + j*STRIDE)], 0.0);
				}
			}
		}
		
		// takes care of (A * c - poles(q) * I)^-1
		getComplexInverseHessenberg (m, invA, ipiv, &info, work);

		if (info != 0)
			return info;
		
		#pragma unroll
		for (int i = 0; i < m; ++i) {
			#pragma unroll
			for (int j = 0; j < m; ++j) {
				phiA[INDEX(i + j*STRIDE)] += 2.0 * cuCreal( cuCmul( cuCdiv(res[q], cuCmul(poles[q], poles[q])), invA[INDEX(i + j*STRIDE)]) );
			}
		}
	}
	return 0;
}

__device__
int phiAc_variable(const int m, const double* __restrict__ A, const double c,
						double* __restrict__ phiA, const solver_memory* __restrict__ solver,
						cuDoubleComplex* __restrict__ work) {
	
	cuDoubleComplex * const __restrict__ invA = solver->invA;
	int * const __restrict__ ipiv = solver->ipiv;
	int info = 0;

	#pragma unroll
	for (int i = 0; i < m; ++i) {
		#pragma unroll
		for (int j = 0; j < m; ++j) {
			phiA[INDEX(i + j*STRIDE)] = 0.0;
		}
	}

	#pragma unroll
	for (int q = 0; q < N_RA; q += 2) {
		
		// compute transpose and multiply with constant
		for (int i = 0; i < m; ++i) {
			for (int j = 0; j < m; ++j) {
				// A - theta * I
				if (i == j) {
					invA[INDEX(i + j*STRIDE)] = cuCsub(make_cuDoubleComplex(c * A[INDEX(i + j*STRIDE)], 0.0), poles[q]);
				} else {
					invA[INDEX(i + j*STRIDE)] = make_cuDoubleComplex(c * A[INDEX(i + j*STRIDE)], 0.0);
				}
			}
		}
		
		// takes care of (A * c - poles(q) * I)^-1
		getComplexInverseHessenberg (m, invA, ipiv, &info, work);

		if (info != 0)
			return info;
		
		#pragma unroll
		for (int i = 0; i < m; ++i) {
			#pragma unroll
			for (int j = 0; j < m; ++j) {
				phiA[INDEX(i + j*STRIDE)] += 2.0 * cuCreal( cuCmul( cuCdiv(res[q], poles[q]), invA[INDEX(i + j*STRIDE)]) );
			}
		}
	}
	return 0;
}

__device__
int expAc_variable(const int m, const double* __restrict__ A, const double c,
						double* __restrict__ phiA, const solver_memory* __restrict__ solver,
						cuDoubleComplex* __restrict__ work) {
	
	cuDoubleComplex * const __restrict__ invA = solver->invA;
	int * const __restrict__ ipiv = solver->ipiv;
	int info = 0;
	
	#pragma unroll
	for (int i = 0; i < m; ++i) {
		#pragma unroll
		for (int j = 0; j < m; ++j) {
			phiA[INDEX(i + j*STRIDE)] = 0.0;
		}
	}

	#pragma unroll
	for (int q = 0; q < N_RA; q += 2) {
		
		// compute transpose and multiply with constant
		for (int i = 0; i < m; ++i) {
			for (int j = 0; j < m; ++j) {
				// A - theta * I
				if (i == j) {
					invA[INDEX(i + j*STRIDE)] = cuCsub(make_cuDoubleComplex(c * A[INDEX(i + j*STRIDE)], 0.0), poles[q]);
				} else {
					invA[INDEX(i + j*STRIDE)] = make_cuDoubleComplex(c * A[INDEX(i + j*STRIDE)], 0.0);
				}
			}
		}
		
		// takes care of (A * c - poles(q) * I)^-1
		getComplexInverseHessenberg (m, invA, ipiv, &info, work);

		if (info != 0)
			return info;
		
		
		#pragma unroll
		for (int i = 0; i < m; ++i) {
			#pragma unroll
			for (int j = 0; j < m; ++j) {
				phiA[INDEX(i + j*STRIDE)] += 2.0 * cuCreal( cuCmul( res[q], invA[INDEX(i + j*STRIDE)]) );
			}
		}
	}
	return 0;
}