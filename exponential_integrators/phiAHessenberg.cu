#include <stdlib.h>
#include "header.cuh"
#include "solver_options.h"
#include "solver_props.cuh"
//#include "linear-algebra.h"
#include "complexInverse.cuh"

extern __device__ __constant__ cuDoubleComplex poles[N_RA];
extern __device__ __constant__ cuDoubleComplex res[N_RA];

__device__
void phi2Ac_variable(const int m, const double* A, const double c, double* phiA) {
	
	cuDoubleComplex invA[STRIDE * STRIDE];
	
	#pragma unroll
	for (int i = 0; i < m; ++i) {
		#pragma unroll
		for (int j = 0; j < m; ++j) {
			phiA[i + j*STRIDE] = 0.0;
		}
	}

	#pragma unroll
	for (int q = 0; q < N_RA; q += 2) {
		
		// compute transpose and multiply with constant
		for (int i = 0; i < m; ++i) {
			for (int j = 0; j < m; ++j) {
				// A - theta * I
				if (i == j) {
					invA[i + j*STRIDE] = cuCsub(make_cuDoubleComplex(c * A[i + j*STRIDE], 0.0), poles[q]);
				} else {
					invA[i + j*STRIDE] = make_cuDoubleComplex(c * A[i + j*STRIDE], 0.0);
				}
			}
		}
		
		// takes care of (A * c - poles(q) * I)^-1
		//getInverseComplex (NN, invA);
		getComplexInverseHessenberg (m, invA);
		
		#pragma unroll
		for (int i = 0; i < m; ++i) {
			#pragma unroll
			for (int j = 0; j < m; ++j) {
				phiA[i + j*STRIDE] += 2.0 * cuCreal( cuCmul( cuCdiv(res[q], cuCmul(poles[q], poles[q])), invA[i + j*STRIDE]) );
			}
		}
	}
}

__device__
void phiAc_variable(const int m, const double* A, const double c, double* phiA) {

	cuDoubleComplex invA[STRIDE * STRIDE];
	
	#pragma unroll
	for (int i = 0; i < m; ++i) {
		#pragma unroll
		for (int j = 0; j < m; ++j) {
			phiA[i + j*STRIDE] = 0.0;
		}
	}

	#pragma unroll
	for (int q = 0; q < N_RA; q += 2) {
		
		// compute transpose and multiply with constant
		for (int i = 0; i < m; ++i) {
			for (int j = 0; j < m; ++j) {
				// A - theta * I
				if (i == j) {
					invA[i + j*STRIDE] = cuCsub(make_cuDoubleComplex(c * A[i + j*STRIDE], 0.0), poles[q]);
				} else {
					invA[i + j*STRIDE] = make_cuDoubleComplex(c * A[i + j*STRIDE], 0.0);
				}
			}
		}
		
		// takes care of (A * c - poles(q) * I)^-1
		//getInverseComplex (NN, invA);
		getComplexInverseHessenberg (m, invA);
		
		#pragma unroll
		for (int i = 0; i < m; ++i) {
			#pragma unroll
			for (int j = 0; j < m; ++j) {
				phiA[i + j*STRIDE] += 2.0 * cuCreal( cuCmul( cuCdiv(res[q], poles[q]), invA[i + j*STRIDE]) );
			}
		}
	}
}

__device__
void expAc_variable(const int m, const double* A, const double c, double* phiA) {

	cuDoubleComplex invA[STRIDE * STRIDE];
	
	#pragma unroll
	for (int i = 0; i < m; ++i) {
		#pragma unroll
		for (int j = 0; j < m; ++j) {
			phiA[i + j*STRIDE] = 0.0;
		}
	}

	#pragma unroll
	for (int q = 0; q < N_RA; q += 2) {
		
		// compute transpose and multiply with constant
		for (int i = 0; i < m; ++i) {
			for (int j = 0; j < m; ++j) {
				// A - theta * I
				if (i == j) {
					invA[i + j*STRIDE] = cuCsub(make_cuDoubleComplex(c * A[i + j*STRIDE], 0.0), poles[q]);
				} else {
					invA[i + j*STRIDE] = make_cuDoubleComplex(c * A[i + j*STRIDE], 0.0);
				}
			}
		}
		
		// takes care of (A * c - poles(q) * I)^-1
		//getInverseComplex (NN, invA);
		getComplexInverseHessenberg (m, invA);
		
		#pragma unroll
		for (int i = 0; i < m; ++i) {
			#pragma unroll
			for (int j = 0; j < m; ++j) {
				phiA[i + j*STRIDE] += 2.0 * cuCreal( cuCmul( res[q], invA[i + j*STRIDE]) );
			}
		}
	}
}