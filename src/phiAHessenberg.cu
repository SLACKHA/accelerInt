#include <stdlib.h>
#include "header.h"
#include "solver_props.h"
//#include "linear-algebra.h"
#include "complexInverse.cuh"

#ifdef DOUBLE
extern __device__ __constant__ cuDoubleComplex poles[N_RA];
extern __device__ __constant__ cuDoubleComplex res[N_RA];
#else
extern __device__ __constant__ cuFloatComplex poles[N_RA];
extern __device__ __constant__ cuFloatComplex res[N_RA];
#endif


__device__
void phi2Ac_variable(const int m, const Real* A, const Real c, Real* phiA) {
	
#ifdef DOUBLE
	cuDoubleComplex invA[STRIDE * STRIDE];
#else
  	cuFloatComplex invA[STRIDE * STRIDE];
#endif
	
	#pragma unroll
	for (int i = 0; i < m; ++i) {
		#pragma unroll
		for (int j = 0; j < m; ++j) {
			phiA[i + j*STRIDE] = ZERO;
		}
	}

	#pragma unroll
	for (int q = 0; q < N_RA; q += 2) {
		
		// compute transpose and multiply with constant
		for (int i = 0; i < m; ++i) {
			for (int j = 0; j < m; ++j) {
				// A - theta * I
			#ifdef DOUBLE
				if (i == j) {
					invA[i + j*STRIDE] = cuCsub(make_cuDoubleComplex(c * A[i + j*STRIDE], 0.0), poles[q]);
				} else {
					invA[i + j*STRIDE] = make_cuDoubleComplex(c * A[i + j*STRIDE], 0.0);
				}
			#else
				if (i == j) {
					invA[i + j*STRIDE] = cuCsubf(make_cuFloatComplex(c * A[i + j*STRIDE], 0.0), poles[q]);
				} else {
					invA[i + j*STRIDE] = make_cuFloatComplex(c * A[i + j*STRIDE], 0.0);
				}
			#endif
			}
		}
		
		// takes care of (A * c - poles(q) * I)^-1
		//getInverseComplex (NN, invA);
		getComplexInverseHessenberg (m, invA);
		
		#pragma unroll
		for (int i = 0; i < m; ++i) {
			#pragma unroll
			for (int j = 0; j < m; ++j) {
			#ifdef DOUBLE
				phiA[i + j*STRIDE] += 2.0 * cuCreal( cuCmul( cuCdiv(res[q], cuCmul(poles[q], poles[q])), invA[i + j*STRIDE]) );
			#else
				phiA[i + j*STRIDE] += 2.0 * cuCrealf( cuCmulf( cuCdivf(res[q], cuCmulf(poles[q], poles[q])), invA[i + j*STRIDE]) );
			#endif
			}
		}
	}
}

__device__
void phiAc_variable(const int m, const Real* A, const Real c, Real* phiA) {
	
#ifdef DOUBLE
	cuDoubleComplex invA[STRIDE * STRIDE];
#else
  	cuFloatComplex invA[STRIDE * STRIDE];
#endif
	
	#pragma unroll
	for (int i = 0; i < m; ++i) {
		#pragma unroll
		for (int j = 0; j < m; ++j) {
			phiA[i + j*STRIDE] = ZERO;
		}
	}

	#pragma unroll
	for (int q = 0; q < N_RA; q += 2) {
		
		// compute transpose and multiply with constant
		for (int i = 0; i < m; ++i) {
			for (int j = 0; j < m; ++j) {
				// A - theta * I
			#ifdef DOUBLE
				if (i == j) {
					invA[i + j*STRIDE] = cuCsub(make_cuDoubleComplex(c * A[i + j*STRIDE], 0.0), poles[q]);
				} else {
					invA[i + j*STRIDE] = make_cuDoubleComplex(c * A[i + j*STRIDE], 0.0);
				}
			#else
				if (i == j) {
					invA[i + j*STRIDE] = cuCsubf(make_cuFloatComplex(c * A[i + j*STRIDE], 0.0), poles[q]);
				} else {
					invA[i + j*STRIDE] = make_cuFloatComplex(c * A[i + j*STRIDE], 0.0);
				}
			#endif
			}
		}
		
		// takes care of (A * c - poles(q) * I)^-1
		//getInverseComplex (NN, invA);
		getComplexInverseHessenberg (m, invA);
		
		#pragma unroll
		for (int i = 0; i < m; ++i) {
			#pragma unroll
			for (int j = 0; j < m; ++j) {
			#ifdef DOUBLE
				phiA[i + j*STRIDE] += 2.0 * cuCreal( cuCmul( cuCdiv(res[q], poles[q]), invA[i + j*STRIDE]) );
			#else
				phiA[i + j*STRIDE] += 2.0 * cuCrealf( cuCmulf( cuCdivf(res[q], poles[q]), invA[i + j*STRIDE]) );
			#endif
			}
		}
	}
}

__device__
void expAc_variable(const int m, const Real* A, const Real c, Real* phiA) {
	
#ifdef DOUBLE
	cuDoubleComplex invA[STRIDE * STRIDE];
#else
  	cuFloatComplex invA[STRIDE * STRIDE];
#endif
	
	#pragma unroll
	for (int i = 0; i < m; ++i) {
		#pragma unroll
		for (int j = 0; j < m; ++j) {
			phiA[i + j*STRIDE] = ZERO;
		}
	}

	#pragma unroll
	for (int q = 0; q < N_RA; q += 2) {
		
		// compute transpose and multiply with constant
		for (int i = 0; i < m; ++i) {
			for (int j = 0; j < m; ++j) {
				// A - theta * I
			#ifdef DOUBLE
				if (i == j) {
					invA[i + j*STRIDE] = cuCsub(make_cuDoubleComplex(c * A[i + j*STRIDE], 0.0), poles[q]);
				} else {
					invA[i + j*STRIDE] = make_cuDoubleComplex(c * A[i + j*STRIDE], 0.0);
				}
			#else
				if (i == j) {
					invA[i + j*STRIDE] = cuCsubf(make_cuFloatComplex(c * A[i + j*STRIDE], 0.0), poles[q]);
				} else {
					invA[i + j*STRIDE] = make_cuFloatComplex(c * A[i + j*STRIDE], 0.0);
				}
			#endif
			}
		}
		
		// takes care of (A * c - poles(q) * I)^-1
		//getInverseComplex (NN, invA);
		getComplexInverseHessenberg (m, invA);
		
		#pragma unroll
		for (int i = 0; i < m; ++i) {
			#pragma unroll
			for (int j = 0; j < m; ++j) {
			#ifdef DOUBLE
				phiA[i + j*STRIDE] += 2.0 * cuCreal( cuCmul( res[q], invA[i + j*STRIDE]) );
			#else
				phiA[i + j*STRIDE] += 2.0 * cuCrealf( cuCmulf( res[q], invA[i + j*STRIDE]) )
			#endif
			}
		}
	}
}