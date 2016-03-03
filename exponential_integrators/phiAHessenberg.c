#include <stdlib.h>
#include <complex.h>

#include "header.h"
#include "lapack_dfns.h"
#include "complexInverse.h"
#include "solver_options.h"

extern double complex poles[N_RA];
extern double complex res[N_RA];

int get_work_size(const int m) {
	int work_size = 0;
	int work_flag = -1;
	zgetri_(&m, 0, 0, 0, &work_size, &work_flag, 0);
	return work_size;
}

void phi2Ac_variable(const int m, const int STRIDE, const double* A, const double c, double* phiA) {
	
	//query work size for inverse
	int work_size = get_work_size(m);

	//allocate arrays
	double complex work* = (double complex*)malloc(work_size * sizeof(double complex));
	int ipiv[STRIDE] = {0};
	double complex invA[STRIDE * STRIDE];
	int info = 0;

	#pragma unroll
	for (int i = 0; i < m; ++i) {
		#pragma unroll
		for (int j = 0; j < m; ++j) {
			phiA[i + j*STRIDE] = 0.0;
		}
	}

	#pragma unroll
	for (int q = 0; q < N_RA; q += 2) {
		
		// init invA
		for (int i = 0; i < m; ++i) {
			for (int j = 0; j < m; ++j) {
				// A - theta * I
				if (i == j) {
					invA[i + j * STRIDE] = c * A[i + j*STRIDE] - poles[q];
				} else {
					invA[i + j * STRIDE] = c * A[i + j*STRIDE];
				}
			}
		}
		
		// takes care of (A * c - poles(q) * I)^-1
		getComplexInverseHessenberg (m, invA, ipiv, &info, work, work_size);
		
		#pragma unroll
		for (int i = 0; i < m; ++i) {
			#pragma unroll
			for (int j = 0; j < m; ++j) {
				phiA[i + j*STRIDE] += 2.0 * creal((res[q] / (poles[q] * poles[q])) * invA[i + j * STRIDE]);
			}
		}
	}
	
	free(work);
}

void phiAc_variable(const int m, const int STRIDE, const double* A, const double c, double* phiA) {
	
	//query work size for inverse
	int work_size = get_work_size(m);

	//allocate arrays
	double complex work* = (double complex*)malloc(work_size * sizeof(double complex));
	int ipiv[STRIDE] = {0};
	double complex invA[STRIDE * STRIDE];
	int info = 0;
	
	#pragma unroll
	for (int i = 0; i < m; ++i) {
		#pragma unroll
		for (int j = 0; j < m; ++j) {
			phiA[i + j*STRIDE] = 0.0;
		}
	}

	#pragma unroll
	for (int q = 0; q < N_RA; q += 2) {
		
		// init invA
		for (int i = 0; i < m; ++i) {
			for (int j = 0; j < m; ++j) {
				// A - theta * I
				if (i == j) {
					invA[i + j * STRIDE] = c * A[i + j*STRIDE] - poles[q];
				} else {
					invA[i + j * STRIDE] = c * A[i + j*STRIDE];
				}
			}
		}
		
		// takes care of (A * c - poles(q) * I)^-1
		getComplexInverseHessenberg (m, invA, ipiv, &info, work, work_size);
		
		#pragma unroll
		for (int i = 0; i < m; ++i) {
			#pragma unroll
			for (int j = 0; j < m; ++j) {
				phiA[i + j*STRIDE] += 2.0 * creal((res[q] / poles[q]) * invA[i + j * STRIDE]);
			}
		}
	}
	
	free (work);
}

void expAc_variable(const int m, const int STRIDE, const double* A, const double c, double* phiA) {
	
	//allocate arrays
	double complex work* = (double complex*)malloc(work_size * sizeof(double complex));
	int ipiv[STRIDE] = {0};
	double complex invA[STRIDE * STRIDE];
	int info = 0;
	
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
					invA[i + j*STRIDE] = c * A[i + j*STRIDE] - poles[q];
				} else {
					invA[i + j*STRIDE] = c * A[i + j*STRIDE];
				}
			}
		}
		
		// takes care of (A * c - poles(q) * I)^-1
		getComplexInverseHessenberg (m, invA, ipiv, &info, work, work_size);
		
		#pragma unroll
		for (int i = 0; i < m; ++i) {
			#pragma unroll
			for (int j = 0; j < m; ++j) {
				phiA[i + j*STRIDE] += 2.0 * creal(res[q] * invA[i + j*WORK]);
			}
		}
	}
	
	free (work);
}