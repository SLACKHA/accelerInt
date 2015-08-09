#include <stdlib.h>
#include <complex.h>

#include "header.h"
//#include "linear-algebra.h"
#include "complexInverse.h"

extern double complex poles[N_RA];
extern double complex res[N_RA];


void phi2Ac_variable(const int m, const int STRIDE, const double* A, const double c, double* phiA) {
	
	double complex invA[m * m];
	
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
					invA[i + j*m] = c * A[i + j*STRIDE] - poles[q];
				} else {
					invA[i + j*m] = c * A[i + j*STRIDE];
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
				phiA[i + j*STRIDE] += 2.0 * creal((res[q] / (poles[q] * poles[q])) * invA[i + j*m]);
			}
		}
	}
	//free (invA);
}

void phiAc_variable(const int m, const int STRIDE, const double* A, const double c, double* phiA) {
	
	double complex invA[m * m];
	
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
					invA[i + j*m] = c * A[i + j*STRIDE] - poles[q];
				} else {
					invA[i + j*m] = c * A[i + j*STRIDE];
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
				phiA[i + j*STRIDE] += 2.0 * creal((res[q] / poles[q]) * invA[i + j*m]);
			}
		}
	}
	//free (invA);
}

void expAc_variable(const int m, const int STRIDE, const double* A, const double c, double* phiA) {
	
	double complex invA[m * m];
	
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
					invA[i + j*m] = c * A[i + j*STRIDE] - poles[q];
				} else {
					invA[i + j*m] = c * A[i + j*STRIDE];
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
				phiA[i + j*STRIDE] += 2.0 * creal(res[q] * invA[i + j*m]);
			}
		}
	}
	//free (invA);
}