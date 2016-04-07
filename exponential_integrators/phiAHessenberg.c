#include <stdlib.h>
#include <complex.h>

#include "header.h"
#include "lapack_dfns.h"
#include "complexInverse.h"
#include "solver_options.h"
#include "solver_props.h"

extern double complex poles[N_RA];
extern double complex res[N_RA];

int phi2Ac_variable(const int m, const double* A, const double c, double* phiA) {
	
	//allocate arrays
	int ipiv[STRIDE] = {0};
	double complex invA[STRIDE * STRIDE];
	int info = 0;
	
	for (int i = 0; i < m; ++i) {
		
		for (int j = 0; j < m; ++j) {
			phiA[i + j*STRIDE] = 0.0;
		}
	}

	
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
		getComplexInverseHessenberg (m, invA, ipiv, &info);

		if (info != 0)
			return info;
		
		
		for (int i = 0; i < m; ++i) {
			
			for (int j = 0; j < m; ++j) {
				phiA[i + j*STRIDE] += 2.0 * creal((res[q] / (poles[q] * poles[q])) * invA[i + j * STRIDE]);
			}
		}
	}
	
	return 0;
}

int phiAc_variable(const int m, const double* A, const double c, double* phiA) {
	
	//allocate arrays
	int ipiv[STRIDE] = {0};
	double complex invA[STRIDE * STRIDE];
	int info = 0;
	
	for (int i = 0; i < m; ++i) {
		
		for (int j = 0; j < m; ++j) {
			phiA[i + j*STRIDE] = 0.0;
		}
	}

	
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
		getComplexInverseHessenberg (m, invA, ipiv, &info);

		if (info != 0)
			return info;
		
		
		for (int i = 0; i < m; ++i) {
			
			for (int j = 0; j < m; ++j) {
				phiA[i + j*STRIDE] += 2.0 * creal((res[q] / poles[q]) * invA[i + j * STRIDE]);
			}
		}
	}
	
	return 0;
}

int expAc_variable(const int m, const double* A, const double c, double* phiA) {
	
	//allocate arrays
	int ipiv[STRIDE] = {0};
	double complex invA[STRIDE * STRIDE];
	int info = 0;
	
	for (int i = 0; i < m; ++i) {
		
		for (int j = 0; j < m; ++j) {
			phiA[i + j*STRIDE] = 0.0;
		}
	}

	
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
		getComplexInverseHessenberg (m, invA, ipiv, &info);

		if (info != 0)
			return info;
		
		
		for (int i = 0; i < m; ++i) {
			
			for (int j = 0; j < m; ++j) {
				phiA[i + j*STRIDE] += 2.0 * creal(res[q] * invA[i + j*STRIDE]);
			}
		}
	}
	
	return 0;
}