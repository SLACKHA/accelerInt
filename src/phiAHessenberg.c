#include <stdlib.h>
#include <complex.h>

#include "header.h"
//#include "linear-algebra.h"
#include "complexInverseHessenberg.h"

extern Real complex poles[N_RA];
extern Real complex res[N_RA];

////////////////////////////////////////////////////////////////////////

/*
void phiAv (const double* A, const double c, const double* v, double* phiAv) {
	
	double complex x[NN];
	
	// temporary matrix
	double complex* At = (double complex *) calloc (NN * NN, sizeof(double complex));
	
	for (int i = 0; i < NN; ++i) {
		phiAv[i] = 0.0;
	}
	
	#pragma unroll
	for (uint q = 0; q < N_RA; q += 2) {
	
		// compute transpose and multiply with constant
		for (int i = 0; i < NN; ++i) {
			for (int j = 0; j < NN; ++j) {
				// A - theta * I
				if (i == j) {
					At[i + j*NN] = c * A[j + i*NN] - poles[q];
				} else {
					At[i + j*NN] = c * A[j + i*NN];
				}
			}
		}
		
		// takes care of (A * c - poles(q) * I)^-1
		linSolveComplex (NN, At, v, x);
		
		#pragma unroll
		for (uint i = 0; i < NN; ++i) {
			phiAv[i] += 2.0 * creal((res[q] / poles[q]) * x[i]);
		}
		
	}
	
	free (At);
}
*/

void phiAc_variable(const int m, const int STRIDE, const Real* A, const Real c, Real* phiA) {
	
	double complex invA[m * m];
	
	#pragma unroll
	for (int i = 0; i < m * m; ++i) {
		phiA[i] = ZERO;
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
		//getInverseComplex (NN, invA);
		getComplexInverseHessenberg (m, STRIDE, invA);
		
		#pragma unroll
		for (int i = 0; i < m * m; ++i) {
			phiA[i] += 2.0 * creal((res[q] / poles[q]) * invA[i]);
		}
		
	}
	
	//free (invA);
}
