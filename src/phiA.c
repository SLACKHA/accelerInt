#include <stdlib.h>
#include <complex.h>

#include "head.h"
//#include "linear-algebra.h"
#include "complexInverse.h"

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
////////////////////////////////////////////////////////////////////////

void phiAc (const double * A, const double c, double * phiA) {
	
	//double complex *invA = (double complex*) calloc (NN * NN, sizeof(double complex));
	double complex invA[NN * NN];
	
	#pragma unroll
	for (int i = 0; i < NN * NN; ++i) {
		phiA[i] = ZERO;
	}
	
	#pragma unroll
	for (int q = 0; q < N_RA; q += 2) {
		
		// takes care of (A * c - poles(q) * I)^-1
		//minv_cramer (A, c, q, invA);
		
		// compute transpose and multiply with constant
		for (int i = 0; i < NN; ++i) {
			for (int j = 0; j < NN; ++j) {
				// A - theta * I
				if (i == j) {
					//invA[i + j*NN] = c * A[j + i*NN] - poles[q];
					invA[i + j*NN] = c * A[i + j*NN] - poles[q];
				} else {
					//invA[i + j*NN] = c * A[j + i*NN];
					invA[i + j*NN] = c * A[i + j*NN];
				}
			}
		}
		
		// takes care of (A * c - poles(q) * I)^-1
		//getInverseComplex (NN, invA);
		getComplexInverse (NN, invA);
		
		#pragma unroll
		for (int i = 0; i < NN * NN; ++i) {
			phiA[i] += 2.0 * creal((res[q] / poles[q]) * invA[i]);
		}
		
	}
	
	//free (invA);
}