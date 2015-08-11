#include <stdlib.h>
#include <cuComplex.h>

#include "header.h"
//#include "linear-algebra.h"
#include "complexInverse.cuh"

extern __device__ __constant__ cuDoubleComplex poles[N_RA];
extern __device__ __constant__ cuDoubleComplex res[N_RA];

////////////////////////////////////////////////////////////////////////

__device__
void phiAc (const double * A, const double c, double * phiA) {
	
	cuDoubleComplex invA[NN * NN];
	
	#pragma unroll
	for (int i = 0; i < NN * NN; ++i) {
		phiA[i] = 0.0;
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
					invA[i + j*NN] = cuCsub(make_cuDoubleComplex(c * A[i + j*NN], 0.0), poles[q]);
				} else {
					invA[i + j*NN] = make_cuDoubleComplex(c * A[i + j*NN], 0.0);
				}
			}
		}
		
		// takes care of (A * c - poles(q) * I)^-1
		getComplexInverse (invA);
		
		#pragma unroll
		for (int i = 0; i < NN * NN; ++i) {
      //phiA[i] += 2.0 * creal((res[q] / poles[q]) * invA[i]);
			phiA[i] += 2.0 * cuCreal( cuCmul( cuCdiv(res[q], poles[q]), invA[i] ) );
		}
		
	}

}