#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <string.h>

#include "header.cuh"
#include "solver_props.cuh"

///////////////////////////////////////////////////////////

__device__
int getMax (const int n, const double *Arr) {
	
	int maxInd = 0;
	if (n == 1)
		return maxInd;
	
	double maxVal = fabs(Arr[0]);
	for (int i = 1; i < n; ++i) {
		if (fabs(Arr[i]) > maxVal) {
			maxInd = i;
			maxVal = fabs(Arr[i]);
		}
	}
	
	return maxInd;
}

///////////////////////////////////////////////////////////

__device__
void scale (const int n, const double val, double* arrX) {
	
	for (int i = 0; i < n; ++i) {
		arrX[i] *= val;
	}
	
}

///////////////////////////////////////////////////////////

__device__
void swap (const int n, double* arrX, const int incX, double* arrY, const int incY) {
	
	int ix = 0;
	int iy = 0;
	
	for (int i = 0; i < n; ++i) {
		double temp = arrX[ix];
		arrX[ix] = arrY[iy];
		arrY[iy] = temp;
		ix += incX;
		iy += incY;
	}
	
}

///////////////////////////////////////////////////////////

__device__
void GERU (const int n, const double alpha, const double* arrX,
									const double* arrY, const int incY, double* A, const int lda) {
	
	for (int j = 0; j < n; ++j) {
    	if (fabs(arrY[j * incY]) > 0.0) {
      
			double temp = alpha * arrY[j * incY];
      
			for (int i = 0; i < n; ++i) {
				A[i + (lda * j)] += arrX[i] * temp;
			}
      
		}    
	}
	
}

///////////////////////////////////////////////////////////

__device__
void getLU (double* A, int* indPivot, int* info) {
	
	#pragma unroll
	for (int j = 0; j < NSP; ++j) {
		
		// find pivot and test for singularity
		
		int jp = j + getMax (NSP - j, &A[j + (NSP * j)]);
		indPivot[j] = jp;

    	if (fabs(A[jp + (NSP * j)]) > 0.0) {
			
			// apply interchange to columns 1:n-1
			if (jp != j)
				swap(NSP, &A[j], NSP, &A[jp], NSP);
			
			// compute elements j+1:m-1 of the jth column
			
			if (j < NSP - 1)
				scale(NSP - j - 1, 1.0 / A[j + (NSP * j)], &A[j + 1 + (NSP * j)]);
			
		} else if (*info == 0) {
			*info = j;
			break;
		}
		
		// update trailing submatrix
		if (j < NSP - 1)
			GERU (NSP - j - 1, -1.0, &A[j + 1 + (NSP * j)], &A[j + NSP * (j + 1)], NSP, &A[j + 1 + NSP * (j + 1)], NSP);	
	}
}