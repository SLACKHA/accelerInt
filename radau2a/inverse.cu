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
	
	double maxVal = fabs(Arr[INDEX(0)]);
	for (int i = 1; i < n; ++i) {
		if (fabs(Arr[INDEX(i)]) > maxVal) {
			maxInd = i;
			maxVal = fabs(Arr[INDEX(i)]);
		}
	}
	
	return maxInd;
}

///////////////////////////////////////////////////////////

__device__
void scale (const int n, const double val, double* arrX) {
	
	for (int i = 0; i < n; ++i) {
		arrX[INDEX(i)] *= val;
	}
	
}

///////////////////////////////////////////////////////////

__device__
void swap (const int n, double* arrX, const int incX, double* arrY, const int incY) {
	
	int ix = 0;
	int iy = 0;
	
	for (int i = 0; i < n; ++i) {
		double temp = arrX[INDEX(ix)];
		arrX[INDEX(ix)] = arrY[INDEX(iy)];
		arrY[INDEX(iy)] = temp;
		ix += incX;
		iy += incY;
	}
	
}

///////////////////////////////////////////////////////////

__device__
void GERU (const int n, const double alpha, const double* arrX,
									const double* arrY, const int incY, double* A, const int lda) {
	
	for (int j = 0; j < n; ++j) {
    	if (fabs(arrY[INDEX(j * incY)]) > 0.0) {
      
			double temp = alpha * arrY[INDEX(j * incY)];
      
			for (int i = 0; i < n; ++i) {
				A[INDEX(i + (lda * j))] += arrX[INDEX(i)] * temp;
			}
      
		}    
	}
	
}

///////////////////////////////////////////////////////////

__device__
void getLU (double* A, int* indPivot, int* info) {
	
	for (int j = 0; j < NSP; ++j) {
		
		// find pivot and test for singularity
		
		int jp = j + getMax (NSP - j, &A[INDEX(j + (NSP * j))]);
		indPivot[INDEX(j)] = jp;

    	if (fabs(A[INDEX(jp + (NSP * j))]) > 0.0) {
			
			// apply interchange to columns 1:n-1
			if (jp != j)
				swap(NSP, &A[INDEX(j)], NSP, &A[INDEX(jp)], NSP);
			
			// compute elements j+1:m-1 of the jth column
			
			if (j < NSP - 1)
				scale(NSP - j - 1, 1.0 / A[INDEX(j + (NSP * j))], &A[INDEX(j + 1 + (NSP * j))]);
			
		} else if (*info == 0) {
			*info = j;
			break;
		}
		
		// update trailing submatrix
		if (j < NSP - 1)
			GERU (NSP - j - 1, -1.0, &A[INDEX(j + 1 + (NSP * j))], &A[INDEX(j + NSP * (j + 1))], NSP, &A[INDEX(j + 1 + NSP * (j + 1))], NSP);	
	}
}