#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <string.h>

#include "header.cuh"
#include "solver_props.cuh"

///////////////////////////////////////////////////////////

__device__
int getMax (const int n, const double * __restrict__ Arr) {
	
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
void scale (const int n, const double val, double* __restrict__ arrX) {
	
	for (int i = 0; i < n; ++i) {
		arrX[INDEX(i)] *= val;
	}
	
}

///////////////////////////////////////////////////////////

__device__
void swap (const int n, double* __restrict__ arrX, const int incX, double* __restrict__ arrY, const int incY) {
	
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
void GERU (const int n, const double alpha, const double* __restrict__ arrX,
				const double* __restrict__ arrY, const int incY, double* __restrict__ A, const int lda) {
	
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
void getLU (const int n, const int LDA, double* __restrict__ A, int* __restrict__ indPivot, int* __restrict__ info) {
	
	for (int j = 0; j < n; ++j) {
		
		// find pivot and test for singularity
		
		int jp = j + getMax (n - j, &A[GRID_DIM * (j + (LDA * j))]);
		indPivot[INDEX(j)] = jp;

    	if (fabs(A[INDEX(jp + (LDA * j))]) > 0.0) {
			
			// apply interchange to columns 1:n-1
			if (jp != j)
				swap(n, &A[GRID_DIM * (j)], LDA, &A[GRID_DIM * (jp)], LDA);
			
			// compute elements j+1:m-1 of the jth column
			
			if (j < n - 1)
				scale(n - j - 1, 1.0 / A[INDEX(j + (LDA * j))], &A[GRID_DIM * (j + 1 + (LDA * j))]);
			
		} else if (*info == 0) {
			*info = j;
			break;
		}
		
		// update trailing submatrix
		if (j < n - 1)
			GERU (n - j - 1, -1.0, &A[GRID_DIM * (j + 1 + (LDA * j))], &A[GRID_DIM * (j + LDA * (j + 1))], LDA, &A[GRID_DIM * (j + 1 + LDA * (j + 1))], LDA);	
	}
}