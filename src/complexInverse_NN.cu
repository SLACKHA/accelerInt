#include "header.h"
#include "solver_props.h"
#include <cuComplex.h>
///////////////////////////////////////////////////////////

__device__
int getComplexMax (const int n, const cuDoubleComplex *complexArr) {
	
	int maxInd = 0;
	if (n == 1)
		return maxInd;
	
	double maxVal = cuCabs(complexArr[0]);
	for (int i = 1; i < n; ++i) {
		if (cuCabs(complexArr[i]) > maxVal) {
			maxInd = i;
			maxVal = cuCabs(complexArr[i]);
		}
	}
	
	return maxInd;
}

///////////////////////////////////////////////////////////

__device__
void scaleComplex (const int n, const cuDoubleComplex val, cuDoubleComplex* arrX) {
	
	for (int i = 0; i < n; ++i) {
		arrX[i] = cuCmul(arrX[i], val);
	}
	
}

///////////////////////////////////////////////////////////

__device__
void swapComplex (const int n, cuDoubleComplex* arrX, const int incX, cuDoubleComplex* arrY, const int incY) {
	
	int ix = 0;
	int iy = 0;
	
	for (int i = 0; i < n; ++i) {
		cuDoubleComplex temp = arrX[ix];
		arrX[ix] = arrY[iy];
		arrY[iy] = temp;
		ix += incX;
		iy += incY;
	}
	
}

///////////////////////////////////////////////////////////

__device__
void complexGERU (const int n, const cuDoubleComplex alpha, const cuDoubleComplex* arrX,
									const cuDoubleComplex* arrY, const int incY, cuDoubleComplex* A, const int lda) {
	
	for (int j = 0; j < n; ++j) {
    	if (cuCabs(arrY[j * incY]) > 0.0) {
      
			cuDoubleComplex temp = cuCmul(alpha, arrY[j * incY]);
      
			for (int i = 0; i < n; ++i) {
				A[i + (lda * j)] = cuCfma(arrX[i], temp, A[i + (lda * j)]);
			}
      
		}    
	}
	
}

///////////////////////////////////////////////////////////

__device__
void getComplexLU (cuDoubleComplex* A, int* indPivot, int* info) {
	
	#pragma unroll
	for (int j = 0; j < NN; ++j) {
		
		// find pivot and test for singularity
		
		int jp = j + getComplexMax (NN - j, &A[j + (NN * j)]);
		indPivot[j] = jp;

    	if (cuCabs(A[jp + (NN * j)]) > 0.0) {
			
			// apply interchange to columns 1:n-1
			if (jp != j)
				swapComplex (NN, &A[j], NN, &A[jp], NN);
			
			// compute elements j+1:m-1 of the jth column
			
			if (j < NN - 1)
				scaleComplex (NN - j - 1, cuCdiv(make_cuDoubleComplex(1.0, 0.0), A[j + (NN * j)]), &A[j + 1 + (NN * j)]);
			
		} else if (*info == 0) {
			*info = j;
			break;
		}
		
		// update trailing submatrix
		if (j < NN - 1)
			complexGERU (NN - j - 1, make_cuDoubleComplex(-1.0, 0.0), &A[j + 1 + (NN * j)], &A[j + NN * (j + 1)], NN, &A[j + 1 + NN * (j + 1)], NN);
		
	}
}