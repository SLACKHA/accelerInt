#include "header.cuh"
#include "solver_props.cuh"
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
	
	for (int j = 0; j < NSP; ++j) {
		
		// find pivot and test for singularity
		
		int jp = j + getComplexMax (NSP - j, &A[j + (NSP * j)]);
		indPivot[j] = jp;

		if (cuCabs(A[jp + (NSP * j)]) > 0.0) {
			
			// apply interchange to columns 1:n-1
			if (jp != j)
				swapComplex (NSP, &A[j], NSP, &A[jp], NSP);
			
			// compute elements j+1:m-1 of the jth column
			
			if (j < NSP - 1)
				scaleComplex (NSP - j - 1, cuCdiv(make_cuDoubleComplex(1.0, 0.0), A[j + (NSP * j)]), &A[j + 1 + (NSP * j)]);
			
		} else if (*info == 0) {
			*info = j;
			break;
		}
		
		// update trailing submatrix
		if (j < NSP - 1)
			complexGERU (NSP - j - 1, make_cuDoubleComplex(-1.0, 0.0), &A[j + 1 + (NSP * j)], &A[j + NSP * (j + 1)], NSP, &A[j + 1 + NSP * (j + 1)], NSP);
		
	}
}