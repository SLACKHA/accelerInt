#include "header.cuh"
#include "solver_props.cuh"
#include <cuComplex.h>
///////////////////////////////////////////////////////////

__device__
int getComplexMax (const int n, const cuDoubleComplex *complexArr) {
	
	int maxInd = 0;
	if (n == 1)
		return maxInd;
	
	double maxVal = cuCabs(complexArr[INDEX(0)]);
	for (int i = 1; i < n; ++i) {
		if (cuCabs(complexArr[INDEX(i)]) > maxVal) {
			maxInd = i;
			maxVal = cuCabs(complexArr[INDEX(i)]);
		}
	}
	
	return maxInd;
}

///////////////////////////////////////////////////////////

__device__
void scaleComplex (const int n, const cuDoubleComplex val, cuDoubleComplex* arrX) {
	
	for (int i = 0; i < n; ++i) {
		arrX[INDEX(i)] = cuCmul(arrX[INDEX(i)], val);
	}
	
}

///////////////////////////////////////////////////////////

__device__
void swapComplex (const int n, cuDoubleComplex* arrX, const int incX, cuDoubleComplex* arrY, const int incY) {
	
	int ix = 0;
	int iy = 0;
	
	for (int i = 0; i < n; ++i) {
		cuDoubleComplex temp = arrX[INDEX(ix)];
		arrX[INDEX(ix)] = arrY[INDEX(iy)];
		arrY[INDEX(iy)] = temp;
		ix += incX;
		iy += incY;
	}
	
}

///////////////////////////////////////////////////////////

__device__
void complexGERU (const int n, const cuDoubleComplex alpha, const cuDoubleComplex* arrX,
									const cuDoubleComplex* arrY, const int incY, cuDoubleComplex* A, const int lda) {
	
	for (int j = 0; j < n; ++j) {
    	if (cuCabs(arrY[INDEX(j * incY)]) > 0.0) {
      
			cuDoubleComplex temp = cuCmul(alpha, arrY[INDEX(j * incY)]);
      
			for (int i = 0; i < n; ++i) {
				A[INDEX(i + (lda * j))] = cuCfma(arrX[INDEX(i)], temp, A[INDEX(i + (lda * j))]);
			}
      
		}    
	}
	
}

///////////////////////////////////////////////////////////

__device__
void getComplexLU (cuDoubleComplex* A, int* indPivot, int* info) {
	
	for (int j = 0; j < NSP; ++j) {
		
		// find pivot and test for singularity
		
		int jp = j + getComplexMax (NSP - j, &A[INDEX(j + (NSP * j))]);
		indPivot[INDEX(j)] = jp;

		if (cuCabs(A[INDEX(jp + (NSP * j))]) > 0.0) {
			
			// apply interchange to columns 1:n-1
			if (jp != j)
				swapComplex (NSP, &A[INDEX(j)], NSP, &A[INDEX(jp)], NSP);
			
			// compute elements j+1:m-1 of the jth column
			
			if (j < NSP - 1)
				scaleComplex (NSP - j - 1, cuCdiv(make_cuDoubleComplex(1.0, 0.0), A[INDEX(j + (NSP * j))]), &A[INDEX(j + 1 + (NSP * j))]);
			
		} else if (*info == 0) {
			*info = j;
			break;
		}
		
		// update trailing submatrix
		if (j < NSP - 1)
			complexGERU (NSP - j - 1, make_cuDoubleComplex(-1.0, 0.0), &A[INDEX(j + 1 + (NSP * j))], &A[INDEX(j + NSP * (j + 1))], NSP, &A[INDEX(j + 1 + NSP * (j + 1))], NSP);
		
	}
}