#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <string.h>

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
		//if (arrY[j * incY] != 0.0) {    
    if (cuCabs(arrY[j * incY]) > 0.0) {
      
			cuDoubleComplex temp = cuCmul(alpha, arrY[j * incY]);
      
			for (int i = 0; i < n; ++i) {
				//A[i + (lda * j)] += cuCmul(arrX[i], temp);
        A[i + (lda * j)] = cuCfma(arrX[i], temp, A[i + (lda * j)]);
			}
      
		}    
	}
	
}

///////////////////////////////////////////////////////////

__device__
void getComplexLU (cuDoubleComplex* A, int* indPivot) {
	
	//int info = 0;
	register cuDoubleComplex alpha = make_cuDoubleComplex(-1.0, 0.0);
	
	#pragma unroll
	for (int j = 0; j < NN; ++j) {
		
		// find pivot and test for singularity
		
		int jp = j + getComplexMax (NN - j, &A[j + (NN * j)]);
		indPivot[j] = jp;
		
		//if (A[jp + (n * j)] != 0.0) {
    	if (cuCabs(A[jp + (NN * j)]) > 0.0) {
			
			// apply interchange to columns 1:n-1
			if (jp != j)
				swapComplex (NN, &A[j], NN, &A[jp], NN);
			
			// compute elements j+1:m-1 of the jth column
			
			if (j < NN - 1)
				scaleComplex (NN - j - 1, cuCdiv(make_cuDoubleComplex(1.0, 0.0), A[j + (NN * j)]), &A[j + 1 + (NN * j)]);
			
		} //else if (info == 0) {
			//info = j + 1;
		//}
		
		// update trailing submatrix
		if (j < NN - 1)
			complexGERU (NN - j - 1, alpha, &A[j + 1 + (NN * j)], &A[j + NN * (j + 1)], NN, &A[j + 1 + NN * (j + 1)], NN);
		
	}
	
	//return info;
}

///////////////////////////////////////////////////////////

__device__
void multiplyComplexUpperMV (const int n, cuDoubleComplex* x, const int lda, const cuDoubleComplex* A) {
	
	for (int j = 0; j < n; ++j) {
		//if (x[j] != 0.0) {
    if (cuCabs(x[j]) > 0.0) {
      
			cuDoubleComplex temp = x[j];
			for (int i = 0; i < j; ++i) {
				//x[i] += temp * A[i + (lda * j)];
       			x[i] = cuCfma(temp, A[i + (lda * j)], x[i]);
			}
			//x[j] *= A[j + (lda * j)];
      		x[j] = cuCmul(x[j], A[j + (lda * j)]);
		}
	}
	
}

///////////////////////////////////////////////////////////

__device__
void complexGEMV (const int m, const int n, const cuDoubleComplex alpha, const cuDoubleComplex* A, 
									const cuDoubleComplex* arrX, cuDoubleComplex* arrY) {
	
	// first: y = beta*y
	// beta = 1, so nothing
	
	// second: y = alpha*A*x + y
	
	for (int j = 0; j < n - 1; ++j) {

    if (cuCabs(arrX[j]) > 0.0) {
			cuDoubleComplex temp = cuCmul(alpha, arrX[j]);
      
			for (int i = 0; i < m; ++i) {
				//arrY[i] += temp * A[i + (m * j)];
        arrY[i] = cuCfma(temp, A[i + (NN * j)], arrY[i]);
			}
		}
	}
	
}

///////////////////////////////////////////////////////////

__device__
void getComplexInverseLU (const int n, cuDoubleComplex* A, const int* indPivot, cuDoubleComplex* work) {
	
	//int info = 0;
	
	// form inv(U)
	for (int j = 0; j < n; ++j) {
		A[j + (NN * j)] = cuCdiv(make_cuDoubleComplex(1.0, 0.0), A[j + (NN * j)]);
		cuDoubleComplex Ajj = cuCmul(make_cuDoubleComplex(-1.0, 0.0), A[j + (NN * j)]);
		
		// compute elements 0:j-1 of jth column
		multiplyComplexUpperMV (j, &A[NN * j], NN, A);
		
		// scale
		scaleComplex (j, Ajj, &A[NN * j]);
	}
	
	// solve equation inv(A)*L = inv(U) for inv(A)
	
	for (int j = n - 1; j >= 0; --j) {
		
		// copy current column of L to work and replace with 0.0s
		for (int i = j + 1; i < n; ++i) {
			work[i] = A[i + (NN * j)];
			A[i + (NN * j)] = make_cuDoubleComplex(0.0, 0.0);
		}
		
		// compute current column of inv(A)
		if (j < n - 1)
			complexGEMV (n, n - j, make_cuDoubleComplex(-1.0, 0.0), &A[NN * (j + 1)], &work[j + 1], &A[NN * j]);
		
	}
	
	// apply column interchanges
	
	for (int j = n - 2; j >= 0; --j) {
    
		if (indPivot[j] != j)
			swapComplex (n, &A[NN * j], 1, &A[NN * indPivot[j]], 1);
	}
	
	//return info;
}

///////////////////////////////////////////////////////////

__device__
void getComplexInverse (cuDoubleComplex* A) {
	
	// pivot indices
	//int* ipiv = (int*) calloc (n, sizeof(int));
  	int ipiv[NN];
	
	// output flag
	//int info = 0;
	
	// first get LU factorization
	getComplexLU (A, ipiv);
	
	// check for successful exit
  /*
	if (info != 0) {
		printf ("getComplexLU failure, info = %d.\n", info);
		exit (1);
	}
  */
	
	// work array
	//cuDoubleComplex* work = (double complex*) calloc (n, sizeof(double complex));
  	cuDoubleComplex work[NN];
	
	// now get inverse
	getComplexInverseLU (NN, A, ipiv, work);
	
	//free (work);
	//free (ipiv);
	
	// check for successful exit
  /*
	if (info != 0) {
		printf ("getComplexInverseLU failure, info = %d.\n", info);
		exit (1);
	}
  */
	
}