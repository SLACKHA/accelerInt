#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <string.h>

#include "header.h"
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
void getComplexLU (const int n, cuDoubleComplex* A, int* indPivot) {
	
	//int info = 0;
	cuDoubleComplex alpha = {-1.0, 0};
	
	for (int j = 0; j < n; ++j) {
		
		// find pivot and test for singularity
		
		int jp = j + getComplexMax (n - j, &A[j + (n * j)]);
		indPivot[j] = jp;
		
		//if (A[jp + (n * j)] != 0.0) {
    if (cuCabs(A[jp + (n * j)]) > 0.0) {
			
			// apply interchange to columns 1:n-1
			if (jp != j)
				swapComplex (n, &A[j], n, &A[jp], n);
			
			// compute elements j+1:m-1 of the jth column
			
			if (j < n - 1)
				scaleComplex (n - j - 1, cuCdiv(make_cuDoubleComplex(1.0, 0.0), A[j + (n * j)]), &A[j + 1 + (n * j)]);
			
		} //else if (info == 0) {
			//info = j + 1;
		//}
		
		// update trailing submatrix
		if (j < n - 1)
			complexGERU (n - j - 1, alpha, &A[j + 1 + (n * j)], &A[j + n * (j + 1)], n, &A[j + 1 + n * (j + 1)], n);
		
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
        arrY[i] = cuCfma(temp, A[i + (m * j)], arrY[i]);
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
		A[j + (n * j)] = cuCdiv(make_cuDoubleComplex(1.0, 0.0), A[j + (n * j)]);
		cuDoubleComplex Ajj = cuCmul(make_cuDoubleComplex(-1.0, 0.0), A[j + (n * j)]);
		
		// compute elements 0:j-1 of jth column
		multiplyComplexUpperMV (j, &A[n * j], n, A);
		
		// scale
		scaleComplex (j, Ajj, &A[n * j]);
	}
	
	// solve equation inv(A)*L = inv(U) for inv(A)
	
	for (int j = n - 1; j >= 0; --j) {
		
		// copy current column of L to work and replace with 0.0s
		for (int i = j + 1; i < n; ++i) {
			work[i] = A[i + (n * j)];
			A[i + (n * j)] = make_cuDoubleComplex(0.0, 0.0);
		}
		
		// compute current column of inv(A)
		if (j < n - 1)
			complexGEMV (n, n - j, make_cuDoubleComplex(-1.0, 0.0), &A[n * (j + 1)], &work[j + 1], &A[n * j]);
		
	}
	
	// apply column interchanges
	
	for (int j = n - 2; j >= 0; --j) {
    
		if (indPivot[j] != j)
			swapComplex (n, &A[n * j], 1, &A[n * indPivot[j]], 1);
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
	getComplexLU (NN, A, ipiv);
	
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