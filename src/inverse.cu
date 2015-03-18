#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <string.h>

#include "header.h"
#include "solver_props.h"

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
		arrX[i] = arrX[i] * val;
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
		//if (arrY[j * incY] != 0.0) {    
    if (fabs(arrY[j * incY]) > 0.0) {
      
			double temp = alpha * arrY[j * incY];
      
			for (int i = 0; i < n; ++i) {
        		A[i + (lda * j)] = fma(arrX[i], temp, A[i + (lda * j)]);
			}
      
		}    
	}
	
}

///////////////////////////////////////////////////////////

__device__
void getLU (double* A, int* indPivot) {
	
	//int info = 0;
	register double alpha = -1.0;
	
	#pragma UNROLL
	for (int j = 0; j < NN; ++j) {
		
		// find pivot and test for singularity
		
		int jp = j + getMax (NN - j, &A[j + (NN * j)]);
		indPivot[j] = jp;
		
    	if (fabs(A[jp + (NN * j)]) > 0.0) {
			
			// apply interchange to columns 1:n-1
			if (jp != j)
				swap (NN, &A[j], NN, &A[jp], NN);
			
			// compute elements j+1:m-1 of the jth column
			
			if (j < NN - 1)
				scale (NN - j - 1, 1.0 / A[j + (NN * j)], &A[j + 1 + (NN * j)]);
			
		} 
		//else if (info == 0) {
			//info = j + 1;
		//}
		
		// update trailing submatrix
		if (j < NN - 1)
			GERU (NN - j - 1, alpha, &A[j + 1 + (NN * j)], &A[j + NN * (j + 1)], NN, &A[j + 1 + NN * (j + 1)], NN);
		
	}
	
	//return info;
}

///////////////////////////////////////////////////////////

__device__
void multiplyUpperMV (const int n, double* x, const int lda, const double* A) {
	
	for (int j = 0; j < n; ++j) {
		//if (x[j] != 0.0) {
    if (fabs(x[j]) > 0.0) {
      
			double temp = x[j];
			for (int i = 0; i < j; ++i) {
				//x[i] += temp * A[i + (lda * j)];
       			x[i] += temp * A[i + (lda * j)];
			}
			//x[j] *= A[j + (lda * j)];
      		x[j] = x[j] * A[j + (lda * j)];
		}
	}
	
}

///////////////////////////////////////////////////////////

__device__
void GEMV (const int m, const int n, const double alpha, const double* A, 
									const double* arrX, double* arrY) {
	
	// first: y = beta*y
	// beta = 1, so nothing
	
	// second: y = alpha*A*x + y
	
	for (int j = 0; j < n - 1; ++j) {

    if (fabs(arrX[j]) > 0.0) {
			double temp = alpha * arrX[j];
      
			for (int i = 0; i < m; ++i) {
				//arrY[i] += temp * A[i + (m * j)];
        		arrY[i] += temp * A[i + (NN * j)];
			}
		}
	}
	
}

///////////////////////////////////////////////////////////

__device__
void getInverseLU (double* A, const int* indPivot, double* work) {
	
	//int info = 0;
	
	// form inv(U)
	#pragma unroll
	for (int j = 0; j < NN; ++j) {
		A[j + (NN * j)] = 1.0 / A[j + (NN * j)];
		double Ajj = -A[j + (NN * j)];
		
		// compute elements 0:j-1 of jth column
		multiplyUpperMV (j, &A[NN * j], NN, A);
		
		// scale
		scale (j, Ajj, &A[NN * j]);
	}
	
	// solve equation inv(A)*L = inv(U) for inv(A)
	
	#pragma unroll
	for (int j = NN - 1; j >= 0; --j) {
		
		// copy current column of L to work and replace with 0.0s
		#pragma unroll
		for (int i = j + 1; i < NN; ++i) {
			work[i] = A[i + (NN * j)];
			A[i + (NN * j)] = 0;
		}
		
		// compute current column of inv(A)
		if (j < NN - 1)
			GEMV (NN, NN - j, -1, &A[NN * (j + 1)], &work[j + 1], &A[NN * j]);
		
	}
	
	// apply column interchanges
	#pragma unroll
	for (int j = NN - 2; j >= 0; --j) {
    
		if (indPivot[j] != j)
			swap (NN, &A[NN * j], 1, &A[NN * indPivot[j]], 1);
	}
	
	//return info;
}

///////////////////////////////////////////////////////////

__device__
void getInverse (double* A) {
	
	// pivot indices
	//int* ipiv = (int*) calloc (n, sizeof(int));
  	int ipiv[NN];
	
	// output flag
	//int info = 0;
	
	// first get LU factorization
	getLU (A, ipiv);
	
	// check for successful exit
  /*
	if (info != 0) {
		printf ("getLU failure, info = %d.\n", info);
		exit (1);
	}
  */
	
	// work array
	//double* work = (double *) calloc (n, sizeof(double ));
  	double work[NN];
	
	// now get inverse
	getInverseLU (A, ipiv, work);
	
	//free (work);
	//free (ipiv);
	
	// check for successful exit
  /*
	if (info != 0) {
		printf ("getInverseLU failure, info = %d.\n", info);
		exit (1);
	}
  */
	
}