#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include "header.h"

///////////////////////////////////////////////////////////

static inline
int getMax (int n, double* arr) {
	
	int maxInd = 0;
	if (n == 1)
		return maxInd;
	
	double maxVal = fabs(arr[0]);
	for (int i = 1; i < n; ++i) {
		if (fabs(arr[i]) > maxVal) {
			maxInd = i;
			maxVal = fabs(arr[i]);
		}
	}
	
	return maxInd;
}

///////////////////////////////////////////////////////////

static inline
void scale (int n, double val, double* arrX) {
	
	for (int i = 0; i < n; ++i) {
		arrX[i] *= val;
	}
	
}

///////////////////////////////////////////////////////////

static inline
void swap (int n, double* arrX, int incX, double* arrY, int incY) {
	
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

static inline
void multiplyUpperMV (int n, double* x, int lda, double* A) {
	
	for (int j = 0; j < n; ++j) {
		//if (x[j] != 0.0) {
    if (fabs(x[j]) > 0.0) {
      
			double temp = x[j];
			for (int i = 0; i < j; ++i) {
				x[i] += temp * A[i + (lda * j)];
			}
			x[j] *= A[j + (lda * j)];
		}
	}
}

///////////////////////////////////////////////////////////

static inline
void GEMV (int m, int n, double alpha, double* A, 
									double* arrX, double* arrY) {
	
	// first: y = beta*y
	// beta = 1, so nothing
	
	// second: y = alpha*A*x + y
	
	for (int j = 0; j < n - 1; ++j) {
		//if (arrX[j] != 0.0) {
    if (fabs(arrX[j]) > 0.0) {
			double temp = alpha * arrX[j];
			for (int i = 0; i < m; ++i) {
				arrY[i] += temp * A[i + (m * j)];
			}
		}
	}
	
}

///////////////////////////////////////////////////////////

static inline
int getInverseLU (int n, double* A, int* indPivot, double* work) {
	
	int info = 0;
	
	// form inv(U)
	for (int j = 0; j < n; ++j) {
		A[j + (n * j)] = 1.0 / A[j + (n * j)];
		double Ajj = -A[j + (n * j)];
		
		// compute elements 0:j-1 of jth column
		multiplyUpperMV (j, &A[n * j], n, A);
		
		// scale
		scale (j, Ajj, &A[n * j]);
	}
	
	// solve equation inv(A)*L = inv(U) for inv(A)
	
	for (int j = n - 1; j >= 0; --j) {
		
		// copy current column of L to work and replace with 0.0s
		for (int i = j + 1; i < n; ++i) {
			work[i] = A[i + (n * j)];
			A[i + (n * j)] = 0.0;
		}
		
		// compute current column of inv(A)
		if (j < n - 1)
			GEMV (n, n - j, -1.0, &A[n * (j + 1)], &work[j + 1], &A[n * j]);
		
	}
	
	// apply column interchanges
	
	for (int j = n - 2; j >= 0; --j) {
    
		if (indPivot[j] != j)
			swap (n, &A[n * j], 1, &A[n * indPivot[j]], 1);
	}
	
	return info;
}

//adapted from Matrix Computations
//Gene H. Golub, Charles F. Van Loan
static inline
int getHessenbergLU(const int n, double* A, int* indPivot)
{
	int last_free = 0;
	for (int i = 0; i < n - 1; i ++)
	{
		if (fabs(A[i * n + i]) < fabs(A[i * n + i + 1]))
		{
			//swap rows
			swap(n - last_free, &A[last_free * n + i], n, &A[last_free * n + i + 1], n);
			indPivot[i] = i + 1;
		}
		else
		{
			indPivot[i] = i;
			last_free = i;
		}
		if (fabs(A[i * n + i]) > 0.0)
		{
			double tau = A[i * n + i + 1] / A[i * n + i];
			for (int j = i + 1; j < n; j++)
			{
				A[j * n + i + 1] -= tau * A[j * n + i];
			}
			A[i * n + i + 1] = tau;
		}
		else 
		{
			return i;
		}
	}
	//last index is not pivoted
	indPivot[n - 1] = n - 1;
	return 0;
}

void getInverseHessenberg (const int n, double* A)
{
	// pivot indices
	int* ipiv = (int*) calloc (n, sizeof(int));
	
	// output flag
	int info = 0;
	
	// first get LU factorization
	info = getHessenbergLU (n, A, ipiv);

#ifndef NDEBUG
	if (info != 0)
	{
		printf ("getHessenbergLU failure, info = %d.\n", info);
		exit (1);
	}
#endif

	// work array
	double* work = (double*) calloc (n, sizeof(double));
  	// memset (work, 0.0, n * sizeof(double));
	
	// now get inverse
	info = getInverseLU (n, A, ipiv, work);
	
	free (work);
	free (ipiv);
	
#ifndef NDEBUG
	// check for successful exit
	if (info != 0) {
		printf ("getInverseLU failure, info = %d.\n", info);
		exit (1);
	}
#endif
}