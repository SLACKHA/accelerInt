#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <complex.h>
#include <string.h>

///////////////////////////////////////////////////////////

static inline
int getComplexMax (int n, double complex* complexArr) {
	
	int maxInd = 0;
	if (n == 1)
		return maxInd;
	
	double maxVal = cabs(complexArr[0]);
	for (int i = 1; i < n; ++i) {
		if (cabs(complexArr[i]) > maxVal) {
			maxInd = i;
			maxVal = cabs(complexArr[i]);
		}
	}
	
	return maxInd;
}

///////////////////////////////////////////////////////////

static inline
void scaleComplex (int n, double complex val, double complex* arrX) {
	
	for (int i = 0; i < n; ++i) {
		arrX[i] *= val;
	}
	
}

///////////////////////////////////////////////////////////

static inline
void swapComplex (int n, double complex* arrX, int incX, double complex* arrY, int incY) {
	
	int ix = 0;
	int iy = 0;
	
	for (int i = 0; i < n; ++i) {
		double complex temp = arrX[ix];
		arrX[ix] = arrY[iy];
		arrY[iy] = temp;
		ix += incX;
		iy += incY;
	}
	
}

///////////////////////////////////////////////////////////

static inline
void multiplyComplexUpperMV (int n, double complex* x, int lda, double complex* A) {
	
	for (int j = 0; j < n; ++j) {
		//if (x[j] != 0.0) {
    if (cabs(x[j]) > 0.0) {
      
			double complex temp = x[j];
			for (int i = 0; i < j; ++i) {
				x[i] += temp * A[i + (lda * j)];
			}
			x[j] *= A[j + (lda * j)];
		}
	}
	
}

///////////////////////////////////////////////////////////

static inline
void complexGEMV (int m, int n, int STRIDE, double complex alpha, double complex* A, 
									double complex* arrX, double complex* arrY) {
	
	// first: y = beta*y
	// beta = 1, so nothing
	
	// second: y = alpha*A*x + y
	
	for (int j = 0; j < n - 1; ++j) {
		//if (arrX[j] != 0.0) {
    if (cabs(arrX[j]) > 0.0) {
			double complex temp = alpha * arrX[j];
			for (int i = 0; i < m; ++i) {
				arrY[i] += temp * A[i + (STRIDE * j)];
			}
		}
	}
	
}

///////////////////////////////////////////////////////////

static inline
int getComplexInverseLU (int n, int STRIDE, double complex* A, int* indPivot, double complex* work) {
	
	int info = 0;
	
	// form inv(U)
	for (int j = 0; j < n; ++j) {
		A[j + (STRIDE * j)] = 1.0 / A[j + (STRIDE * j)];
		double complex Ajj = -A[j + (STRIDE * j)];
		
		// compute elements 0:j-1 of jth column
		multiplyComplexUpperMV (j, &A[STRIDE * j], STRIDE, A);
		
		// scale
		scaleComplex (j, Ajj, &A[STRIDE * j]);
	}
	
	// solve equation inv(A)*L = inv(U) for inv(A)
	
	for (int j = n - 1; j >= 0; --j) {
		
		// copy current column of L to work and replace with 0.0s
		for (int i = j + 1; i < n; ++i) {
			work[i] = A[i + (STRIDE * j)];
			A[i + (STRIDE * j)] = 0.0;
		}
		
		// compute current column of inv(A)
		if (j < n - 1)
			complexGEMV (n, n - j, STRIDE, -1.0, &A[STRIDE * (j + 1)], &work[j + 1], &A[STRIDE * j]);
		
	}
	
	// apply column interchanges
	
	for (int j = n - 2; j >= 0; --j) {
    
		if (indPivot[j] != j)
			swapComplex (n, &A[STRIDE * j], 1, &A[STRIDE * indPivot[j]], 1);
	}
	
	return info;
}

//adapted from Matrix Computations
//Gene H. Golub, Charles F. Van Loan
static inline
int getHessenbergLU(const int n, const int STRIDE, double complex* A, int* indPivot)
{
	for (int i = 0; i < n - 1; i ++)
	{
		if (cabs(A[i * STRIDE + i]) < cabs(A[i * STRIDE + i + 1]))
		{
			indPivot[i] = i + 1;
			//swap rows
			swapComplex(n, &A[i], STRIDE, &A[i + 1], STRIDE);
		}
		else
		{
			indPivot[i] = i;
		}
		if (cabs(A[i * STRIDE + i]) > 0.0)
		{
			double complex tau = A[i * STRIDE + i + 1] / A[i * STRIDE + i];
			for (int j = i + 1; j < n; j++)
			{
				A[j * STRIDE + i + 1] -= tau * A[j * STRIDE + i];
			}
			A[i * STRIDE + i + 1] = tau;
		}
		else 
		{
			return i;
		}
	}
	return 0;
}

int getHessenbergLU_test(const int n, const int STRIDE, double complex* A, int* indPivot)
{
	return getHessenbergLU(n, STRIDE, A, indPivot);
}

void getComplexInverseHessenberg (const int n, const int STRIDE, double complex* A)
{
	// pivot indices
	int* ipiv = (int*) calloc (n, sizeof(int));
	
	// output flag
	int info = 0;
	
	// first get LU factorization
	getHessenbergLU (n, STRIDE, A, ipiv);

	if (info != 0)
	{
		printf ("getHessenbergLU failure, info = %d.\n", info);
		exit (1);
	}

	// work array
	double complex* work = (double complex*) calloc (n, sizeof(double complex));
  	// memset (work, 0.0, n * sizeof(double complex));
	
	// now get inverse
	getComplexInverseLU (n, STRIDE, A, ipiv, work);
	
	free (work);
	free (ipiv);
	
	// check for successful exit
	if (info != 0) {
		printf ("getComplexInverseLU failure, info = %d.\n", info);
		exit (1);
	}
}