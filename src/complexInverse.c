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
void complexGERU (int n, double complex alpha, double complex* arrX,
									double complex* arrY, int incY, double complex* A, int lda) {
	
	for (int j = 0; j < n; ++j) {
		//if (arrY[j * incY] != 0.0) {
    
    if (cabs(arrY[j * incY]) > 0.0) {      
			double complex temp = alpha * arrY[j * incY];
			for (int i = 0; i < n; ++i) {
				A[i + (lda * j)] += arrX[i] * temp;
			}
		}
    
	}
	
}

///////////////////////////////////////////////////////////

static inline
int getComplexLU (int n, double complex* A, int* indPivot) {
	
	int info = 0;
	
	for (int j = 0; j < n; ++j) {
		
		// find pivot and test for singularity
		
		int jp = j + getComplexMax (n - j, &A[j + (n * j)]);
		indPivot[j] = jp;
		
		//if (A[jp + (n * j)] != 0.0) {
    if (cabs(A[jp + (n * j)]) > 0.0) {
			
			// apply interchange to columns 1:n-1
			if (jp != j)
				swapComplex (n, &A[j], n, &A[jp], n);
			
			// compute elements j+1:m-1 of the jth column
			
			if (j < n - 1)
				scaleComplex (n - j - 1, 1.0 / A[j + (n * j)], &A[j + 1 + (n * j)]);
			
		} else if (info == 0) {
			info = j + 1;
		}
		
		// update trailing submatrix
		if (j < n - 1)
			complexGERU (n - j - 1, -1.0, &A[j + 1 + (n * j)], &A[j + n * (j + 1)], n, &A[j + 1 + n * (j + 1)], n);
		
	}
	
	return info;
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
void complexGEMV (int m, int n, double complex alpha, double complex* A, 
									double complex* arrX, double complex* arrY) {
	
	// first: y = beta*y
	// beta = 1, so nothing
	
	// second: y = alpha*A*x + y
	
	for (int j = 0; j < n - 1; ++j) {
		//if (arrX[j] != 0.0) {
    if (cabs(arrX[j]) > 0.0) {
			double complex temp = alpha * arrX[j];
			for (int i = 0; i < m; ++i) {
				arrY[i] += temp * A[i + (m * j)];
			}
		}
	}
	
}

///////////////////////////////////////////////////////////

static inline
int getComplexInverseLU (int n, double complex* A, int* indPivot, double complex* work) {
	
	int info = 0;
	
	// form inv(U)
	for (int j = 0; j < n; ++j) {
		A[j + (n * j)] = 1.0 / A[j + (n * j)];
		double complex Ajj = -A[j + (n * j)];
		
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
			A[i + (n * j)] = 0.0;
		}
		
		// compute current column of inv(A)
		if (j < n - 1)
			complexGEMV (n, n - j, -1.0, &A[n * (j + 1)], &work[j + 1], &A[n * j]);
		
	}
	
	// apply column interchanges
	
	for (int j = n - 2; j >= 0; --j) {
    
		if (indPivot[j] != j)
			swapComplex (n, &A[n * j], 1, &A[n * indPivot[j]], 1);
	}
	
	return info;
}

///////////////////////////////////////////////////////////

void getComplexInverse (int n, double complex* A) {
	
	// pivot indices
	int* ipiv = (int*) calloc (n, sizeof(int));
	
	// output flag
	int info = 0;
	
	// first get LU factorization
	info = getComplexLU (n, A, ipiv);
	
	// check for successful exit
	if (info != 0) {
		printf ("getComplexLU failure, info = %d.\n", info);
		exit (1);
	}
	
	// work array
	double complex* work = (double complex*) calloc (n, sizeof(double complex));
  // memset (work, 0.0, n * sizeof(double complex));
	
	// now get inverse
	info = getComplexInverseLU (n, A, ipiv, work);
	
	free (work);
	free (ipiv);
	
	// check for successful exit
	if (info != 0) {
		printf ("getComplexInverseLU failure, info = %d.\n", info);
		exit (1);
	}
}

//adapted from Matrix Computations
//Gene H. Golub, Charles F. Van Loan
static inline
int getHessenbergLU(const int n, double complex* A, int* indPivot)
{
	for (int i = 0; i < n - 1; i ++)
	{
		if (cabs(A[i * n + i]) < cabs(A[i * n + i + 1]))
		{
			//swap rows
			swapComplex(n, &A[i], n, &A[i + 1], n);
			indPivot[i] = i + 1;
		}
		else
		{
			indPivot[i] = i;
		}
		if (cabs(A[i * n + i]) > 0.0)
		{
			double complex tau = A[i * n + i + 1] / A[i * n + i];
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

//#ifdef COMPILE_TESTING_METHODS
	int getComplexLU_test(const int n, double complex* A, int* indPivot)
	{
		return getComplexLU(n, A, indPivot);
	}

	int getHessenbergLU_test(const int n, double complex* A, int* indPivot)
	{
		return getHessenbergLU(n, A, indPivot);
	}
//#endif

void getComplexInverseHessenberg (const int n, double complex* A)
{
	// pivot indices
	int* ipiv = (int*) calloc (n, sizeof(int));
	
	// output flag
	int info = 0;
	
	// first get LU factorization
	info = getHessenbergLU (n, A, ipiv);

	if (info != 0)
	{
		printf ("getHessenbergLU failure, info = %d.\n", info);
		exit (1);
	}

	// work array
	double complex* work = (double complex*) calloc (n, sizeof(double complex));
  	// memset (work, 0.0, n * sizeof(double complex));
	
	// now get inverse
	info = getComplexInverseLU (n, A, ipiv, work);
	
	free (work);
	free (ipiv);
	
	// check for successful exit
	if (info != 0) {
		printf ("getComplexInverseLU failure, info = %d.\n", info);
		exit (1);
	}
}