#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <complex.h>
#include <string.h>
#include "lapack_dfns.h"

void swapComplex (const int n, double complex* __restrict__ arrX, const int incX,
					double complex* __restrict__ arrY, const int incY) {
	
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

//Matrix Algorithms: Volume 1: Basic Decompositions
//By G. W. Stewart
static inline
void getHessenbergLU(const int n, const int LDA, double complex* __restrict__ A,
						int* __restrict__ indPivot, int* __restrict__ info)
{
	int last_free = 0;
	for (int i = 0; i < n - 1; i ++)
	{
		if (cabs(A[i * LDA + i]) < cabs(A[i * LDA + i + 1]))
		{
			//swap rows
			swapComplex(n - last_free, &A[last_free * LDA + i], LDA, &A[last_free * LDA + i + 1], LDA);
			indPivot[i] = i + 1;
		}
		else
		{
			indPivot[i] = i;
			last_free = i;
		}
		if (cabs(A[i * LDA + i]) > 0.0)
		{
			double complex tau = A[i * LDA + i + 1] / A[i * LDA + i];
			for (int j = i + 1; j < n; j++)
			{
				A[j * LDA + i + 1] -= tau * A[j * LDA + i];
			}
			A[i * LDA + i + 1] = tau;
		}
		else 
		{
			*info = i;
			return;
		}
	}
	//last index is not pivoted
	indPivot[n - 1] = n - 1;
	*info = 0;
}

void getComplexInverseHessenberg (const int n, const int LDA, double complex* __restrict__ A,
									int* __restrict__ ipiv, int* __restrict__ info,
									double complex* __restrict__ work, const int work_size)
{
	// first get LU factorization
	getHessenbergLU (n, LDA, A, ipiv, info);

	if (*info != 0)
		return;

	// now get inverse
	zgetri_(&n, A, &LDA, ipiv, work, &work_size, info);

}