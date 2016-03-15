#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <complex.h>
#include <string.h>
#include "solver_props.h"
#include "lapack_dfns.h"

static int ARRAYSIZE = STRIDE;

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
void getHessenbergLU(const int n, double complex* __restrict__ A,
						int* __restrict__ indPivot, int* __restrict__ info)
{
	int last_free = 0;
	for (int i = 0; i < n - 1; i ++)
	{
		if (cabs(A[i * STRIDE + i]) < cabs(A[i * STRIDE + i + 1]))
		{
			//swap rows
			swapComplex(n - last_free, &A[last_free * STRIDE + i], STRIDE, &A[last_free * STRIDE + i + 1], STRIDE);
			indPivot[i] = i + 1;
		}
		else
		{
			indPivot[i] = i;
			last_free = i;
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
			*info = i;
			return;
		}
	}
	//last index is not pivoted
	indPivot[n - 1] = n - 1;
	*info = 0;
}

void getComplexInverseHessenberg (const int n, double complex* __restrict__ A,
									int* __restrict__ ipiv, int* __restrict__ info,
									double complex* __restrict__ work, const int work_size)
{
	// first get LU factorization
	getHessenbergLU (n, A, ipiv, info);

	if (*info != 0)
		return;

	// now get inverse
	zgetri_(&n, A, &ARRAYSIZE, ipiv, work, &work_size, info);

}