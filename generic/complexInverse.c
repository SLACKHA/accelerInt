#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <complex.h>
#include <string.h>
#include "solver_props.h"
#include "lapack_dfns.h"

static inline
int getHessenbergLU(const int n, double complex* __restrict__ A,
						int* __restrict__ indPivot)
{
	int last_pivot = 0;
	for (int i = 0; i < n - 1; i ++)
	{
		if (cabs(A[i * STRIDE + i]) < cabs(A[i * STRIDE + i + 1]))
		{
			//swap rows
			for(int k = last_pivot; k < n; ++k)
			{
				double complex temp = A[k * STRIDE + i];
				A[k * STRIDE + i] = A[k * STRIDE + i + 1];
				A[k * STRIDE + i + 1] = temp;
			}
			indPivot[i] = i + 1;
		}
		else
		{
			indPivot[i] = i;
			last_pivot = i;
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
	//last index is not pivoted
	indPivot[n - 1] = n - 1;
	return 0;
}

void scaleComplex (const int n, const double complex val, double complex* __restrict__ arrX) {
    
    for (int i = 0; i < n; ++i) {
        arrX[i] *= val;
    }
    
}


//note: can't use __restrict__ here
void multiplyComplexUpperMV (const int n, double complex* x, const int lda,
								const double complex* A) {
    
    for (int j = 0; j < n; ++j) {
        if (cabs(x[j]) > 0.0) {
            double complex temp = x[j];
            for (int i = 0; i < j; ++i) {
                x[i] += temp * A[i + (lda * j)];
                //x[INDEX(i)] = cuCfma(temp, A[INDEX(i + (lda * j))], x[INDEX(i)]);
            }
            x[j] *= A[j + (lda * j)];
            //x[INDEX(j)] = cuCmul(x[INDEX(j)], A[INDEX(j + (lda * j))]);
        }
    }
    
}

///////////////////////////////////////////////////////////

void complexGEMV (const int m, const int n, const int lda, const double alpha, const double complex* __restrict__ A,
                                    const double complex* arrX, double complex* arrY) {
    
    // first: y = beta*y
    // beta = 1, so nothing
    
    // second: y = alpha*A*x + y
    
    for (int j = 0; j < n - 1; ++j) {

        if (cabs(arrX[j]) > 0.0) {
            double complex temp = alpha * arrX[j];
            for (int i = 0; i < m; ++i) {
                arrY[i] += temp * A[i + (lda * j)];
                //arrY[INDEX(i)] = cuCfma(temp, A[INDEX(i + (lda * j))], arrY[INDEX(i)]);
            }
        }
    }
}

int getComplexInverseHessenbergLU (const int n, double complex* __restrict__ A,
										const int* __restrict__ indPivot) {
    
	double complex work[STRIDE] = {0};

    // form inv(U)
    for (int j = 0; j < n; ++j) {
    	if (cabs(A[j + (STRIDE * j)]) == 0)
    		return j;
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
            work[i] = A[i + STRIDE * j];
            A[i + (STRIDE * j)] = 0;
        }
        
        // compute current column of inv(A)
        if (j < n - 1)
            complexGEMV (n, n - j, STRIDE, -1, &A[STRIDE * (j + 1)], &work[j + 1], &A[STRIDE * j]);
        
    }
    
    // apply column interchanges
    
    for (int j = n - 2; j >= 0; --j) {
    
        if (indPivot[j] != j)
        {
        	for (int i = 0; i < n; ++i) {
				double complex temp = A[STRIDE * j + i];
				A[STRIDE * j + i] = A[STRIDE * indPivot[j] + i];
				A[STRIDE * indPivot[j] + i] = A[STRIDE * j + i];
			}
        }
    }
    return 0;
}

void getComplexInverseHessenberg (const int n, double complex* __restrict__ A,
									int* __restrict__ ipiv, int* __restrict__ info)
{
	// first get LU factorization
	*info = getHessenbergLU (n, A, ipiv);

	if (*info != 0)
		return;

	// now get inverse
	*info = getComplexInverseHessenbergLU(n, A, ipiv);
}