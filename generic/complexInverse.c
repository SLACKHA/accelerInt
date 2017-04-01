/**
 * \file
 * \brief Implementation of LU factorization of complex (variable-sized) matricies
 *
 * Adapted from [Lapack](http://www.netlib.org/lapack/) LU factorization and inversion routines
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <complex.h>
#include <string.h>
#include "solver_props.h"
#include "lapack_dfns.h"

/**
 * \brief Computes the LU factorization of a (n x STRIDE) Hessenberg Matrix using partial pivoting with row interchanges.
          @see STRIDE
 * \param[in]       n           The matrix size
 * \param[in,out]   A           The matrix to factorize (nxn) with stride defined in solver_props.h @see STRIDE
 * \param[out]      indPivot    indPivot is an array of dimension (n).
                                The pivot indices from getHessenbergLU; for 0<=i<=n-1, row i of the
                                matrix was interchanged with row indPiv[i].
 *
 * The factorization has the form:
        \f$A = P * L * U\f$
 *  where P is a permutation matrix, L is lower triangular with unit
 *  diagonal elements (lower trapezoidal if m > n), and U is upper
 *  triangular (upper trapezoidal if m < n).
 * For full reference see:
 * G. W. Stewart, Matrix Algorithms: Volume 1: Basic Decompositions, SIAM, Philadelphia, 1998. doi:10.1137/1.9781611971408.
 */
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

/**
 * \brief scaleComplex scales a vector (with increment equal to one) by a constant val.
 * \param[in]       n           The vector size
 * \param[out]      val         The value to scale by
 * \param[out]      arrX        The vector to scale
 *
 */
void scaleComplex (const int n, const double complex val, double complex* __restrict__ arrX) {

    for (int i = 0; i < n; ++i) {
        arrX[i] *= val;
    }

}

/**
 * \brief Performs the matrix-vector operation \f$x_v:= A*x_v\f$
 * \param[in]       n           On entry, n specifies the order of the matrix A.
                                n must be at least zero.
 * \param[out]      x           x is an array of dimension at least (n).
                                Before entry, the incremented array X must contain the n
                                element vector \f$x_v\f$. On exit, X is overwritten with the
                                transformed vector \f$x_v\f$.
 * \param[in]       lda         The stride of the matrix @see STRIDE
 * \param[in]       A           A is an array of dimension (lda, n).
                                Before entry the leading n by n
                                upper triangular part of the array A must contain the upper
                                triangular matrix and the strictly lower triangular part of
                                A is not referenced.
 *
 * Note: These pointers can't use the \_\_restrict\_\_ attribute, as they may overlap
 */
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

/**
 * \brief Computes the matrix-vector operation \f$alpha*A*x + y\f$ where alpha is a scalar,
          x and y are vectors and A is an m by n matrix.
 * \param[in]       m           On entry, M specifies the number of rows of the matrix A. Must be >= 0
 * \param[out]      n           On entry, N specifies the number of columns of the matrix A. Must be >= 0
 * \param[in]       lda         The stride of the matrix @see STRIDE
 * \param[in]       alpha       The scalar value
 * \param[in]       A           A is an array of dimension (lda, n).
                                Before entry, the leading m by n part of the array A must
                                contain the matrix of coefficients.
 * \param[in]       arrX        arrX is an array of dimension at least (n)
                                Before entry, the incremented array arrX must contain the
                                vector x.
 * \param[in,out]   arrY        arrY is an array of dimension at least (m).
                                Before entry, the incremented array arrY must contain the vector y.
                                On exit, arrY is overwritten by the updated vector y.
 *
 * Note: These pointers cannot use the \_\_restrict\_\_ modifier, as they may overlap
 */
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

/**  \brief getComplexInverseHessenbergLU computes the inverse of a matrix using the LU factorization
     computed by getHessenbergLU.

     This method inverts U and then computes inv(A) by solving the system
     inv(A)*L = inv(U) for inv(A).
 *
 *  \param[in]          n           The order of the matrix A.  n >= 0.
 *  \param[in,out]      A           The array, dimension (STRIDE, n) @see STRIDE
 *  \param[in]          indPivot    indPivot is an array of dimension (n).
                                    The pivot indices from getHessenbergLU; for 0<=i<=n-1, row i of the
                                    matrix was interchanged with row indPiv[i].
 */
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
				A[STRIDE * indPivot[j] + i] = temp;
			}
        }
    }
    return 0;
}

/** \brief getComplexInverseHessenberg computes the inverse of an upper Hessenberg matrix A using a LU factorization method
 *
 *  \param[in]          n           The order of the matrix A.  n >= 0.
 *  \param[in,out]      A           The array, dimension (STRIDE, n) @see STRIDE
 *  \param[out]         ipiv        ipiv is an array of dimension (n).
                                    The pivot indices from getHessenbergLU; for 0<=i<=n-1, row i of the
                                    matrix was interchanged with row indPiv[i].
 *  \param[out]         info        If not zero, an error occured during factorization
 */
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