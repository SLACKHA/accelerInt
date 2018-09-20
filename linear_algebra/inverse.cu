/**
 * \file
 * \brief CUDA LU decomposition implementation
 */

#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <string.h>

#include "header.cuh"
#include "solver_props.cuh"

///////////////////////////////////////////////////////////

/**
 *  \brief getMax finds the index of the first element having maximum absolute value.
 *
 *  \param[in]          n               The size of Arr
 *  \param[in]          Arr      		The (nx1) vector to determine the maximum value of
 */
__device__
int getMax (const int n, const double * __restrict__ Arr) {

	int maxInd = 0;
	if (n == 1)
		return maxInd;

	double maxVal = fabs(Arr[INDEX(0)]);
	for (int i = 1; i < n; ++i) {
		if (fabs(Arr[INDEX(i)]) > maxVal) {
			maxInd = i;
			maxVal = fabs(Arr[INDEX(i)]);
		}
	}

	return maxInd;
}

///////////////////////////////////////////////////////////

/**
 * \brief scale multiplies a vector (with increment equal to one) by a constant val.
 * \param[in]       n           The vector size
 * \param[out]      val         The value to scale by
 * \param[out]      arrX        The vector to scale
 *
 */
__device__
void scale (const int n, const double val, double* __restrict__ arrX) {

	for (int i = 0; i < n; ++i) {
		arrX[INDEX(i)] *= val;
	}

}

///////////////////////////////////////////////////////////

/**
 * \brief interchanges two vectors arrX and arrY.
 *
 * \param[in]			n			the vector size
 * \param[in]			arrX		the first vector to swap
 * \param[in]			incX		the increment of the arrX vector
 * \param[in]			arrY		the second vector to swap
 * \param[in]			incY		the increment of the arrY vector
 */
__device__
void swap (const int n, double* __restrict__ arrX, const int incX, double* __restrict__ arrY, const int incY) {

	int ix = 0;
	int iy = 0;

	for (int i = 0; i < n; ++i) {
		double temp = arrX[INDEX(ix)];
		arrX[INDEX(ix)] = arrY[INDEX(iy)];
		arrY[INDEX(iy)] = temp;
		ix += incX;
		iy += incY;
	}

}

///////////////////////////////////////////////////////////

/**
 * \brief GERU performs the rank 1 operation \f$A := alpha * arrX * arrY **T + A\f$
          where alpha is a scalar, arrX and arrY are n element vectors, and A is a (lda x n) matrix
 *
 * \param[in]           n           The matrix/vector size
 * \param[in]           alpha       The value to scale by
 * \param[in]           arrX        arrX is an array of dimension at least n.
                                    Before entry, the incremented array arrX must contain the n
                                    element vector x.
 * \param[in]           arrY        arrY is an array of dimension at least 1 + (n - 1) * incY.
                                    Before entry, the incremented array arrY must contain the n
                                    element vector y.
 * \param[in]           incY        On entry, INCY specifies the increment for the elements of arrY. incY must not be zero.
 * \param[out]          A           A is an array of dimension (lda x n).
                                    Before entry, the leading n by n part of the array A must
                                    contain the matrix of coefficients. On exit, A is
                                    overwritten by the updated matrix.
 * \param[in]           lda         On entry, lda specifies the first dimension of A as declared
                                    in the calling (sub) program. lda must be at least
                                    max( 1, n ).
 */
__device__
void GERU (const int n, const double alpha, const double* __restrict__ arrX,
				const double* __restrict__ arrY, const int incY, double* __restrict__ A, const int lda) {

	for (int j = 0; j < n; ++j) {
    	if (fabs(arrY[INDEX(j * incY)]) > 0.0) {

			double temp = alpha * arrY[INDEX(j * incY)];

			for (int i = 0; i < n; ++i) {
				A[INDEX(i + (lda * j))] += arrX[INDEX(i)] * temp;
			}

		}
	}

}

///////////////////////////////////////////////////////////

/**
 * \brief Computes the LU factorization of a (n x n) matrix using partial pivoting with row interchanges.
          @see STRIDE
 * \param[in]       n           The matrix size
 * \param[in,out]   A           The matrix to factorize (n x n) with stride defined in solver_props.h @see STRIDE
 * \param[out]      indPivot    indPivot is an array of dimension (n).
                                The pivot indices from getHessenbergLU; for 0<=i<=n-1, row i of the
                                matrix was interchanged with row indPiv[i].
 & \param[out]      info        An information variable
 *
 * The factorization has the form:
        \f$A = P * L * U\f$
 *  where P is a permutation matrix, L is lower triangular with unit
 *  diagonal elements (lower trapezoidal if m > n), and U is upper
 *  triangular (upper trapezoidal if m < n).
 */
__device__
void getLU (const int n, double* __restrict__ A, int* __restrict__ indPivot, int* __restrict__ info) {

	for (int j = 0; j < n; ++j) {

		// find pivot and test for singularity

		int jp = j + getMax (n - j, &A[GRID_DIM * (j + (STRIDE * j))]);
		indPivot[INDEX(j)] = jp;

    	if (fabs(A[INDEX(jp + (STRIDE * j))]) > 0.0) {

			// apply interchange to columns 1:n-1
			if (jp != j)
				swap(n, &A[GRID_DIM * (j)], STRIDE, &A[GRID_DIM * (jp)], STRIDE);

			// compute elements j+1:m-1 of the jth column

			if (j < n - 1)
				scale(n - j - 1, 1.0 / A[INDEX(j + (STRIDE * j))], &A[GRID_DIM * (j + 1 + (STRIDE * j))]);

		} else if (*info == 0) {
			*info = j;
			break;
		}

		// update trailing submatrix
		if (j < n - 1)
			GERU (n - j - 1, -1.0, &A[GRID_DIM * (j + 1 + (STRIDE * j))], &A[GRID_DIM * (j + STRIDE * (j + 1))], STRIDE, &A[GRID_DIM * (j + 1 + STRIDE * (j + 1))], STRIDE);
	}
}