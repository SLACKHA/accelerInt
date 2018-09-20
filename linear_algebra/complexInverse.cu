/**
 * \file
 * \brief Implementation of LU factorization of complex (variable-sized) matricies for CUDA
 *
 * Adapted from [Lapack](http://www.netlib.org/lapack/) LU factorization and inversion routines
 * \author Nick Curtis
 */


#include "header.cuh"
#include "solver_props.cuh"
#include <cuComplex.h>


/**
 *  \brief getComplexMax finds the index of the first element having maximum absolute value.
 *
 *  \param[in]          n               The size of complexArr
 *  \param[in]          complexArr      The (nx1) vector to determine the maximum value of
 */
__device__
int getComplexMax (const int n, const cuDoubleComplex * __restrict__ complexArr) {

    int maxInd = 0;
    if (n == 1)
        return maxInd;

    double maxVal = cuCabs(complexArr[INDEX(0)]);
    for (int i = 1; i < n; ++i) {
        if (cuCabs(complexArr[INDEX(i)]) > maxVal) {
            maxInd = i;
            maxVal = cuCabs(complexArr[INDEX(i)]);
        }
    }

    return maxInd;
}

///////////////////////////////////////////////////////////

/**
 * \brief scaleComplex scales a vector (with increment equal to one) by a constant val.
 * \param[in]       n           The vector size
 * \param[out]      val         The value to scale by
 * \param[out]      arrX        The vector to scale
 *
 */
__device__
void scaleComplex (const int n, const cuDoubleComplex val, cuDoubleComplex* __restrict__ arrX) {

    for (int i = 0; i < n; ++i) {
        arrX[INDEX(i)] = cuCmul(arrX[INDEX(i)], val);
    }

}

///////////////////////////////////////////////////////////

/*
__device__
void swapComplex (const int n, cuDoubleComplex* __restrict__ arrX, const int incX,
    cuDoubleComplex* __restrict__ arrY, const int incY) {

    int ix = 0;
    int iy = 0;

    for (int i = 0; i < n; ++i) {
        cuDoubleComplex temp = arrX[INDEX(ix)];
        arrX[INDEX(ix)] = arrY[INDEX(iy)];
        arrY[INDEX(iy)] = temp;
        ix += incX;
        iy += incY;
    }

}*/

///////////////////////////////////////////////////////////

/**
 * \brief complexGERU performs the rank 1 operation \f$A := alpha * arrX * arrY **T + A\f$
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
void complexGERU (const int n, const cuDoubleComplex alpha, const cuDoubleComplex* arrX,
                                    const cuDoubleComplex* arrY, const int incY, cuDoubleComplex* A, const int lda) {

    for (int j = 0; j < n; ++j) {
        if (cuCabs(arrY[INDEX(j * incY)]) > 0.0) {

            cuDoubleComplex temp = cuCmul(alpha, arrY[INDEX(j * incY)]);

            for (int i = 0; i < n; ++i) {
                A[INDEX(i + (lda * j))] = cuCfma(arrX[INDEX(i)], temp, A[INDEX(i + (lda * j))]);
            }

        }
    }

}

///////////////////////////////////////////////////////////
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
__device__
void multiplyComplexUpperMV (const int n, cuDoubleComplex* x, const int lda, const cuDoubleComplex* A) {

    for (int j = 0; j < n; ++j) {
        if (cuCabs(x[INDEX(j)]) > 0.0) {
            cuDoubleComplex temp = x[INDEX(j)];
            for (int i = 0; i < j; ++i) {
                //x[i] += temp * A[i + (lda * j)];
                x[INDEX(i)] = cuCfma(temp, A[INDEX(i + (lda * j))], x[INDEX(i)]);
            }
            //x[j] *= A[j + (lda * j)];
            x[INDEX(j)] = cuCmul(x[INDEX(j)], A[INDEX(j + (lda * j))]);
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
__device__
void complexGEMV (const int m, const int n, const int lda, const cuDoubleComplex alpha, const cuDoubleComplex* A,
                                    const cuDoubleComplex* arrX, cuDoubleComplex* arrY) {

    // first: y = beta*y
    // beta = 1, so nothing

    // second: y = alpha*A*x + y

    for (int j = 0; j < n - 1; ++j) {

        if (cuCabs(arrX[INDEX(j)]) > 0.0) {
            cuDoubleComplex temp = cuCmul(alpha, arrX[INDEX(j)]);
            for (int i = 0; i < m; ++i) {
                //arrY[i] += temp * A[i + (m * j)];
                arrY[INDEX(i)] = cuCfma(temp, A[INDEX(i + (lda * j))], arrY[INDEX(i)]);
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
void getComplexLU (const int n, cuDoubleComplex* __restrict__ A,
                    int* __restrict__ indPivot, int* __restrict__ info) {

    for (int j = 0; j < n; ++j) {

        // find pivot and test for singularity

        int jp = j + getComplexMax (n - j, &A[GRID_DIM * (j + (STRIDE * j))]);
        indPivot[INDEX(j)] = jp;

        if (cuCabs(A[INDEX(jp + (STRIDE * j))]) > 0.0) {

            // apply interchange to columns 1:n-1
            if (jp != j)
            {
                for (int i = 0; i < n; ++i) {
                    cuDoubleComplex temp = A[INDEX(STRIDE * i + j)];
                    A[INDEX(STRIDE * i + j)] = A[INDEX(STRIDE * i + jp)];
                    A[INDEX(STRIDE * i + jp)] = temp;
                }
                //swapComplex (n, &A[GRID_DIM * (j)], STRIDE, &A[GRID_DIM * (jp)], STRIDE);
            }

            // compute elements j+1:m-1 of the jth column

            if (j < STRIDE - 1)
                scaleComplex (n - j - 1, cuCdiv(make_cuDoubleComplex(1.0, 0.0), A[INDEX(j + (STRIDE * j))]), &A[GRID_DIM * (j + 1 + (STRIDE * j))]);

        } else if (*info == 0) {
            *info = j;
            break;
        }

        // update trailing submatrix
        if (j < n - 1)
            complexGERU (n - j - 1, make_cuDoubleComplex(-1.0, 0.0), &A[GRID_DIM * (j + 1 + (STRIDE * j))], &A[GRID_DIM * (j + STRIDE * (j + 1))], STRIDE, &A[GRID_DIM * (j + 1 + STRIDE * (j + 1))], STRIDE);

    }
}

/**  \brief getComplexInverseLU computes the inverse of a matrix using the LU factorization
     computed by getHessenbergLU or getComplexLU.

     This method inverts U and then computes inv(A) by solving the system
     inv(A)*L = inv(U) for inv(A).
 *
 *  \param[in]          n           The order of the matrix A.  n >= 0.
 *  \param[in,out]      A           The array, dimension (STRIDE, n) @see STRIDE
 *  \param[in]          indPivot    indPivot is an array of dimension (n).
                                    The pivot indices from getHessenbergLU; for 0<=i<=n-1, row i of the
                                    matrix was interchanged with row indPiv[i].
 *  \param[out]         work        A work array that will be overwritten
 */
__device__
int getComplexInverseLU (const int n, cuDoubleComplex* __restrict__ A,
                            const int* __restrict__ indPivot,
                            cuDoubleComplex* __restrict__ work) {

    // form inv(U)
    for (int j = 0; j < n; ++j) {
        if (cuCabs(A[INDEX(j + (STRIDE * j))]) == 0)
            return j;
        A[INDEX(j + (STRIDE * j))] = cuCdiv(make_cuDoubleComplex(1.0, 0.0), A[INDEX(j + (STRIDE * j))]);
        cuDoubleComplex Ajj = cuCmul(make_cuDoubleComplex(-1.0, 0.0), A[INDEX(j + (STRIDE * j))]);

        // compute elements 0:j-1 of jth column
        multiplyComplexUpperMV (j, &A[GRID_DIM * (STRIDE * j)], STRIDE, A);

        // scale
        scaleComplex (j, Ajj, &A[GRID_DIM * (STRIDE * j)]);
    }

    // solve equation inv(A)*L = inv(U) for inv(A)

    for (int j = n - 1; j >= 0; --j) {

        // copy current column of L to work and replace with 0.0s
        for (int i = j + 1; i < n; ++i) {
            work[INDEX(i)] = A[INDEX(i + (STRIDE * j))];
            A[INDEX(i + (STRIDE * j))] = make_cuDoubleComplex(0.0, 0.0);
        }

        // compute current column of inv(A)
        if (j < n - 1)
            complexGEMV (n, n - j, STRIDE, make_cuDoubleComplex(-1.0, 0.0), &A[GRID_DIM * (STRIDE * (j + 1))], &work[GRID_DIM * (j + 1)], &A[GRID_DIM * (STRIDE * j)]);

    }

    // apply column interchanges

    for (int j = n - 2; j >= 0; --j) {

        int jp = indPivot[INDEX(j)];
        if (jp != j)
        {
            for (int i = 0; i < n; ++i) {
                cuDoubleComplex temp = A[INDEX(STRIDE * j + i)];
                A[INDEX(STRIDE * j + i)] = A[INDEX(STRIDE * jp + i)];
                A[INDEX(STRIDE * jp + i)] = temp;
            }
        }
    }
    return 0;
}

/** \brief getComplexInverse computes the inverse of an a general matrix A using a LU factorization method
 *
 *  \param[in]          n           The order of the matrix A.  n >= 0.
 *  \param[in,out]      A           The array, dimension (STRIDE, n) @see STRIDE
 *  \param[out]         ipiv        ipiv is an array of dimension (n).
                                    The pivot indices from getHessenbergLU; for 0<=i<=n-1, row i of the
                                    matrix was interchanged with row indPiv[i].
 *  \param[out]         info        If not zero, an error occured during facotrization
 *  \param[out]         work        A work array that will be overwritten
 */
__device__
void getComplexInverse (const int n, cuDoubleComplex* __restrict__ A,
                            int* __restrict__ ipiv, int* __restrict__ info,
                            cuDoubleComplex* __restrict__ work) {

    // first get LU factorization
    getComplexLU (n, A, ipiv, info);

    // check for successful exit
    if (*info != 0) {
        return;
    }

    // now get inverse
    *info = getComplexInverseLU (n, A, ipiv, work);
}

/**
 * \brief Computes the LU factorization of a (n x STRIDE) Hessenberg Matrix using partial pivoting with row interchanges.
          @see STRIDE
 * \param[in]       n           The matrix size
 * \param[in,out]   A           The matrix to factorize (nxn) with stride defined in solver_props.h @see STRIDE
 * \param[out]      indPivot    indPivot is an array of dimension (n).
                                The pivot indices from getHessenbergLU; for 0<=i<=n-1, row i of the
                                matrix was interchanged with row indPiv[i].
 *  \param[out]     info        If not zero, an error occured during factorization
 *
 * The factorization has the form:
        \f$A = P * L * U\f$
 *  where P is a permutation matrix, L is lower triangular with unit
 *  diagonal elements (lower trapezoidal if m > n), and U is upper
 *  triangular (upper trapezoidal if m < n).
 * For full reference see:
 * G. W. Stewart, Matrix Algorithms: Volume 1: Basic Decompositions, SIAM, Philadelphia, 1998. doi:10.1137/1.9781611971408.
 */
__device__
void getHessenbergLU(const int n, cuDoubleComplex* A, int* __restrict__ indPivot, int* __restrict__ info)
{
    int last_pivot = 0;
    for (int i = 0; i < n - 1; i ++)
    {
        if (cuCabs(A[INDEX(i * STRIDE + i)]) < cuCabs(A[INDEX(i * STRIDE + i + 1)]))
        {
            //swap rows
            for(int k = 0; k < n; ++k)
            {
                if (k >= last_pivot)
                {
                    cuDoubleComplex temp = A[INDEX(k * STRIDE + i)];
                    A[INDEX(k * STRIDE + i)] = A[INDEX(k * STRIDE + i + 1)];
                    A[INDEX(k * STRIDE + i + 1)] = temp;
                }
            }
            indPivot[INDEX(i)] = i + 1;
        }
        else
        {
            indPivot[INDEX(i)] = i;
            last_pivot = i;
        }
        if (cuCabs(A[INDEX(i * STRIDE + i)]) > 0.0)
        {
            cuDoubleComplex tau = cuCdiv(A[INDEX(i * STRIDE + i + 1)], A[INDEX(i * STRIDE + i)]);
            for (int j = i + 1; j < n; j++)
            {
                A[INDEX(j * STRIDE + i + 1)] = cuCsub(A[INDEX(j * STRIDE + i + 1)], cuCmul(tau, A[INDEX(j * STRIDE + i)]));
            }
            A[INDEX(i * STRIDE + i + 1)] = tau;
        }
        else
        {
            *info = i;
            return;
        }
    }
    //last index is not pivoted
    indPivot[INDEX(n - 1)] = n - 1;
}

/** \brief getComplexInverseHessenberg computes the inverse of an upper Hessenberg matrix A using a LU factorization method
 *
 *  \param[in]          n           The order of the matrix A.  n >= 0.
 *  \param[in,out]      A           The array, dimension (STRIDE, n) @see STRIDE
 *  \param[out]         ipiv        ipiv is an array of dimension (n).
                                    The pivot indices from getHessenbergLU; for 0<=i<=n-1, row i of the
                                    matrix was interchanged with row indPiv[i].
 *  \param[out]         info        If not zero, an error occured during factorization
 *  \param[out]         work        A work array that will be overwritten
 */
__device__
void getComplexInverseHessenberg (const int n, cuDoubleComplex* __restrict__ A,
                                    int* __restrict__ ipiv, int* __restrict__ info,
                                    cuDoubleComplex* __restrict__ work)
{
    // first get LU factorization
    getHessenbergLU (n, A, ipiv, info);

    if (*info != 0)
        return;

    // now get inverse
    *info = getComplexInverseLU (n, A, ipiv, work);
}