/**
 * \file
 * \brief Various linear algebra routines needed for the Carathéodory-Fejér method
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include <complex.h>

////////////////////////////////////////////////////////////////////////////////////////

/** \brief Interface function to LAPACK matrix inversion subroutine.
 *
 * Performs inversion of square matrix. Uses LAPACK subroutines DGETRF and DGETRI.
 *
 * \param[in]     n     order of matrix
 * \param[in]     A     the input matrix, size n*n
 * \return        info  success/fail integer flag
 */
void getInverseComplex (int n, double complex* A) {

  // Prototype for LAPACK ZGETRF (Fortran) function
  extern void zgetrf_ (int *m, int *n, double complex* A, int* lda, int* ipiv, int* info);

  // Prototype for LAPACK ZGETRI (Fortran) function
  extern void zgetri_ (int* n, double complex* A, int* lda, int* ipiv, double complex* work,
                        int* lwork, int* info);

	// pivot indices
	int* ipiv = (int*) calloc (n, sizeof(int));

	// output flag
	int info;

	// first call zgetrf for LU factorization
	zgetrf_ (&n, &n, A, &n, ipiv, &info);

	// check for successful exit
	if (info < 0) {
		printf ("ZGETRF argument %d had illegal argument.\n", info);
		exit (1);
	} else if (info > 0) {
		printf ("ZGETRF failure, U is singular\n.");
		exit (1);
	}

	// work array
	double complex* work = (double complex*) calloc (1, sizeof(double complex));
	int lwork = -1;

	// call zgetri for work array size
	zgetri_ (&n, A, &n, ipiv, work, &lwork, &info);

	// increase size of work array
	lwork = (int) creal(work[0]);
	work = (double complex*) realloc (work, lwork * sizeof(double complex));

	// now call zgetri for inversion
	zgetri_ (&n, A, &n, ipiv, work, &lwork, &info);

	// check for successful exit
	if (info < 0) {
		printf ("ZGETRI argument %d had illegal argument.\n", info);
		exit (1);
	} else if (info > 0) {
		printf ("ZGETRI failure, matrix is singular\n.");
		exit (1);
	}

	free (ipiv);
	free (work);
}

///////////////////////////////////////////////////////////////////////////////////////////

/** \brief Solves the complex linear system Ax = B
 *
 * Performs inversion of square matrix. Uses LAPACK subroutines DGETRF and DGETRI.
 *
 * \param[in]     n     order of matrix
 * \param[in]     A     the LHS matrix, size n*n
 * \param[in]     B     the RHS matrix, size n*n
 * \param[out]    x     the solved vector, size n*1
 */
void linSolveComplex (int n, double complex* A, double complex* B, double complex* x) {

	extern void zcgesv_ (int* n, int* nrhs, double complex* A, int* lda, int* ipiv,
											double complex* B, int* ldb, double complex* x, int* ldx,
											double complex* work, float complex* swork, double* rwork,
											int* iter, int* info);

	// number of right-hand sides: 1
	int nrhs = 1;

	// double complex work array
	double complex* work = (double complex*) calloc (n * nrhs, sizeof(double complex));

	// single complex work array
	float complex* swork = (float complex*) calloc (n * (n + nrhs), sizeof(float complex));

	// double real work array
	double* rwork = (double*) calloc (n, sizeof(double));

	// output flags
	int ipiv, iter, info;

	zcgesv_ (&n, &nrhs, A, &n, &ipiv, B, &n, x, &n, work, swork, rwork, &iter, &info);

	// check for successful exit
	if (info < 0) {
		printf ("ZCGESV argument %d illegal argument.\n", info);
		exit (1);
	} else if (info > 0) {
		printf ("ZCGESV failure, U is singular\n.");
		exit (1);
	}

	free (work);
	free (swork);
	free (rwork);

}

///////////////////////////////////////////////////////////////////////////////////////

/** \brief Polynomial root finding function.
 *
 * This function calculates the roots of a polynomial represented by its
 * coefficients in the array v. This is done by constructing a companion
 * matrix out of the polynomial coefficients, then using the LAPACK
 * subroutine DGEEV to calculate its eigenvalues. These are the roots
 * of the polynomial.
 *
 * \param[in]		n   size of v;
 * \param[in]		v		array of polynomial coefficients (real)
 * \param[out]		rt	array of polynomial roots (complex), size n - 1
 */
void roots (int n, double* v, double complex* rt) {

	// number of roots
	int m = n - 1;

	// Prototype for LAPACK DGEEV (Fortran) function
	extern void dgeev_ (char* jobvl, char* jobvr, int* n, double* A, int* lda,
											double* wr, double* wi, double* vl, int* ldvl, double *vr,
											int* ldvr, double* work, int* lwork, int* info);

	// construct companion matrix
	double *A = (double*) calloc (m * m, sizeof(double));
	memset (A, 0.0, m * m * sizeof(double));		// fill with 0

	for (int i = 0; i < (m - 1); ++i) {
		A[i + 1 + m * i] = 1.0;
	}

	for (int i = 0; i < m; ++i) {
		A[m * i] = -v[i + 1] / v[0];
	}

	int info, lwork;
	double* vl;
	double* vr;
	double wkopt;
	double* work;

	// don't need eigenvectors
	char job = 'N';

	// real and imaginary parts of eigenvalues
	double* wr = (double*) calloc (m, sizeof(double));
	double* wi = (double*) calloc (m, sizeof(double));

	// first compute optimal workspace
	lwork = -1;
	dgeev_ (&job, &job, &m, A, &m, wr, wi, vl, &m, vr, &m, &wkopt, &lwork, &info);
	lwork = (int) wkopt;
	work = (double*) calloc (lwork, sizeof(double));

	// compute eigenvalues
	dgeev_ (&job, &job, &m, A, &m, wr, wi, vl, &m, vr, &m, work, &lwork, &info);

	// check for convergence
	if (info > 0) {
		printf ("DGEEV failed to compute the eigenvalues.\n");
		exit (1);
	}

	for (int i = 0; i < m; ++i) {
		rt[i] = wr[i] + I * wi[i];
	}

	free (A);
	free (work);
	free (wr);
	free (wi);
}

///////////////////////////////////////////////////////////////////////////////////////

/** \brief Singular value decomposition function.
 *
 * Decomposes a matrix A into U * S * V', where S (here an array, really
 * a diagonal matrix) holds the singular values. The function uses the
 * LAPACK subroutine DGESVD.
 *
 * \param[in]		n		leading dimension of array
 * \param[in]		A		array to be decomposed, size n * n
 * \param[out]		S		array with singular values, size n
 * \param[out]		U		array with left singular vectors, size n * n
 * \param[out]		V		array with right singular vectors, size n * n
 */
void svd (int n, double* A, double* S, double* U, double* V) {

	// Prototype for LAPACK DGESVD (Fortran) function
	extern void dgesvd_ (char* jobu, char* jobvt, int* m, int* n, double* A,
	                		int* lda, double* s, double* u, int* ldu, double* vt,
											int* ldvt, double* work, int* lwork, int* info);

	int info, lwork;
	double wkopt;
	double* work;
	double* Vt = (double*) calloc (n * n, sizeof(double));

	// want all of U and V
	char job = 'A';

	// first compute optimal workspace
	lwork = -1;
	dgesvd_ (&job, &job, &n, &n, A, &n, S, U, &n, Vt, &n, &wkopt, &lwork, &info);
	lwork = (int) wkopt;
	work = (double*) calloc (lwork, sizeof(double));

  // Compute SVD
	dgesvd_ (&job, &job, &n, &n, A, &n, S, U, &n, Vt, &n, work, &lwork, &info);

	// check for convergence
	if (info > 0) {
		printf ("DGESVD failed.\n");
		exit (1);
	}

	// transpose Vt
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			V[i + j*n] = Vt[j + i*n];
		}
	}

	free (Vt);
	free (work);
}
