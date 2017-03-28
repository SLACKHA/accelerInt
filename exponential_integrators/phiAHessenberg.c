/**
 * \file phiAHessenberg.c
 * \brief Computes various matrix exponential functions on the Krylov Hessenberg matricies
 */

#include <stdlib.h>
#include <complex.h>

#include "header.h"
#include "lapack_dfns.h"
#include "complexInverse.h"
#include "solver_options.h"
#include "solver_props.h"

extern double complex poles[N_RA];
extern double complex res[N_RA];

/** \brief Compute the 2nd order Phi (exponential) matrix function
 *
 *  Computes \f$\phi_2(c*A)\f$
 *
 *  \param[in]		m		The Hessenberg matrix size (mxm)
 *  \param[in]		A		The input Hessenberg matrix
 *  \param[in]		c		The scaling factor
 *  \param[out]		phiA	The resulting exponential matrix
 */
int phi2Ac_variable(const int m, const double* A, const double c, double* phiA) {

	//allocate arrays
	int ipiv[STRIDE] = {0};
	double complex invA[STRIDE * STRIDE];
	int info = 0;

	for (int i = 0; i < m; ++i) {

		for (int j = 0; j < m; ++j) {
			phiA[i + j*STRIDE] = 0.0;
		}
	}


	for (int q = 0; q < N_RA; q += 2) {

		// init invA
		for (int i = 0; i < m; ++i) {
			for (int j = 0; j < m; ++j) {
				// A - theta * I
				if (i == j) {
					invA[i + j * STRIDE] = c * A[i + j*STRIDE] - poles[q];
				} else {
					invA[i + j * STRIDE] = c * A[i + j*STRIDE];
				}
			}
		}

		// takes care of (A * c - poles(q) * I)^-1
		getComplexInverseHessenberg (m, invA, ipiv, &info);

		if (info != 0)
			return info;


		for (int i = 0; i < m; ++i) {

			for (int j = 0; j < m; ++j) {
				phiA[i + j*STRIDE] += 2.0 * creal((res[q] / (poles[q] * poles[q])) * invA[i + j * STRIDE]);
			}
		}
	}

	return 0;
}

/** \brief Compute the first order Phi (exponential) matrix function
 *
 *  Computes \f$\phi_1(c*A)\f$
 *
 *  \param[in]		m		The Hessenberg matrix size (mxm)
 *  \param[in]		A		The input Hessenberg matrix
 *  \param[in]		c		The scaling factor
 *  \param[out]		phiA	The resulting exponential matrix
 */
int phiAc_variable(const int m, const double* A, const double c, double* phiA) {

	//allocate arrays
	int ipiv[STRIDE] = {0};
	double complex invA[STRIDE * STRIDE];
	int info = 0;

	for (int i = 0; i < m; ++i) {

		for (int j = 0; j < m; ++j) {
			phiA[i + j*STRIDE] = 0.0;
		}
	}


	for (int q = 0; q < N_RA; q += 2) {

		// init invA
		for (int i = 0; i < m; ++i) {
			for (int j = 0; j < m; ++j) {
				// A - theta * I
				if (i == j) {
					invA[i + j * STRIDE] = c * A[i + j*STRIDE] - poles[q];
				} else {
					invA[i + j * STRIDE] = c * A[i + j*STRIDE];
				}
			}
		}

		// takes care of (A * c - poles(q) * I)^-1
		getComplexInverseHessenberg (m, invA, ipiv, &info);

		if (info != 0)
			return info;


		for (int i = 0; i < m; ++i) {

			for (int j = 0; j < m; ++j) {
				phiA[i + j*STRIDE] += 2.0 * creal((res[q] / poles[q]) * invA[i + j * STRIDE]);
			}
		}
	}

	return 0;
}

/** \brief Compute the zeroth order Phi (exponential) matrix function.
 *		   This is the regular matrix exponential
 *
 *  Computes \f$\phi_0(c*A)\f$
 *
 *  \param[in]		m		The Hessenberg matrix size (mxm)
 *  \param[in]		A		The input Hessenberg matrix
 *  \param[in]		c		The scaling factor
 *  \param[out]		phiA	The resulting exponential matrix
 */
int expAc_variable(const int m, const double* A, const double c, double* phiA) {

	//allocate arrays
	int ipiv[STRIDE] = {0};
	double complex invA[STRIDE * STRIDE];
	int info = 0;

	for (int i = 0; i < m; ++i) {

		for (int j = 0; j < m; ++j) {
			phiA[i + j*STRIDE] = 0.0;
		}
	}


	for (int q = 0; q < N_RA; q += 2) {

		// compute transpose and multiply with constant
		for (int i = 0; i < m; ++i) {
			for (int j = 0; j < m; ++j) {
				// A - theta * I
				if (i == j) {
					invA[i + j*STRIDE] = c * A[i + j*STRIDE] - poles[q];
				} else {
					invA[i + j*STRIDE] = c * A[i + j*STRIDE];
				}
			}
		}

		// takes care of (A * c - poles(q) * I)^-1
		getComplexInverseHessenberg (m, invA, ipiv, &info);

		if (info != 0)
			return info;


		for (int i = 0; i < m; ++i) {

			for (int j = 0; j < m; ++j) {
				phiA[i + j*STRIDE] += 2.0 * creal(res[q] * invA[i + j*STRIDE]);
			}
		}
	}

	return 0;
}