/**
 * \file phiAHessenberg.c
 * \brief Computes various matrix exponential functions on the Krylov Hessenberg matricies
 */

#include <cstdlib>
#include <complex>

#include "lapack_dfns.h"
#include "complexInverse.hpp"
#include "exp_solver.hpp"

namespace c_solvers
{

	/** \brief Compute the 2nd order Phi (exponential) matrix function
	 *
	 *  Computes \f$\phi_2(c*A)\f$
	 *
	 *  \param[in]		m		The Hessenberg matrix size (mxm)
	 *  \param[in]		A		The input Hessenberg matrix
	 *  \param[in]		c		The scaling factor
	 *  \param[out]		phiA	The resulting exponential matrix
	 */
	int ExponentialIntegrator::phi2Ac_variable(const int m, const double* A, const double c, double* phiA) {

		//allocate arrays
		int tid = omp_get_thread_num();
		int* ipiv = _unique<int>(tid, _ipiv);
		std::complex<double>* invA = _unique<std::complex<double>>(tid, _invA);
		std::complex<double>* work = _unique<std::complex<double>>(tid, _w);
		int info = 0;

		for (int i = 0; i < m; ++i) {

			for (int j = 0; j < m; ++j) {
				phiA[i + j*STRIDE] = 0.0;
			}
		}


		for (int q = 0; q < N_RA(); q += 2) {

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
			getComplexInverseHessenberg (m, STRIDE, invA, ipiv, &info, work);

			if (info != 0)
				return info;


			for (int i = 0; i < m; ++i) {

				for (int j = 0; j < m; ++j) {
					phiA[i + j*STRIDE] += 2.0 * std::real((res[q] / (poles[q] * poles[q])) * invA[i + j * STRIDE]);
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
	int ExponentialIntegrator::phiAc_variable(const int m, const double* A, const double c, double* phiA) {

		//allocate arrays
		int tid = omp_get_thread_num();
		int* ipiv = _unique<int>(tid, _ipiv);
		std::complex<double>* invA = _unique<std::complex<double>>(tid, _invA);
		std::complex<double>* work = _unique<std::complex<double>>(tid, _w);
		int info = 0;

		for (int i = 0; i < m; ++i) {

			for (int j = 0; j < m; ++j) {
				phiA[i + j*STRIDE] = 0.0;
			}
		}


		for (int q = 0; q < N_RA(); q += 2) {

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
			getComplexInverseHessenberg (m, STRIDE, invA, ipiv, &info, work);

			if (info != 0)
				return info;


			for (int i = 0; i < m; ++i) {

				for (int j = 0; j < m; ++j) {
					phiA[i + j*STRIDE] += 2.0 * std::real((res[q] / poles[q]) * invA[i + j * STRIDE]);
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
	int ExponentialIntegrator::expAc_variable(const int m, const double* A, const double c, double* phiA) {

		//allocate arrays
		int tid = omp_get_thread_num();
		int* ipiv = _unique<int>(tid, _ipiv);
		std::complex<double>* invA = _unique<std::complex<double>>(tid, _invA);
		std::complex<double>* work = _unique<std::complex<double>>(tid, _w);
		int info = 0;

		for (int i = 0; i < m; ++i) {

			for (int j = 0; j < m; ++j) {
				phiA[i + j*STRIDE] = 0.0;
			}
		}


		for (int q = 0; q < N_RA(); q += 2) {

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
			getComplexInverseHessenberg (m, STRIDE, invA, ipiv, &info, work);

			if (info != 0)
				return info;


			for (int i = 0; i < m; ++i) {

				for (int j = 0; j < m; ++j) {
					phiA[i + j*STRIDE] += 2.0 * std::real(res[q] * invA[i + j*STRIDE]);
				}
			}
		}

		return 0;
	}

}
