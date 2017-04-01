/**
 * \file
 * \brief File containing functions for best rational approximation to matrix exponential.
 * \author Kyle E. Niemeyer
 * \date 07/19/2012
 *
 * Contains main and linear algebra functions for Carathéodory-Fejér method of Rational Approximants
 */

#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "solver_options.h"
#include "linear-algebra.h"

/** Defined for pi */
#define M_PI  4 * atan(1)

/** Complex math */
#include <complex.h>

/** Fast Fourier tranform functions */
#include <fftw3.h>

////////////////////////////////////////////////////////////////////////

/** \brief Function that calculates the poles and residuals of best rational
 * (partial fraction) approximant to the matrix exponential.
 *
 * Uses the Carathéodory-Fejér method; based on the MATLAB code in
 * L.N. Trefethen, J.A.C. Weideman, T. Schmelzer, "Talbot quadratures
 * and rational approximations," BIT Numer. Math. 46 (2006) 653–670.
 *
 * \param[in]	n				size of approximation (n, n)
 * \param[out]	poles_r			array with real parts of poles, size n
 * \param[out]	poles_i			array with imaginary parts of poles, size n
 * \param[out]	res_r			array with real parts of residuals, size n
 * \param[out]	res_i			array with imaginary parts of residuals, size n
 */
void cf ( int n, double* poles_r, double* poles_i, double* res_r, double* res_i ) {
	// number of Chebyshev coefficients
	const int K = 75;

	// number of points for FFT
	const int nf = 1024;

	// roots of unity
	fftw_complex *w = fftw_alloc_complex(nf);
	for (int i = 0; i < nf; ++i) {
		w[i] = cexp(2.0 * I * M_PI * (double)i / (double)nf);
	}

	// Chebyshev points (twice over)
	double *t = (double*) calloc (nf, sizeof(double));
	for (int i = 0; i < nf; ++i) {
		t[i] = creal(w[i]);
	}

	// scale factor for stability
	double scl = 9.0;

	// exp(x) transpl. to [-1,1]
	fftw_complex *F = fftw_alloc_complex(nf);
	for (int i = 0; i < nf; ++i) {
		F[i] = cexp(scl * (t[i] - 1.0) / (t[i] + 1.0 + ATOL));
	}

	free (t);

	// Chebyshev coefficients of F
	fftw_complex *fftF = fftw_alloc_complex(nf);
	fftw_plan p;
	p = fftw_plan_dft_1d(nf, F, fftF, FFTW_FORWARD, FFTW_ESTIMATE);
	fftw_execute(p);

	double *c = (double*) calloc (nf, sizeof(double));
	for (int i = 0; i < nf; ++i) {
		c[i] = creal(fftF[i]) / (double)nf;
	}

	fftw_free (fftF);
	fftw_free (F);

	// analytic part of f of F
	fftw_complex *f = fftw_alloc_complex(nf);
	memset (f, 0.0, nf * sizeof(fftw_complex)); // set to zero

	for (int i = 0; i < nf; ++i) {
		for (int j = K; j >= 0; --j) {
			f[i] += c[j] * cpow(w[i], j);
		}
	}

	// SVD of Hankel matrix
	double *S = (double*) calloc (K, sizeof(double));
	double *U = (double*) calloc (K * K, sizeof(double));
	double *V = (double*) calloc (K * K, sizeof(double));
	double *hankel = (double*) calloc (K * K, sizeof(double));

	memset (hankel, 0.0, K * K * sizeof(double));		// fill with 0

	for (int i = 0; i < K; ++i) {
		for (int j = 0; j <= i; ++j) {
			hankel[(i - j) + K * j] = c[i + 1];
		}
	}
	svd (K, hankel, S, U, V);

	free (c);
	free (hankel);

	// singular value
	double s_val = S[n];

	// singular vectors
	// need u and v to be complex type for fft
	fftw_complex *u = fftw_alloc_complex(nf);
	fftw_complex *v = fftw_alloc_complex(nf);

	// fill with zeros
	memset (u, 0.0, nf * sizeof(fftw_complex));
	memset (v, 0.0, nf * sizeof(fftw_complex));

	for (int i = 0; i < K; ++i) {
		u[i] = U[(K - i - 1) + K * n];
		v[i] = V[i + K * n];
	}

	free (U);
	free (V);
	free (S);

	// create copy of v for later use
	double *v2 = (double*) calloc (K, sizeof(double));
	for (int i = 0; i < K; ++i) {
		v2[i] = creal(v[i]);
	}

	// finite Blaschke product
	fftw_complex *b1 = fftw_alloc_complex(nf);
	fftw_complex *b2 = fftw_alloc_complex(nf);
	fftw_complex *b = fftw_alloc_complex(nf);

	p = fftw_plan_dft_1d(nf, u, b1, FFTW_FORWARD, FFTW_ESTIMATE);
	fftw_execute(p);

	p = fftw_plan_dft_1d(nf, v, b2, FFTW_FORWARD, FFTW_ESTIMATE);
	fftw_execute(p);

	for (int i = 0; i < nf; ++i) {
		b[i] = b1[i] / b2[i];
	}

	fftw_free (u);
	fftw_free (v);
	fftw_free (b1);
	fftw_free (b2);

	// extended function r-tilde
	fftw_complex *rt = fftw_alloc_complex(nf);
	for (int i = 0; i < nf; ++i) {
		rt[i] = f[i] - s_val * cpow(w[i], K) * b[i];
	}

	fftw_free (f);
	fftw_free (b);

	// poles
	double complex *zr = (double complex*) calloc (K - 1, sizeof(double complex));

	// get roots of v
	roots (K, v2, zr);

	free (v2);

	double complex *qk = (double complex*) calloc (n, sizeof(double complex));
	memset (qk, 0.0, n * sizeof(double complex));

	int i = 0;
	for (int j = 0; j < K - 1; ++j) {
		if (cabs(zr[j]) > 1.0) {
			qk[i] = zr[j];
			i += 1;
		}
	}

	free (zr);

	// coefficients of denominator
	double complex *qc_i = (double complex*) calloc (n + 1, sizeof(double complex));
	memset (qc_i, 0.0, (n + 1) * sizeof(double complex));		// fill with 0

	qc_i[0] = 1.0;
	for (int j = 0; j < n; ++j) {
		double complex qc_old1 = qc_i[0];
		double complex qc_old2;

		for (int i = 1; i < j + 2; ++i) {
			qc_old2 = qc_i[i];
			qc_i[i] = qc_i[i] - qk[j] * qc_old1;
			qc_old1 = qc_old2;
		}
	}

	// qc_i will only have real parts, but want real array
	double *qc = (double*) calloc (n + 1, sizeof(double));
	for (int i = 0; i < n + 1; ++i) {
		qc[i] = creal(qc_i[i]);
	}

	free (qc_i);

	// numerator
	fftw_complex *pt = fftw_alloc_complex(nf);
	memset (pt, 0.0, nf * sizeof(fftw_complex));
	for (int i = 0; i < nf; ++i) {
		for (int j = 0; j < n + 1; ++j) {
			pt[i] += qc[j] * cpow(w[i], n - j);
		}
		pt[i] *= rt[i];
	}

	fftw_free (w);
	free (qc);
	fftw_free (rt);

	// coefficients of numerator
	fftw_complex *ptc_i = fftw_alloc_complex(nf);
	p = fftw_plan_dft_1d(nf, pt, ptc_i, FFTW_FORWARD, FFTW_ESTIMATE);
	fftw_execute(p);

	double *ptc = (double*) calloc(n + 1, sizeof(double));
	for (int i = 0; i < n + 1; ++i) {
		ptc[i] = creal(ptc_i[n - i]) / (double)nf;
	}

	fftw_destroy_plan(p);
	fftw_free (pt);
	fftw_free (ptc_i);

	// poles
	double complex *res = (double complex*) calloc (n, sizeof(double complex));
	memset (res, 0.0, n * sizeof(double complex));		// fill with 0

	double complex *qk2 = (double complex*) calloc (n - 1, sizeof(double complex));
	double complex *q2 = (double complex*) calloc (n, sizeof(double complex));

	// calculate residues
	for (int k = 0; k < n; ++k) {
		double complex q = qk[k];

		int j = 0;
		for (int i = 0; i < n; ++i) {
			if (i != k) {
				qk2[j] = qk[i];
				j += 1;
			}
		}

		memset (q2, 0.0, n * sizeof(double complex));		// fill with 0
		q2[0] = 1.0;
		for (int j = 0; j < n - 1; ++j) {
			double complex q_old1 = q2[0];
			double complex q_old2;
			for (int i = 1; i < j + 2; ++i) {
				q_old2 = q2[i];
				q2[i] = q2[i] - qk2[j] * q_old1;
				q_old1 = q_old2;
			}
		}

		double complex ck1 = 0.0;
		for (int i = 0; i < n + 1; ++i) {
			ck1 += ptc[i] * cpow(q, n - i);
		}

		double complex ck2 = 0.0;
		for (int i = 0; i < n; ++i) {
			ck2 += q2[i] * cpow(q, n - 1 - i);
		}

		res[k] = ck1 / ck2;
	}

	free (ptc);
	free (qk2);
	free (q2);

	double complex *poles = (double complex*) calloc (n, sizeof(double complex));
	for (int i = 0; i < n; ++i) {
		// poles in z-plane
		poles[i] = scl * (qk[i] - 1.0) * (qk[i] - 1.0) / ((qk[i] + 1.0) * (qk[i] + 1.0));

		// residues in z-plane
		res[i] = 4.0 * res[i] * poles[i] / (qk[i] * qk[i] - 1.0);
	}

	// separate real and imaginary parts
	for (int i = 0; i < n; ++i) {
		poles_r[i] = creal(poles[i]);
		poles_i[i] = cimag(poles[i]);

		res_r[i] = creal(res[i]);
		res_i[i] = cimag(res[i]);
	}

	free (qk);
	free (poles);
	free (res);
  fftw_cleanup();
}
