/**
 * \file
 * \brief Header definitions for LU factorization routines
 */

#ifndef COMPLEX_INVERSE_H
#define COMPLEX_INVERSE_H

#include <complex>

void getComplexInverseHessenberg (const int, const int, std::complex<double>* __restrict__, int * __restrict__,
								  int * __restrict__, std::complex<double>* __restrict__);

#endif
