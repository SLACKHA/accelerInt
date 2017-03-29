/**
 * \file
 * \brief External lapack routine definitions
 */

#ifndef LAPACK_DFNS_H
#define LAPACK_DFNS_H

#include <complex.h>

//defines the external lapack routines
extern void dgetrf_ (const int* m, const int* n, double* A, const int* lda, int* ipiv, int* info);
extern void zgetrf_ (const int* m, const int* n, double complex* A, const int* lda, int* ipiv, int* info);
extern void dgetrs_ (const char* trans, const int* n, const int* nrhs, double* A, const int* LDA, int* ipiv, double* B, const int* LDB, int* info);
extern void zgetrs_ (const char* trans, const int* n, const int* nrhs, double complex* A, const int* LDA, int* ipiv, double complex* B, const int* LDB, int* info);
extern void zgetri_ (const int* n, double complex* A, const int* LDA, int* ipiv, double complex* work, const int* LDB, int* info);

#endif