#ifndef LAPACK_DFNS_H
#define LAPACK_DFNS_H

#include <complex.h>

//defines the external lapack routines
extern void dgetrf_ (int* m, int* n, double* A, int* lda, int* ipiv, int* info);
extern void zgetrf_ (int* m, int* n, double complex* A, int* lda, int* ipiv, int* info);
extern void dgetrs_ (char* trans, int* n, int* nrhs, double* A, int* LDA, int* ipiv, double* B, int* LDB, int* info);
extern void zgetrs_ (char* trans, int* n, int* nrhs, double complex* A, int* LDA, int* ipiv, double complex* B, int* LDB, int* info);
extern void zgetri_ (const int* n, double complex* A, const int* LDA, int* ipiv, double complex* work, const int* LDB, int* info);

#endif