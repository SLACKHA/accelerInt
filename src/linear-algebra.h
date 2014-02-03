#ifndef LINEAR_ALGEBRA_HEAD
#define LINEAR_ALGEBRA_HEAD

#include <complex.h>

void getInverseComplex (int, double complex*);
void linSolveComplex (int, double complex*, double complex*, double complex*);
void roots (int, double*, double complex*);
void svd (int, double*, double*, double*, double*);

#endif