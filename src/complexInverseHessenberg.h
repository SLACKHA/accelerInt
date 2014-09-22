#ifndef COMPLEX_INVERSE_H
#define COMPLEX_INVERSE_H

void getComplexInverseHessenberg (const int n, const int STRIDE, double complex* A);
int getHessenbergLU_test(const int n, const int STRIDE, double complex* A, int* indPivot);
#endif