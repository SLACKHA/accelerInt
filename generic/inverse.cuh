#ifndef INVERSE_CUH
#define INVERSE_CUH

__device__
void getLU (const int n, const int LDA, double* A, int* indPivot, int* info);

#endif