#ifndef COMPLEX_INVERSE_CUH
#define COMPLEX_INVERSE_CUH

#include <cuComplex.h>

__device__ void getComplexInverse (cuDoubleComplex*);
__device__ void getComplexInverseHessenberg (const int, cuDoubleComplex*);

#endif