#ifndef COMPLEX_INVERSE_CUH
#define COMPLEX_INVERSE_CUH

#include <cuComplex.h>

__device__ void getComplexLU (cuDoubleComplex*, int*, int*);

#endif