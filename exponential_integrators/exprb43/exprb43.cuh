#ifndef EXPRB43_HEAD
#define EXPRB43_HEAD

#include "header.cuh"

__device__ void integrate (const double, const double, const double,
							double* __restrict__, const mechanism_memory* __restrict__,
							const solver_memory* __restrict__);

#endif