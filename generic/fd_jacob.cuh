#ifndef FD_JACOB_CU_HEAD
#define FD_JACOB_CU_HEAD

#include <math.h>
#include <float.h>
#include "header.cuh"
#include "dydt.cuh"
#include "gpu_macros.cuh"
#include "solver_options.cuh"

__device__
void eval_jacob (const double, const double, const double * __restrict__,
                    double * __restrict__, const mechanism_memory* __restrict__,
                    double* __restrict__, double* __restrict__);

#endif