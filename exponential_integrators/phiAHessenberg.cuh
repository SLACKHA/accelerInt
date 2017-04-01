/**
 * \file
 * \brief Header for Matrix exponential (phi) methods
 */

#ifndef PHIA_HEAD_HESSENBERG_CU
#define PHIA_HEAD_HESSENBERG_CU

#include "header.cuh"

//void phiAv (const double*, const double, const double*, double*);
__device__ int phi2Ac_variable(const int, const double* __restrict__, const double, double* __restrict__,
								const solver_memory* __restrict__, cuDoubleComplex* __restrict__);
__device__ int phiAc_variable(const int, const double* __restrict__, const double, double* __restrict__,
								const solver_memory* __restrict__, cuDoubleComplex* __restrict__);
__device__ int expAc_variable(const int, const double* __restrict__, const double, double* __restrict__,
								const solver_memory* __restrict__, cuDoubleComplex* __restrict__);

#endif