#ifndef PHIA_HEAD_HESSENBERG_CU
#define PHIA_HEAD_HESSENBERG_CU

#include "header.cuh"

//void phiAv (const double*, const double, const double*, double*);
__device__ void phi2Ac_variable(const int, const double*, const double, double*);
__device__ void phiAc_variable(const int, const double*, const double, double*);
__device__ void expAc_variable(const int, const double*, const double, double*);

#endif