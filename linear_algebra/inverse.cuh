/**
 * \file
 * \brief Headers for CUDA LU decomposition implementation
 */

#ifndef INVERSE_CUH
#define INVERSE_CUH

__device__
void getLU (const int, double* __restrict__, int* __restrict__, int* __restrict__);

#endif