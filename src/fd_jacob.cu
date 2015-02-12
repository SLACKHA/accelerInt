#include "header.h"
#include <math.h>
#include <float.h>
#include "dydt.cuh"
#include "gpu_macros.cuh"

#define FD_ORD 6

#ifdef GLOBAL_MEM
extern __device__ Real* dy;
extern __device__ Real* error;
extern __device__ Real* f_temp;
#endif

// Finite difference coefficients
#if FD_ORD == 2
  __constant__ Real x_coeffs[FD_ORD] = {-1.0, 1.0};
  __constant__ Real y_coeffs[FD_ORD] = {-0.5, 0.5};
#elif FD_ORD == 4
  __constant__ Real x_coeffs[FD_ORD] = {-2.0, -1.0, 1.0, 2.0};
  __constant__ Real y_coeffs[FD_ORD] = {1.0 / 12.0, -2.0 / 3.0, 2.0 / 3.0, -1.0 / 12.0};
#elif FD_ORD == 6
  __constant__ Real x_coeffs[FD_ORD] = {-3.0, -2.0, - 1.0, 1.0, 2.0, 3.0};
  __constant__ Real y_coeffs[FD_ORD] = {-1.0 / 60.0, 3.0 / 20.0, -3.0 / 4.0, 3.0 / 4.0, -3.0 / 20.0, 1.0 / 60.0};
#endif

__device__
void eval_jacob (const Real t, const Real pres, Real * y, Real * jac) {
  
  dydt (t, pres, y, dy);
  
#ifndef GLOBAL_MEM
  Real ewt[NN];
#endif
  #pragma unroll
  for (uint i = 0; i < NN; ++i) {
    error[INDEX(i)] = ATOL + (RTOL * fabs(y[INDEX(i)]));
  }
  
  // unit roundoff of machine
  Real srur = sqrt(DBL_EPSILON);
  
  Real sum = 0.0;
  #pragma unroll
  for (uint i = 0; i < NN; ++i) {
    sum += (error[INDEX(i)] * dy[INDEX(i)]) * (error[INDEX(i)] * dy[INDEX(i)]);
  }
  Real fac = sqrt(sum / ((Real)(NN)));
  Real r0 = 1000.0 * RTOL * DBL_EPSILON * ((Real)(NN)) * fac;
  
#ifndef GLOBAL_MEM
  Real f_temp[NN];
#endif
  
  #pragma unroll
  for (uint j = 0; j < NN; ++j) {
    Real yj_orig = y[INDEX(j)];
    Real r = fmax(srur * fabs(yj_orig), r0 / error[INDEX(j)]);
    
    #pragma unroll
    for (uint i = 0; i < NN; ++i) {
      jac[INDEX(i + NN*j)] = ZERO;
    }
    
    #pragma unroll
    for (uint k = 0; k < FD_ORD; ++k) {
      y[INDEX(j)] = yj_orig + x_coeffs[k] * r;
      dydt (t, pres, y, f_temp);
      
      #pragma unroll
      for (uint i = 0; i < NN; ++i) {
        jac[INDEX(i + NN*j)] += y_coeffs[k] * f_temp[INDEX(i)];
      }
    }
    
    #pragma unroll
    for (uint i = 0; i < NN; ++i) {
      jac[INDEX(i + NN*j)] /= r;
    }
    
    y[INDEX(j)] = yj_orig;
  }
  
}