#include "header.h"
#include <math.h>
#include <float.h>
#include "dydt.cuh"
#include "gpu_macros.cuh"

#define FD_ORD 6

// Finite difference coefficients
#if FD_ORD == 1
  __constant__ double x_coeffs[FD_ORD] = {1.0};
  __constant__ double y_coeffs[FD_ORD] = {1.0};
#elif FD_ORD == 2
  __constant__ double x_coeffs[FD_ORD] = {-1.0, 1.0};
  __constant__ double y_coeffs[FD_ORD] = {-0.5, 0.5};
#elif FD_ORD == 4
  __constant__ double x_coeffs[FD_ORD] = {-2.0, -1.0, 1.0, 2.0};
  __constant__ double y_coeffs[FD_ORD] = {1.0 / 12.0, -2.0 / 3.0, 2.0 / 3.0, -1.0 / 12.0};
#elif FD_ORD == 6
  __constant__ double x_coeffs[FD_ORD] = {-3.0, -2.0, - 1.0, 1.0, 2.0, 3.0};
  __constant__ double y_coeffs[FD_ORD] = {-1.0 / 60.0, 3.0 / 20.0, -3.0 / 4.0, 3.0 / 4.0, -3.0 / 20.0, 1.0 / 60.0};
#endif

__device__
void eval_jacob (const double t, const double pres, double * y, double * jac) {
  double dy[NN];
  double error[NN];

  dydt (t, pres, y, dy);
  
  #pragma unroll
  for (uint i = 0; i < NN; ++i) {
    error[INDEX(i)] = ATOL + (RTOL * fabs(y[INDEX(i)]));
  }
  
  // unit roundoff of machine
  double srur = sqrt(DBL_EPSILON);
  
  double sum = 0.0;
  #pragma unroll
  for (uint i = 0; i < NN; ++i) {
    sum += (error[INDEX(i)] * dy[INDEX(i)]) * (error[INDEX(i)] * dy[INDEX(i)]);
  }
  double fac = sqrt(sum / ((double)(NN)));
  double r0 = 1000.0 * RTOL * DBL_EPSILON * ((double)(NN)) * fac;
  
#ifndef GLOBAL_MEM
  double f_temp[NN];
#endif
  
  #pragma unroll
  for (uint j = 0; j < NN; ++j) {
    double yj_orig = y[INDEX(j)];
    double r = fmax(srur * fabs(yj_orig), r0 / error[INDEX(j)]);
    
    #pragma unroll
    for (uint i = 0; i < NN; ++i) {
      jac[INDEX(i + NN*j)] = ZERO;
    }
    
    #if FD_ORD == 1
      y[INDEX(j)] = yj_orig + r;
      dydt (t, pres, y, f_temp);
        
      #pragma unroll
      for (uint i = 0; i < NN; ++i) {
        jac[INDEX(i + NN*j)] += (f_temp[INDEX(i)] - dy[INDEX(i)]);
      }
    #else
      #pragma unroll
      for (uint k = 0; k < FD_ORD; ++k) {
        y[INDEX(j)] = yj_orig + x_coeffs[k] * r;
        dydt (t, pres, y, f_temp);
        
        #pragma unroll
        for (uint i = 0; i < NN; ++i) {
          jac[INDEX(i + NN*j)] += y_coeffs[k] * f_temp[INDEX(i)];
        }
      }
    #endif
    
    #pragma unroll
    for (uint i = 0; i < NN; ++i) {
      jac[INDEX(i + NN*j)] /= r;
    }
    
    y[INDEX(j)] = yj_orig;
  }
  
}