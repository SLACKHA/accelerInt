#include "header.h"
#include "derivs.h"
#include <math.h>
#include <float.h>

#define FD_ORD 2

void eval_jacob (const Real t, const Real pres, Real * y, Real * jac) {
  
  Real ydot[NN];
  dydt (t, pres, y, ydot);
  
  // Finite difference coefficients
  Real x_coeffs[FD_ORD];
  Real y_coeffs[FD_ORD];
  
  if (FD_ORD == 2) {
    // 2nd order central difference
    x_coeffs[0] = -1.0;
    x_coeffs[1] = 1.0;
    y_coeffs[0] = -0.5;
    y_coeffs[1] = 0.5;
  } else if (FD_ORD == 4) {
    // 4th order central difference
    x_coeffs[0] = -2.0;
    x_coeffs[1] = -1.0;
    x_coeffs[2] = 1.0;
    x_coeffs[3] = 2.0;
    y_coeffs[0] = 1.0 / 12.0;
    y_coeffs[1] = -2.0 / 3.0;
    y_coeffs[2] = 2.0 / 3.0;
    y_coeffs[3] = -1.0 / 12.0;
  } else {
    // 6th order central difference
    x_coeffs[0] = -3.0;
    x_coeffs[1] = -2.0;
    x_coeffs[2] = -1.0;
    x_coeffs[3] = 1.0;
    x_coeffs[4] = 2.0;
    x_coeffs[5] = 3.0;
    
    y_coeffs[0] = -1.0 / 60.0;
    y_coeffs[1] = 3.0 / 20.0;
    y_coeffs[2] = -3.0 / 4.0;
    y_coeffs[3] = 3.0 / 4.0;
    y_coeffs[4] = -3.0 / 20.0;
    y_coeffs[5] = 1.0 / 60.0;
  }
  
  Real ewt[NN];
  #pragma unroll
  for (uint i = 0; i < NN; ++i) {
    ewt[i] = ATOL + (RTOL * fabs(y[i]));
  }
  
  // unit roundoff of machine
  Real srur = sqrt(DBL_EPSILON);
  
  Real sum = 0.0;
  #pragma unroll
  for (uint i = 0; i < NN; ++i) {
    sum += (ewt[i] * ydot[i]) * (ewt[i] * ydot[i]);
  }
  Real fac = sqrt(sum / ((Real)(NN)));
  Real r0 = 1000.0 * RTOL * DBL_EPSILON * ((Real)(NN)) * fac;
  
  Real ftemp[NN];
  
  #pragma unroll
  for (uint j = 0; j < NN; ++j) {
    Real yj_orig = y[j];
    Real r = fmax(srur * fabs(yj_orig), r0 / ewt[j]);
    
    #pragma unroll
    for (uint i = 0; i < NN; ++i) {
      jac[i + NN*j] = ZERO;
    }
    
    #pragma unroll
    for (uint k = 0; k < FD_ORD; ++k) {
      y[j] = yj_orig + x_coeffs[k] * r;
      dydt (t, pres, y, ftemp);
      
      #pragma unroll
      for (uint i = 0; i < NN; ++i) {
        jac[i + NN*j] += y_coeffs[k] * ftemp[i];
      }
    }
    
    #pragma unroll
    for (uint i = 0; i < NN; ++i) {
      jac[i + NN*j] /= r;
    }
    
    y[j] = yj_orig;
  }
  
}