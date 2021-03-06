/**
 * \file
 * \brief Finite Difference Jacobian implementation based on CVODEs
 */

#include "fd_jacob.cuh"

//! The finite difference order [Default: 1]
#define FD_ORD 1

// Finite difference coefficients
#if FD_ORD == 2
  __constant__ double x_coeffs[FD_ORD] = {-1.0, 1.0};
  __constant__ double y_coeffs[FD_ORD] = {-0.5, 0.5};
#elif FD_ORD == 4
  __constant__ double x_coeffs[FD_ORD] = {-2.0, -1.0, 1.0, 2.0};
  __constant__ double y_coeffs[FD_ORD] = {1.0 / 12.0, -2.0 / 3.0, 2.0 / 3.0, -1.0 / 12.0};
#elif FD_ORD == 6
  __constant__ double x_coeffs[FD_ORD] = {-3.0, -2.0, - 1.0, 1.0, 2.0, 3.0};
  __constant__ double y_coeffs[FD_ORD] = {-1.0 / 60.0, 3.0 / 20.0, -3.0 / 4.0, 3.0 / 4.0, -3.0 / 20.0, 1.0 / 60.0};
#endif

/**
 * \brief Computes a finite difference Jacobian of order FD_ORD of the RHS function dydt at the given pressure and state
 *
 * \param[in]         t           the current system time
 * \param[in]         pres        the current system pressure
 * \param[in]         cy          the system state vector
 * \param[out]        jac         the resulting Jacobian
 * \param[in]         d_mem       the mechanism_memory object used in computing dydt
 * \param[in]         y_temp      a work array for the state vector
 * \param[in]         ewt         a storage for the error weights in computing the Jacobian perturbation factor
 */
__device__
void eval_jacob (const double t, const double pres, const double * __restrict__ cy,
                    double * __restrict__ jac, const mechanism_memory* __restrict__ d_mem,
                    double* __restrict__ y_temp, double* __restrict__ ewt) {
  double* dy = d_mem->dy;

  #pragma unroll
  for (int i = 0; i < NSP; ++i) {
    y_temp[INDEX(i)] = cy[INDEX(i)];
    ewt[INDEX(i)] = ATOL + (RTOL * fabs(cy[INDEX(i)]));
  }

  dydt (t, pres, cy, dy, d_mem);
  #if FD_ORD == 1
  #pragma unroll
  for (int j = 0; j < NSP; ++j) {
      #pragma unroll
      for (int i = 0; i < NSP; ++i) {
        jac[INDEX(i + NSP*j)] = dy[INDEX(i)];
      }
  }
  #endif

  // unit roundoff of machine
  double srur = sqrt(DBL_EPSILON);

  double sum = 0.0;
  #pragma unroll
  for (int i = 0; i < NSP; ++i) {
    sum += (ewt[INDEX(i)] * dy[INDEX(i)]) * (ewt[INDEX(i)] * dy[INDEX(i)]);
  }
  double fac = sqrt(sum / ((double)(NSP)));
  double r0 = 1000.0 * RTOL * DBL_EPSILON * ((double)(NSP)) * fac;


  #pragma unroll
  for (int j = 0; j < NSP; ++j) {
    double yj_orig = y_temp[INDEX(j)];
    double r = fmax(srur * fabs(yj_orig), r0 / ewt[INDEX(j)]);

    #if FD_ORD == 1
      y_temp[INDEX(j)] = yj_orig + r;
      dydt (t, pres, y_temp, dy, d_mem);

      #pragma unroll
      for (int i = 0; i < NSP; ++i) {
        jac[INDEX(i + NSP*j)] = (dy[INDEX(i)] - jac[INDEX(i + NSP*j)]) / r;
      }
    #else
      #pragma unroll
      for (int i = 0; i < NSP; ++i) {
        jac[INDEX(i + NSP*j)] = 0.0;
      }
      #pragma unroll
      for (int k = 0; k < FD_ORD; ++k) {
        y_temp[INDEX(j)] = yj_orig + x_coeffs[k] * r;
        dydt (t, pres, y_temp, dy, d_mem);

        #pragma unroll
        for (int i = 0; i < NSP; ++i) {
          jac[INDEX(i + NSP*j)] += y_coeffs[k] * y_temp[INDEX(i)];
        }
      }
      #pragma unroll
      for (int i = 0; i < NSP; ++i) {
        jac[INDEX(i + NSP*j)] /= r;
      }
    #endif

    y_temp[INDEX(j)] = yj_orig;
  }

}