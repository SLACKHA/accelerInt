/**
 * \file
 * \brief An implementation of the Oregonator jacobian \f$\frac{\partial \dot{\vec{y}}}{\partial \vec{y}}\f$
 *
 * Implementes the evaluation of Oregonator Jacobian
 *
 */

#include "header.h"

#ifdef GENERATE_DOCS
//put this in the Oregonator namespace for documentation
namespace oregonator {
#endif

#define TRANSPOSE

/**
 * \brief An implementation of the Oregonator jacobian
 *
 * \param[in]           t               The current system time
 * \param[in]           mu              Dummy parameter, needed for compatibility only
 * \param[in]           y               The state vector at time t
 * \param[out]          jac             The jacobian to populate
 *
 *  The Jacobian is in a local Column-major (Fortran) order.  As with dydt(), this function operates on local
 *  copies of the global state vector and jacobian.  Hence simple linear indexing can be used here.
 *  @see solver_generic.c
 *
 */
void eval_jacob (const double t, const double mu, const double * __restrict__ y, double * __restrict__ jac)
{

  double s = 77.27;
  double q = 8.375E-6;
  double w = 0.161;

#ifndef TRANSPOSE
  // dydot_0 / dy0
  jac[0] = s * (-1 * y[1] + 1 - q * 2 * y[0]);
  // dydot_0 / dy1
  jac[1] = s * (1 - y[0]);
  // dydot_0 / dy2
  jac[2] = 0;
  // dydot_1 / dy0
  jac[3] = -1 * y[1] / s;
  // dydot_1 / dy1
  jac[4] = (-1 - y[0]) / s;
  // dydot_1 / dy2
  jac[5] = 1 / s;
  // dydot_2 / dy0
  jac[6] = w;
  // dydot_2 / dy1
  jac[7] = 0;
  // dydot_2 / dy2
  jac[8] = -1 * w;
#else
  // dydot_0 / dy0
  jac[0] = s * (-1 * y[1] + 1 - q * 2 * y[0]);
  // dydot_1 / dy0
  jac[1] = -1 * y[1] / s;
  // dydot_2 / dy0
  jac[2] = w;
  // dydot_0 / dy1
  jac[3] = s * (1 - y[0]);
  // dydot_1 / dy1
  jac[4] = (-1 - y[0]) / s;
  // dydot_2 / dy1
  jac[5] = 0;
  // dydot_0 / dy2
  jac[6] = 0;
  // dydot_1 / dy2
  jac[7] = 1 / s;
  // dydot_2 / dy2
  jac[8] = -1 * w;
#endif

}

#ifdef GENERATE_DOCS
}
#endif
