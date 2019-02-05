/**
 * \file
 * \brief A CUDA implementation of the Oregonator jacobian \f$\frac{\partial \dot{\vec{y}}}{\partial \vec{y}}\f$
 *
 * Implementes the CUDA evaluation of Oregonator Jacobian
 *
 */

#include "header.cuh"

#ifdef GENERATE_DOCS
//put this in the Oregonator namespace for documentation
namespace oregonator_cu {
#endif

/**
 * \brief An implementation of the Oregonator jacobian
 *
 * \param[in]           t               The current system time
 * \param[in]           mu              Dummy parameter, needed for compatibility only
 * \param[in]           y               The state vector at time t
 * \param[out]          jac             The jacobian to populate
 * \param[in]           d_mem           The mechanism_memory struct.  In future versions, this will be used to access the \f$\mu\f$ parameter to have a consistent interface.
 *
 *  The Jacobian is in Column-major (Fortran) order.  As with dydt(), this function operates directly on
 *  global state vector and jacobian.  Hence we use the #INDEX macro defined in gpu_macros.cuh here.
 *  @see dydt()
 *
 */
__device__
void eval_jacob (const double t, const double mu, const double * __restrict__ y, double * __restrict__ jac, const mechanism_memory * __restrict__ d_mem)
{
  jac[INDEX(0)] = s * (-1 * y[INDEX(1)] + 1 - q * 2 * y[INDEX(0)]);
  jac[INDEX(1)] = s * (1 - y[INDEX(0)]);
  jac[INDEX(2)] = 0;
  jac[INDEX(3)] = -1 * y[INDEX(1)] / s;
  jac[INDEX(4)] = (-1 - y[INDEX(0)]) / s;
  jac[INDEX(5)] = 1 / s;
  jac[INDEX(6)] = w;
  jac[INDEX(7)] = 0;
  jac[INDEX(8)] = -1 * w;

}

#ifdef GENERATE_DOCS
}
#endif
