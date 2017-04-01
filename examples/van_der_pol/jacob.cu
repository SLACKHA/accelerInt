/**
 * \file
 * \brief A CUDA implementation of the van der Pol jacobian \f$\frac{\partial \dot{\vec{y}}}{\partial \vec{y}}\f$
 *
 * Implementes the CUDA evaluation of van der Pol Jacobian
 *
 */

#include "header.cuh"

#ifdef GENERATE_DOCS
//put this in the van der Pol namespace for documentation
namespace van_der_pol_cu {
#endif

/**
 * \brief An implementation of the van der Pol jacobian
 *
 * \param[in]           t               The current system time
 * \param[in]           mu              The van der Pol parameter
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
    //Note, to reach index [i, j] of the Jacobian, we multiply `i` by NSP, the size of the first dimension of Jacobian and add j, i.e.:
    //jac[i, j] -> jac[i * NSP + j]
    //!jac[0, 0] = \f$\frac{\partial \dot{y_1}}{\partial y_1}\f$
    jac[INDEX(0 * NSP + 0)] = 0;
    //!jac[0, 1] = \f$\frac{\partial \dot{y_2}}{\partial y_1}\f$
    jac[INDEX(0 * NSP + 1)] = -2 * mu * y[INDEX(0)] * y[INDEX(1)] - 1;
    //!jac[1, 0] = \f$\frac{\partial \dot{y_1}}{\partial y_2}\f$
    jac[INDEX(1 * NSP + 0)] = 1;
    //!jac[1, 1] = \f$\frac{\partial \dot{y_2}}{\partial y_2}\f$
    jac[INDEX(1 * NSP + 1)] = mu * (1 - y[INDEX(0)] * y[INDEX(0)]);
}

#ifdef GENERATE_DOCS
}
#endif