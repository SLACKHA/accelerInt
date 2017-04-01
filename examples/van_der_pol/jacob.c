/**
 * \file
 * \brief An implementation of the van der Pol jacobian \f$\frac{\partial \dot{\vec{y}}}{\partial \vec{y}}\f$
 *
 * Implementes the evaluation of van der Pol Jacobian
 *
 */

#include "header.h"

#ifdef GENERATE_DOCS
//put this in the van der Pol namespace for documentation
namespace van_der_pol {
#endif

/**
 * \brief An implementation of the van der Pol jacobian
 *
 * \param[in]           t               The current system time
 * \param[in]           mu              The van der Pol parameter
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
    //Note, to reach index [i, j] of the Jacobian, we multiply `i` by NSP, the size of the first dimension of Jacobian and add j, i.e.:
    //jac[i, j] -> jac[i * NSP + j]
    //!jac[0, 0] = \f$\frac{\partial \dot{y_1}}{\partial y_1}\f$
    jac[0 * NSP + 0] = 0;
    //!jac[0, 1] = \f$\frac{\partial \dot{y_2}}{\partial y_1}\f$
    jac[0 * NSP + 1] = -2 * mu * y[0] * y[1] - 1;
    //!jac[1, 0] = \f$\frac{\partial \dot{y_1}}{\partial y_2}\f$
    jac[1 * NSP + 0] = 1;
    //!jac[1, 1] = \f$\frac{\partial \dot{y_2}}{\partial y_2}\f$
    jac[1 * NSP + 1] = mu * (1 - y[0] * y[0]);
}

#ifdef GENERATE_DOCS
}
#endif