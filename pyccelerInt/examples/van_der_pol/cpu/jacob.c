/**
 * \file
 * \brief An implementation of the van der Pol jacobian \f$\frac{\partial \dot{\vec{y}}}{\partial \vec{y}}\f$
 *
 * Implements the evaluation of van der Pol Jacobian
 *
 */

#include "jacob.h"

/**
 *
 *  The Jacobian is in a local Column-major (Fortran) order.  As with dydt(), this function operates on local
 *  copies of the global state vector and jacobian.  Hence simple linear indexing can be used here.
 *  @see solver_generic.c
 *
 */
void eval_jacob (const double t, const double mu, const double * __restrict__ y, double * __restrict__ jac,
                 double* __restrict__ rwk)
{
    //! \note To get the value at index [i, j] of the Jacobian,
    //! we multiply `j` by NSP, the size of the first dimension of Jacobian
    //! and add `j`, i.e.:
    //! jac[i, j] -> jac[i + j * NSP].
    //!
    //! Remember that the Jacobian is flatten in column-major order, hence:
    //! \code
    //!     jac = [jac[0, 0], jac[1, 0], jac[0, 1], jac[1, 1]]
    //! \endcode

    //!jac[0, 0] = \f$\frac{\partial \dot{y_1}}{\partial y_1}\f$
    jac[0 + 0 * 2] = 0;
    //!jac[1, 0] = \f$\frac{\partial \dot{y_2}}{\partial y_1}\f$
    jac[1 + 0 * 2] = -2 * mu * y[0] * y[1] - 1;
    //!jac[0, 1] = \f$\frac{\partial \dot{y_1}}{\partial y_2}\f$
    jac[0 + 1 * 2] = 1;
    //!jac[1, 1] = \f$\frac{\partial \dot{y_2}}{\partial y_2}\f$
    jac[1 + 1 * 2] = mu * (1 - y[0] * y[0]);
}
