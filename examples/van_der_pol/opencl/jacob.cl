/**
 * \file
 * \brief An implementation of the van der Pol jacobian \f$\frac{\partial \dot{\vec{y}}}{\partial \vec{y}}\f$
 *
 * Implements the evaluation of van der Pol Jacobian
 *
 */

// include indexing macros
#include "jacob.h"

// indexing macros
#define __phiIndex(ind) (__getIndex1D(neq, (ind)))
#define __paramIndex(ind) (__getIndex1D(1, (ind)))
#define __jacIndex(row, col) (__getIndex2D(neq, neq, row, col))

/**
 *
 *  The Jacobian is in a local Column-major (Fortran) order.  As with dydt(), this function operates on local
 *  copies of the global state vector and jacobian.  Hence simple linear indexing can be used here.
 *  @see solver_generic.c
 *
 */
void jacobian (__private __ValueType const t, __global __ValueType const * __restrict__ mu,
               __global __ValueType const * __restrict__ y, __global __ValueType * __restrict__ jac,
               __global __ValueType* __restrict__ rwk)
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
    jac[__jacIndex(0, 0)] = 0;
    //!jac[1, 0] = \f$\frac{\partial \dot{y_2}}{\partial y_1}\f$
    jac[__jacIndex(1, 0)] = -2 * mu[__paramIndex(0)] * y[__phiIndex(0)] * y[__phiIndex(1)] - 1;
    //!jac[0, 1] = \f$\frac{\partial \dot{y_1}}{\partial y_2}\f$
    jac[__jacIndex(0, 1)] = 1;
    //!jac[1, 1] = \f$\frac{\partial \dot{y_2}}{\partial y_2}\f$
    jac[__jacIndex(1, 1)] = mu[__paramIndex(0)] * (1 - y[__phiIndex(0)] * y[__phiIndex(0)]);
}

#undef __phiIndex
#undef __paramIndex
#undef __jacIndex
