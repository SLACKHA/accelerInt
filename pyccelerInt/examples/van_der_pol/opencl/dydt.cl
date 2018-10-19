/**
 * \file
 * \brief An implementation of the van der Pol right hand side (y' = f(y)) function.
 *
 * Implements the evaluation of the derivative of the state vector with respect to time, \f$\dot{\vec{y}}\f$
 * for OpenCL
 *
 */

// include indexing macros for OpenCL to make life easy
#include "dydt.h"

// indexing macros
#define __phiIndex(ind) (__getIndex1D(neq, (ind)))
#define __paramIndex(ind) (__getIndex1D(1, (ind)))

/**
 * The `y` and `dy` vectors supplied here are local versions of the global state vectors.
 * They have been transformed from the global Column major (Fortan) ordering to a local 1-D vector.
 * Hence the vector accesses can be done in a simple manner below, i.e. y[0] -> \f$y_1\f$, y[1] -> \f$y_2\f$, etc.
 * @see solver_generic.c
 */
void dydt (__private __ValueType const t, __global __ValueType const * __restrict__ mu,
           __global __ValueType const * __restrict__ y, __global __ValueType * __restrict__ dy,
           __global __ValueType* __restrict__ rwk) {

  // y1' = y2
  dy[__phiIndex(0)] = y[__phiIndex(1)];
  // y2' = mu(1 - y1^2)y2 - y1
  dy[__phiIndex(1)] =  mu[__paramIndex(0)] * (1 - y[__phiIndex(0)] * y[__phiIndex(0)]) * y[__phiIndex(1)] - y[__phiIndex(0)];

} // end dydt

#undef __phiIndex
#undef __paramIndex
