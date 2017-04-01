/**
 * \file
 * \brief An implementation of the van der Pol right hand side (y' = f(y)) function.
 *
 * Implements the evaluation of the derivative of the state vector with respect to time, \f$\dot{\vec{y}}\f$
 *
 */

#include "header.h"

#ifdef GENERATE_DOCS
//put this in the van der Pol namespace for documentation
namespace van_der_pol {
#endif

/**
 * \brief An implementation of the RHS of the van der Pol equation
 * \param[in]        t         The current system time
 * \param[in]        mu        The van der Pol parameter
 * \param[in]        y         The state vector
 * \param[out]       dy        The output RHS (dydt) vector
 *
 * The `y` and `dy` vectors supplied here are local versions of the global state vectors.
 * They have been transformed from the global Column major (Fortan) ordering to a local 1-D vector
 * Hence the vector accesses can be done in a simple manner below, i.e. y[0] -> \f$y_1\f$, y[1] -> \f$y_2\f$, etc.
 * @see solver_generic.c
 */
void dydt (const double t, const double mu, const double * __restrict__ y, double * __restrict__ dy) {

  // y1' = y2
  dy[0] = y[1];
  // y2' = mu(1 - y1^2)y2 - y1
  dy[1] = mu * (1 - y[0] * y[0]) * y[1] - y[0];

} // end dydt


#ifdef GENERATE_DOCS
}
#endif