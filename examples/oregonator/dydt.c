/**
 * \file
 * \brief An implementation of the Oregonator right hand side (y' = f(y)) function.
 *
 * Implements the evaluation of the derivative of the state vector with respect to time, \f$\dot{\vec{y}}\f$
 *
 */

#include "header.h"

#ifdef GENERATE_DOCS
//put this in the Oregonator namespace for documentation
namespace oregonator {
#endif

/**
 * \brief An implementation of the RHS of the Oregonator equation
 * \param[in]        t         The current system time
 * \param[in]        mu        Dummy parameter, needed for compatibility only
 * \param[in]        y         The state vector
 * \param[out]       dy        The output RHS (dydt) vector
 *
 * The `y` and `dy` vectors supplied here are local versions of the global state vectors.
 * They have been transformed from the global Column major (Fortan) ordering to a local 1-D vector
 * Hence the vector accesses can be done in a simple manner below, i.e. y[0] -> \f$y_1\f$, y[1] -> \f$y_2\f$, etc.
 * @see solver_generic.c
 */
void dydt (const double t, const double mu, const double * __restrict__ y, double * __restrict__ dy) {

  double s = 77.27;
  double q = 8.375E-6;
  double w = 0.161;

  dy[0] = s * (y[0] - y[0] * y[1] + y[1] - (q * y[0]* y[0]));
  dy[1] = (y[2] - y[1] - y[0] * y[1]) / s;
  dy[2] = w * (y[0] - y[2]);

} // end dydt


#ifdef GENERATE_DOCS
}
#endif
