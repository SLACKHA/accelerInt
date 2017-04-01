/**
 * \file
 * \brief A CUDA implementation of the van der Pol right hand side (y' = f(y)) function.
 *
 * Implements the CUDA evaluation of the derivative of the state vector with respect to time, \f$\dot{\vec{y}}\f$
 *
 */

#include "header.cuh"
#include "gpu_macros.cuh"

#ifdef GENERATE_DOCS
//put this in the van der Pol namespace for documentation
namespace van_der_pol_cu {
#endif

/**
 * \brief An implementation of the RHS of the van der Pol equation
 * \param[in]        t         The current system time
 * \param[in]        mu        The van der Pol parameter
 * \param[in]        y         The state vector
 * \param[out]       dy        The output RHS (dydt) vector
 * \param[in]        d_mem     The mechanism_memory struct.  In future versions, this will be used to access the \f$\mu\f$ parameter to have a consistent interface.
 *
 * The `y` and `dy` vectors supplied here are the global state vectors.
 * Hence the vector accesses must be done with the global thread ID.
 * The gpu_macros.cuh file defines some useful macros to simplify indexing
 * @see gpu_macros.cuh
 */
 __device__
void dydt (const double t, const double mu, const double * __restrict__ y, double * __restrict__ dy,
           const mechanism_memory * __restrict__ d_mem) {

  // y1' = y2
  dy[INDEX(0)] = y[INDEX(1)];
  // y2' = mu(1 - y1^2)y2 - y1
  dy[INDEX(1)] = mu * (1 - y[INDEX(0)] * y[INDEX(0)]) * y[INDEX(1)] - y[INDEX(0)];

} // end dydt


#ifdef GENERATE_DOCS
}
#endif