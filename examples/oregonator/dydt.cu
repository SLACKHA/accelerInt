/**
 * \file
 * \brief A CUDA implementation of the Oregonator right hand side (y' = f(y)) function.
 *
 * Implements the CUDA evaluation of the derivative of the state vector with respect to time, \f$\dot{\vec{y}}\f$
 *
 */

#include "header.cuh"
#include "gpu_macros.cuh"

#ifdef GENERATE_DOCS
//put this in the Oregonator namespace for documentation
namespace oregonator_cu {
#endif

/**
 * \brief An implementation of the RHS of the Oregonator equation
 * \param[in]        t         The current system time
 * \param[in]        mu        Dummy parameter, needed for compatibility only
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

   double s = 77.27;
   double q = 8.375E-6;
   double w = 0.161;

   dy[INDEX(0)] = s * (y[INDEX(0)] - y[INDEX(0)] * y[INDEX(1)] + y[INDEX(1)] - (q * y[INDEX(0)]* y[INDEX(0)]));
   dy[INDEX(1)] = (y[INDEX(2)] - y[INDEX(1)] - y[INDEX(0)] * y[INDEX(1)]) / s;
   dy[INDEX(2)] = w * (y[INDEX(0)] - y[INDEX(2)]);

} // end dydt


#ifdef GENERATE_DOCS
}
#endif
