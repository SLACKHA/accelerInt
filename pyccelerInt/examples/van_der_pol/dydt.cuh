/**
 * \file
 * \brief Contains header definitions for the CUDA RHS function for the van der Pol example
 *
 */

#ifndef DYDT_CUH
#define DYDT_CUH

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
 */
__device__
void dydt (const double t, const double mu, const double * __restrict__ y, double * __restrict__ dy,
           const mechanism_memory * __restrict__ d_mem);

#ifdef GENERATE_DOCS
}
#endif

#endif