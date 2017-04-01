/**
 * \file
 * \brief Contains header definitions for the RHS function for the van der Pol example
 *
 */

#ifndef DYDT_H
#define DYDT_H

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
 */
void dydt (const double t, const double mu, const double * __restrict__ y, double * __restrict__ dy);

#ifdef GENERATE_DOCS
}
#endif

#endif