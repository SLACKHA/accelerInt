/**
 * \file
 * \brief Contains header definitions for the RHS function for the Oregonator example
 *
 */

#ifndef DYDT_H
#define DYDT_H

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
 */
void dydt (const double t, const double mu, const double * __restrict__ y, double * __restrict__ dy);

#ifdef GENERATE_DOCS
}
#endif

#endif
