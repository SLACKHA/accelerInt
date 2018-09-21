/**
 * \file
 * \brief Contains a header definition for the IVP source term function
 *
 */

#ifndef DYDT_H
#define DYDT_H

/**
 * \brief The expected signature of the source term function of the IVP
 * \param[in]        t         The current system time
 * \param[in]        param     The system parameter
 * \param[in]        y         The state vector
 * \param[out]       dy        The output RHS (dydt) vector
 */
void dydt (const double t, const double param, const double * __restrict__ y, double * __restrict__ dy);

#endif
