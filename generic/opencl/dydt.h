/**
 * \file
 * \brief Contains a header definition for the IVP source term function
 *
 */

#ifndef DYDT_H
#define DYDT_H

#include "solver.h"

/**
 * \brief The expected signature of the source term function of the IVP
 * \param[in]        t         The current system time
 * \param[in]        param     The system parameter
 * \param[in]        y         The state vector
 * \param[out]       dy        The output RHS (dydt) vector
 */
void dydt (const __ValueType* t, const __ValueType* param, const __ValueType * __restrict__ y, __ValueType * __restrict__ dy);

#endif
