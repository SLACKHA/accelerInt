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
 * \param[in]        rwk       The working buffer for source rate evaluation
 */
void dydt (__private __ValueType const t, __global __ValueType const * __restrict__ param,
           __global __ValueType const * __restrict__ y, __global __ValueType * __restrict__ dy,
           __global __ValueType* __restrict__ rwk);

#endif
