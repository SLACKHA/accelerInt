/**
 * \file
 * \brief Contains a header definition for the IVP source term function
 *
 */

#ifndef JAC_H
#define JAC_H

#include "solver.h"

/**
 * \brief The expected signature of the jacobian function of the IVP
 * \param[in]        t         The current system time
 * \param[in]        param     The system parameter
 * \param[in]        y         The state vector
 * \param[out]       jac       The output Jacobian array
 * \param[in]        rwk       The working buffer for source rate evaluation
 */
void jacob (__global __ValueType const * t, __global __ValueType const * __restrict__ param,
            __global __ValueType const * __restrict__ y, __global __ValueType * __restrict__ jac,
            __global __ValueType* __restrict__ rwk);

#endif
