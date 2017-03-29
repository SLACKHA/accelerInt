/**
 * \file
 * \brief Various macros controlling behaviour of RADAU2A algorithm
 * \author Nicholas Curtis
 * \date 03/10/2015
 */

#ifndef RADAU2A_PROPS_H
#define RADAU2A_PROPS_H

#include "header.h"
#include <stdio.h>

#ifdef GENERATE_DOCS
namespace radau2a {
#endif

//! the matrix dimensions
#define STRIDE (NSP)

/**
 * \addtogroup ErrorCodes Return codes of Integrators
 * @{
 */
/**
 * \defgroup RK_ErrCodes Return codes of Radau-IIa integrator
 * @{
 */

//! Successful time step
#define EC_success (0)
//! Maximum number of consecutive internal timesteps with error reached @see #Max_consecutive_errs
#define EC_consecutive_steps (1)
//! Maximum number of internal timesteps exceeded @see #Max_no_steps
#define EC_max_steps_exceeded (2)
//! Timescale reduced such that t + h == t in floating point math
#define EC_h_plus_t_equals_h (3)
//! Maximum allowed Newton Iteration steps exceeded @see #NewtonMaxit
#define EC_newton_max_iterations_exceeded (4)

/**
 * @}
 */
/**
 * @}
 */

#ifdef GENERATE_DOCS
}
#endif

#endif