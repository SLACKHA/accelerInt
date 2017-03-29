/*!
 * \file exprb43_props.h
 * \brief Various macros controlling behaviour of EXPRB43 algorithm
 * \author Nicholas Curtis
 * \date 03/10/2015
 */

#ifndef RB43_PROPS_H
#define RB43_PROPS_H

#include "header.h"
#include <stdio.h>

#ifdef GENERATE_DOCS
namespace exprb43 {
#endif


//if defined, uses (I - h * Hm)^-1 to smooth the krylov error vector
//#define USE_SMOOTHED_ERROR
 //! max order of the phi functions (for error estimation)
#define P 4
//! order of embedded methods
#define ORD 3.0
//! maximum Krylov dimension (without phi order)
#define M_MAX NSP
//! Krylov matrix stride
#define STRIDE (M_MAX + P)
//! Maximum allowed internal timesteps per integration step
#define MAX_STEPS (100000)
//! Number of consecutive errors on internal integration steps allowed before exit
#define MAX_CONSECUTIVE_ERRORS (5)

/**
 * \addtogroup ErrorCodes Return codes of Integrators
 * @{
 */
/**
 * \defgroup exprb43_ErrCodes Return codes of EXPRB43 integrator
 * @{
 */

//! Successful integration step
#define EC_success (0)
//! Maximum consecutive errors on internal integration steps reached
#define EC_consecutive_steps (1)
//! Maximum number of internal integration steps reached
#define EC_max_steps_exceeded (2)
//! Timestep reduced such that update would have no effect on simulation time
#define EC_h_plus_t_equals_h (3)

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