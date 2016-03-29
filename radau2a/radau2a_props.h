/*rb43_props.h
 *Various macros controlling behaviour of RADAU2A algorithm
 * \file RADAU2A_props.h
 * \author Nicholas Curtis
 * \date 03/10/2015
 */

#ifndef RADAU2A_PROPS_H
#define RADAU2A_PROPS_H

#include "header.h"
 
#define STRIDE (NSP)

#define EC_success (0)
#define EC_consecutive_steps (1)
#define EC_max_steps_exceeded (2)
#define EC_h_plus_t_equals_h (3)
#define EC_newton_max_iterations_exceeded (4)
 
#endif