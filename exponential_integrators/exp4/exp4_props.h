/*rb43_props.h
 *Various macros controlling behaviour of RB43 algorithm
 * \file rb43_props.h
 * \author Nicholas Curtis
 * \date 03/10/2015
 */

#ifndef EXP4_PROPS_H
#define EXP4_PROPS_H

#include "header.h"
#include "solver_options.h"
#include "solver_props.h"

//if defined, uses (I - h * Hm)^-1 to smooth the krylov error vector
//#define USE_SMOOTHED_ERROR
//max order of the phi functions (i.e. for error estimation)
#define P 1
//order of embedded methods
#define ORD 3.0
#define M_MAX NSP
#define STRIDE (M_MAX + P)

#endif