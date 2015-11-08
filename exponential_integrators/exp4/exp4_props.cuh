/*exp4_props.cuh
 *Various macros controlling behaviour of EXP4 algorithm
 * \file exp4_props.cuh
 * \author Nicholas Curtis
 * \date 03/10/2015
 */

#ifndef EXP4_PROPS_CUH
#define EXP4_PROPS_CUH

#include "header.cuh"
 
//if defined, uses (I - h * Hm)^-1 to smooth the krylov error vector
//#define USE_SMOOTHED_ERROR
//max order of the phi functions (i.e. for error estimation)
#define P 1
//order of embedded methods
#define ORD 3.0
#define M_MAX NSP
#define STRIDE (M_MAX + P)

#endif