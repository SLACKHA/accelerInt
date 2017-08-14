/**
 * \file
 * \brief simple convenience file to include the correct solver properties file
 */

#ifndef SOLVER_PROPS_CUH
#define SOLVER_PROPS_CUH

#ifdef RB43
 	#include "exprb43_props.cuh"
#elif EXP4
 	#include "exp4_props.cuh"
#elif RADAU2A
 	#include "radau2a_props.cuh"
#elif RKC
    #include "rkc_props.cuh"
#else
 	struct solver_memory {};
#endif

#ifdef GENERATE_DOCS
namespace genericcu {
#endif

__host__
void check_error(int, int*);

#ifdef GENERATE_DOCS
namespace }
#endif

#endif