/**
 * \file
 * \brief simple convenience file to include the correct solver properties file
 */

#ifndef SOLVER_PROPS_H
#define SOLVER_PROPS_H

#ifdef RB43
 	#include "exprb43_props.h"
#elif EXP4
 	#include "exp4_props.h"
#elif RADAU2A
 	#include "radau2a_props.h"
#elif RKC
    #include "rkc_props.h"
#endif

#ifdef GENERATE_DOCS
namespace generic {
#endif

void check_error(int, int);

#ifdef GENERATE_DOCS
}
#endif

#endif