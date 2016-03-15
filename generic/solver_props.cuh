/*solver_props.cuh
 *simple convencience file to include the correct solver properties file
 */

#ifndef SOLVER_PROPS_CUH
#define SOLVER_PROPS_CUH

#ifdef RB43
 	#include "exprb43_props.cuh"
#elif EXP4
 	#include "exp4_props.cuh"
#elif RADAU2A
 	#include "radau2a_props.cuh"
#else
 	struct solver_memory {};
#endif

#endif