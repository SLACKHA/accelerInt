/*solver_props.h
 *simple convencience file to include the correct solver properties file
 */

#ifndef SOLVER_PROPS_H
#define SOLVER_PROPS_H

#ifdef RB43
 	#include "exprb43_props.h"
#elif EXP4
 	#include "exp4_props.h"
#elif RADAU2A
 	#include "radau2a_props.h"
#endif

#endif