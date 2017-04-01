/**
 * \file
 * \brief Contains error checking for EXP4 return codes
 *
 * \author Nicholas J. Curtis
 * \date 09/02/2014
 *
 */

#include "exp4_props.h"

#ifdef GENERATE_DOCS
namespace exp4 {
#endif


/*! \fn void check_error(int tid, int code)
	\brief Checks the return code of the given thread (IVP) for an error, and exits if found
	\param tid The thread (IVP) index
	\param code The return code of the thread
	@see exp4_ErrCodes
 */
void check_error(int tid, int code)
{
	switch(code)
	{
		case EC_consecutive_steps :
			printf("During integration of ODE# %d, an error occured on too many consecutive integration steps,"
				    "exiting...\n", tid);
			exit(code);
		case EC_max_steps_exceeded :
			printf("During integration of ODE# %d, the allowed number of integration steps was exceeded,"
				    "exiting...\n", tid);
			exit(code);
		case EC_h_plus_t_equals_h :
			printf("During integration of ODE# %d, the stepsize 'h' was decreased such that h = t + h,"
				    "exiting...\n", tid);
			exit(code);
	}
}

#ifdef GENERATE_DOCS
}
#endif