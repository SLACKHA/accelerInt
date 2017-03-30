/**
 * \file
 * \brief Error checking for the CPU Radua-IIa solver
 */

#include "radau2a_props.cuh"

/*! \fn void check_error(int tid, int code)
	\brief Checks the return code of the given thread (IVP) for an error, and exits if found
	\param tid The thread (IVP) index
	\param code The return code of the thread
	@see ErrorCodes
 */
__host__
void check_error(int num_cond, int* codes)
{
	for (int tid = 0; tid < num_cond; ++tid)
	{
		int code = codes[tid];
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
			case EC_newton_max_iterations_exceeded :
				printf("During integration of ODE# %d, the allowed number of newton iteration steps was exceeded,"
					   "exiting...\n", tid);
				exit(code);
		}
	}
}
