/**
 * \file
 * \brief Error checking for the EXP4 algorithm
 * \author Nicholas Curtis
 * \date 03/10/2015
 */

#include "exp4_props.cuh"

#ifdef GENERATE_DOCS
namespace exp4cu {
#endif


/*! \fn void check_error(int tid, int code)
	\brief Checks the return code of the given thread (IVP) for an error, and exits if found
	\param num_cond The total number of IVPs to check
	\param codes The array of return codes
	@see exp4cu_ErrCodes
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
		}
	}
}

#ifdef GENERATE_DOCS
}
#endif