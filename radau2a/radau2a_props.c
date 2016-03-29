#include "radau2a_props.h"

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
		case EC_newton_max_iterations_exceeded :
			printf("During integration of ODE# %d, the allowed number of newton iteration steps was exceeded,"
				    "exiting...\n", tid);
			exit(code);
	}
}