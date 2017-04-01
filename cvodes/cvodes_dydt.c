/**
 * \file
 * \brief CVODEs Wrapper for the RHS function
 */

#include "dydt.h"
#include "cvodes_dydt.h"

/*!
    This method translates the N_Vector y and ydot's to simple double pointers
    that may be passed to the internal dydt function.
    Additionally, the user data (f) is cast to a double as supplied as the 'Pressure'
    parameter for dydt.
*/

int dydt_cvodes(realtype t, N_Vector y, N_Vector ydot, void* f)
{
	double* local_y = NV_DATA_S(y);
	double* local_dy = NV_DATA_S(ydot);
	dydt((double)t, *(double*)f, local_y, local_dy);
	return 0;
}