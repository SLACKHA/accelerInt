#include "dydt.h"
#include "dydt_cvodes.h"

int dydt_cvodes(realtype t, N_Vector y, N_Vector ydot, void* f)
{
	double* local_y = NV_DATA_S(y);
	double* local_dy = NV_DATA_S(ydot);
	dydt((Real)t, *(Real*)f, local_y, local_dy);
	return 0;
}