/*
* cvodes_jac.c
* A simple wrapper, allowing for use of the analytical jacobian w/ CVODES
*/

#include "cvodes_jac.h"

int eval_jacob_cvodes(long int N, Real t, N_Vector y, N_Vector ydot, DlsMat jac, void* f, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3) 
{
	double* local_y = NV_DATA_S(y);
	eval_jacob((Real)t, *(Real*)f, local_y, (Real*)jac->data);
	return 0;
}