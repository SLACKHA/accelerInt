/*!
* \file cvodes_jac.c
* \brief A simple wrapper, allowing for use of the analytical jacobian w/ CVODES
*/

#include "cvodes_jac.h"

/*!
This function converts the N_Vectors `y` and `y_dot` to simple double pointers, the user data `f` to a double
and outputs the Jacobian supplied by `eval_jacob` to the CVODE jacobian `jac`
Currently, CV_SUCCESS is always returned.
*/
int eval_jacob_cvodes(
    #ifdef have_problem_size
    long int N,
    #endif
    double t, N_Vector y, N_Vector ydot, jac_type jac,
                      void* f, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3)
{
    // 'use' parameters to silence gcc warnings
    (void)N;
    (void)ydot;
    (void)tmp1;
    (void)tmp2;
    (void)tmp3;

	double* local_y = NV_DATA_S(y);
    // unpack user data
    CVUserData* ud = (CVUserData*)f;
	eval_jacob((double)t, ud->param, local_y, (double*)jac->data, ud->rwk);
	return 0;
}
