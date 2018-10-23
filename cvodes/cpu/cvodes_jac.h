/*!
* \file cvodes_jac.h
* \brief A simple wrapper, allowing for use of the analytical jacobian w/ CVODES
*/


#ifndef JAC_HEAD_CVODES
#define JAC_HEAD_CVODES

#include "jacob.h"
#include "cvodes_includes.h"


#if SUNDIALS_MAJOR_VERSION >= 3
#define jac_type SUNMatrix
#else
#define jac_type DlsMat
#define have_problem_size
#endif

/*!
   \brief The CVODEs Jacobian interface for a direct dense Jacobian
   \param N the problem size
   \param t The current time of the system
   \param y The current state vector (in CVODE format)
   \param ydot The RHS vector to be populated (in CVODE format)
   \param jac The Jacobian matrix (in CVODE format) to output to
   \param f User data set during CVODEs setup (e.g. the system pressure)
   \param tmp1 Temporary storage used by CVODEs
   \param tmp2 Temporary storage used by CVODEs
   \param tmp3 Temporary storage used by CVODEs
   \return cvode_return_code The CVODE output constant returned (see sec B.2 of CVODE documentation), currently always returns CV_SUCCESS
*/

int eval_jacob_cvodes(
   #ifdef have_problem_size
   long int N,
   #endif
   double t, N_Vector y, N_Vector ydot, jac_type Jac, void* f, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3);

#endif
