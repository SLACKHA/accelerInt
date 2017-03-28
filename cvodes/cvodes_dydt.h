/*! \file cvodes_dydt.h
    \brief Header file for CVODEs interface to RHS of ODEs

    This defines an interface to the right hand side of the ODEs to pass to CVODEs
*/
#ifndef DYDT_HEAD_CVODES
#define DYDT_HEAD_CVODES

#include "header.h"
#include "sundials/sundials_nvector.h"
#include "nvector/nvector_serial.h"


/*! \fn int dydt_cvodes(double t, N_Vector y, N_Vector ydot, void* f)
   \brief The CVODEs RHS interface
   \param t The current time of the system
   \param y The current state vector (in CVODE format)
   \param ydot The RHS vector to be populated (in CVODE format)
   \param f User data set during CVODEs setup (e.g. the system pressure)
   \return cvode_return_code The CVODE output constant returned (see sec B.2 of CVODE documentation), currently always returns CV_SUCCESS
*/

int dydt_cvodes(double t, N_Vector y, N_Vector ydot, void* f);

#endif
