#ifndef JAC_HEAD_CVODES
#define JAC_HEAD_CVODES

#include "header.h"
#include "sundials/sundials_nvector.h"
#include "sundials/sundials_direct.h"
#include "nvector/nvector_serial.h"
#include "jacob.h"

int eval_jacob_cvodes(long int N, double t, N_Vector y, N_Vector ydot, DlsMat Jac, void* f, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3);

#endif
