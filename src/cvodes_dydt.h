#ifndef DYDT_HEAD_CVODES
#define DYDT_HEAD_CVODES

#include "header.h"
#include "sundials/sundials_nvector.h"
#include "nvector/nvector_serial.h"

int dydt_cvodes(Real t, N_Vector y, N_Vector ydot, void* f);

#endif
