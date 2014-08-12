#ifndef DERIVS_HEAD
#define DERIVS_HEAD

#include "header.h"

void dydt (Real t, Real pres, Real* y, Real* dy);
void eval_jacob (Real t, Real pres, Real* y, Real* jac);
void eval_fd_jacob (const Real t, const Real pres, Real * y, Real * jac);

#endif