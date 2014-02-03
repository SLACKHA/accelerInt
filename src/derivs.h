#ifndef DERIVS_HEAD
#define DERIVS_HEAD

#include "head.h"

void dydt (Real t, Real pres, Real* y, Real* dy);
void eval_jacob (Real t, Real pres, Real* y, Real* jac);

#endif