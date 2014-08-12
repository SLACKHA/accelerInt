#ifndef DERIVS_HEAD_DEVICE
#define DERIVS_HEAD_DEVICE

#include "header.h"

__device__ void dydt (const Real, const Real, const Real*, Real*);
__device__ void eval_jacob (const Real, const Real, const Real*, Real*);
__device__ void eval_fd_jacob (const Real, const Real, Real *, Real *);

#endif