#ifndef CHEM_UTILS_HEAD
#define CHEM_UTILS_HEAD

#include "head.h"

__device__ void eval_h (const Real, Real*);
__device__ void eval_u (const Real, Real*);
__device__ void eval_cv (const Real, Real*);
__device__ void eval_cp (const Real, Real*);

#endif
