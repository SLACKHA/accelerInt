#ifndef RATES_HEAD
#define RATES_HEAD

#include "head.h"

__device__ void eval_rxn_rates (const Real, const Real*, Real*);
__device__ void eval_spec_rates (const Real*, const Real*, Real*);
__device__ void get_rxn_pres_mod (const Real, const Real, const Real*, Real*);

#endif
