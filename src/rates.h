#ifndef RATES_HEAD
#define RATES_HEAD

#include "head.h"

void eval_rxn_rates (const Real, const Real*, Real*, Real*);
void eval_spec_rates (const Real*, const Real*, const Real*, Real*);
void get_rxn_pres_mod (const Real, const Real, const Real*, Real*);

#endif
