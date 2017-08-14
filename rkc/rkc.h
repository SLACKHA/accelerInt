#ifndef RKC_HEAD
#define RKC_HEAD

#include "header.h"
#include "rkc_props.h"

//Real rkc_spec_rad (const Real, const Real, const Real*, const Real*, Real*, Real*);
Real rkc_spec_rad (const Real, const Real, const Real, const Real*, const Real*, Real*, Real*);
void rkc_step (const Real, const Real, const Real, const Real*, const Real*, const int, Real*);
void rkc_driver (Real, const Real, const Real, int, Real*, Real*);

#endif