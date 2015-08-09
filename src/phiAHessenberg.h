#ifndef PHIA_HEAD_HESSENBERG
#define PHIA_HEAD_HESSENBERG

#include "header.h"

//void phiAv (const double*, const double, const double*, double*);
void phi2Ac_variable(const int, const int, const double*, const double, double*);
void phiAc_variable(const int, const int, const double*, const double, double*);
void expAc_variable(const int, const int, const double*, const double, double*);

#endif