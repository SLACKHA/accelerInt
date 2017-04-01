/**
 * \file
 * \brief Header for Matrix exponential (phi) methods
 */

#ifndef PHIA_HEAD_HESSENBERG
#define PHIA_HEAD_HESSENBERG

#include "header.h"

//void phiAv (const double*, const double, const double*, double*);
int phi2Ac_variable(const int, const double*, const double, double*);
int phiAc_variable(const int, const double*, const double, double*);
int expAc_variable(const int, const double*, const double, double*);

#endif