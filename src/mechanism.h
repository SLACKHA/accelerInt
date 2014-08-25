#ifndef MECHANISM_H
#define MECHANISM_H

#ifdef __cplusplus
extern "C" {
#endif

#include "header.h"
#include "mass_mole.h"
#include <stdio.h>
#include <time.h>
#include <sys/time.h>

//implemented on a per mechanism basis in mechanism.c
void set_same_initial_conditions(int NUM, double* y_host, double* pres_host, double* rho_host);

#ifdef __cplusplus
}
#endif

#endif