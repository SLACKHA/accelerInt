#ifndef MECHANISM_H
#define MECHANISM_H

#include "header.h"
#include "mass_mole.h"
#include <stdio.h>
#include <time.h>
#include <sys/time.h>

//implemented on a per mechanism basis in mechanism.c
void set_same_initial_conditions(int NUM, double* y_host, double* pres_host, double* rho_host);

#endif