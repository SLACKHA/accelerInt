#ifndef MECHANISM_H
#define MECHANISM_H

/** Number of species.
 * 0 H
 * 1 H2
 * 2 O
 * 3 OH
 * 4 H2O
 * 5 O2
 * 6 HO2
 * 7 H2O2
 * 8 N2
 * 9 AR
 * 10 HE
 * 11 CO
 * 12 CO2
 */
#define NSP 13
/** Number of variables. NN = NSP + 1 (temperature). */
#define NN 14

#ifdef __cplusplus
extern "C" {
#endif

//implemented on a per mechanism basis in mechanism.c
void set_same_initial_conditions(int NUM, double* y_host, double* pres_host, double* rho_host);

#ifdef __cplusplus
}
#endif

#endif