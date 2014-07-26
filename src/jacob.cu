#include <math.h>
#include "head.h"
#include "chem_utils.cuh"
#include "rates.cuh"

__device__ void eval_jacob (const Real t, const Real pres, const Real * y, Real * jac) {

  Real T = y[0];

  // average molecular weight
  Real mw_avg;
  mw_avg = (y[1] / 1.00797) + (y[2] / 2.01594) + (y[3] / 15.9994) + (y[4] / 17.00737)
      + (y[5] / 18.01534) + (y[6] / 31.9988) + (y[7] / 33.00677) + (y[8] / 34.01474)
      + (y[9] / 28.0134) + (y[10] / 39.948) + (y[11] / 4.0026) + (y[12] / 28.01055)
      + (y[13] / 44.00995);
  mw_avg = 1.0 / mw_avg;
  // mass-averaged density
  Real rho;
  rho = pres * mw_avg / (8.31451000e+07 * T);

  // species molar concentrations
  Real conc[13];
  conc[0] = rho * y[1] / 1.00797;
  conc[1] = rho * y[2] / 2.01594;
  conc[2] = rho * y[3] / 15.9994;
  conc[3] = rho * y[4] / 17.00737;
  conc[4] = rho * y[5] / 18.01534;
  conc[5] = rho * y[6] / 31.9988;
  conc[6] = rho * y[7] / 33.00677;
  conc[7] = rho * y[8] / 34.01474;
  conc[8] = rho * y[9] / 28.0134;
  conc[9] = rho * y[10] / 39.948;
  conc[10] = rho * y[11] / 4.0026;
  conc[11] = rho * y[12] / 28.01055;
  conc[12] = rho * y[13] / 44.00995;

  // evaluate reaction rates
  Real fwd_rxn_rates[54];
  eval_rxn_rates (T, conc, fwd_rxn_rates);

  // get pressure modifications to reaction rates
  Real pres_mod[12];
  get_rxn_pres_mod (T, pres, conc, pres_mod);

  // evaluate rate of change of species molar concentration
  Real sp_rates[13];
  eval_spec_rates (fwd_rxn_rates, pres_mod, sp_rates);

  register Real m = pres / (8.314510e+07 * T);
  register Real logT = log(T);
  register Real Pr;
  Real Fcent, A, B, lnF_AB;
  //partial of omega_H wrt T;
  jac[1] = -1.0 * ((1.0 / T) * (fwd_rxn_rates[0] * ((7.69216995e+03 / T) + 1.0 - 2.0)));
  jac[1] += ((1.0 / T) * (fwd_rxn_rates[1] * (4.04780000e-01 + (-7.48736077e+02 / T) + 1.0 - 2.0)));
  jac[1] += ((1.0 / T) * (fwd_rxn_rates[2] * ((3.99956606e+03 / T) + 1.0 - 2.0)));
  jac[1] += -1.0 * ((1.0 / T) * (fwd_rxn_rates[3] * (-5.35330000e-02 + (3.32706731e+03 / T) + 1.0 - 2.0)));
  jac[1] += ((1.0 / T) * (fwd_rxn_rates[4] * ((9.64666348e+03 / T) + 1.0 - 2.0)));
  jac[1] += -1.0 * ((1.0 / T) * (fwd_rxn_rates[5] * (-5.35330000e-02 + (8.97436602e+03 / T) + 1.0 - 2.0)));
  jac[1] += ((1.0 / T) * (fwd_rxn_rates[6] * (1.51000000e+00 + (1.72603316e+03 / T) + 1.0 - 2.0)));
  jac[1] += -1.0 * ((1.0 / T) * (fwd_rxn_rates[7] * (1.18290000e+00 + (9.55507805e+03 / T) + 1.0 - 2.0)));
  jac[1] += 2.0 * ((-pres_mod[0] * fwd_rxn_rates[10] / T) + (pres_mod[0] / T) * (fwd_rxn_rates[10] * (-1.40000000e+00 + (5.25257556e+04 / T) + 1.0 - 1.0)));
  jac[1] += -2.0 * ((-pres_mod[1] * fwd_rxn_rates[11] / T) + (pres_mod[1] / T) * (fwd_rxn_rates[11] * (-1.42340000e+00 + (2.21510944e+01 / T) + 1.0 - 2.0)));
  jac[1] += 2.0 * ((1.0 / T) * (fwd_rxn_rates[12] * (-1.10000000e+00 + (5.25257556e+04 / T) + 1.0 - 2.0)));
  jac[1] += -2.0 * ((1.0 / T) * (fwd_rxn_rates[13] * (-1.12340000e+00 + (2.21510944e+01 / T) + 1.0 - 3.0)));
  jac[1] += 2.0 * ((1.0 / T) * (fwd_rxn_rates[14] * (-1.10000000e+00 + (5.25257556e+04 / T) + 1.0 - 2.0)));
  jac[1] += -2.0 * ((1.0 / T) * (fwd_rxn_rates[15] * (-1.12340000e+00 + (2.21510944e+01 / T) + 1.0 - 3.0)));
  jac[1] += -1.0 * ((-pres_mod[4] * fwd_rxn_rates[22] / T) + (pres_mod[4] / T) * (fwd_rxn_rates[22] * (-1.00000000e+00 + 1.0 - 2.0)));
  jac[1] += ((-pres_mod[5] * fwd_rxn_rates[23] / T) + (pres_mod[5] / T) * (fwd_rxn_rates[23] * (-1.03010000e+00 + (5.18313166e+04 / T) + 1.0 - 1.0)));
  jac[1] += ((-pres_mod[6] * fwd_rxn_rates[24] / T) + (pres_mod[6] / T) * (fwd_rxn_rates[24] * (-3.32200000e+00 + (6.07835411e+04 / T) + 1.0 - 1.0)));
  jac[1] += -1.0 * ((-pres_mod[7] * fwd_rxn_rates[25] / T) + (pres_mod[7] / T) * (fwd_rxn_rates[25] * (-3.01830000e+00 + (4.50771425e+02 / T) + 1.0 - 2.0)));
  jac[1] += ((1.0 / T) * (fwd_rxn_rates[26] * (-2.44000000e+00 + (6.04765789e+04 / T) + 1.0 - 2.0)));
  jac[1] += -1.0 * ((1.0 / T) * (fwd_rxn_rates[27] * (-2.13630000e+00 + (1.43809259e+02 / T) + 1.0 - 3.0)));
  Pr = (m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * (exp(1.87346128e+01 - 2.16 * logT - (2.64088106e+02 / T)));
  Fcent = 5.00000000e-01 * exp(-T / 1.00000000e-30) + 5.00000000e-01 * exp(T / 1.00000000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  lnF_AB = 2.0 * log(Fcent) * A / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)));
  jac[1] += -1.0 * (pres_mod[8]* (((-2.1600e+00 + (2.64088106e+02 / T) - 1.0) / (T * (1.0 + Pr))) + (((1.0 / (Fcent * (1.0 + A * A / (B * B)))) - lnF_AB * (-2.90977303e-01 * B + 5.10817170e-01 * A) / Fcent) * (-5.00000000e+29 * exp(-T / 1.00000000e-30) - 5.00000000e-31 * exp(-T / 1.00000000e+30))) - lnF_AB * (4.34294482e-01 * B + 6.08012275e-02 * A) * (-2.16000000e+00 + (2.64088106e+02 / T) - 1.0) / T) * fwd_rxn_rates[28] + (pres_mod[8] / T) * (fwd_rxn_rates[28] * (4.40000000e-01 + 1.0 - 2.0)));
  Pr = (m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * (exp(1.87345962e+01 - 2.159967 * logT - (2.64188750e+02 / T)));
  Fcent = 5.00000000e-01 * exp(-T / 1.00000000e-30) + 5.00000000e-01 * exp(T / 1.00000000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  lnF_AB = 2.0 * log(Fcent) * A / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)));
  jac[1] += (pres_mod[9]* (((-2.1600e+00 + (2.64188750e+02 / T) - 1.0) / (T * (1.0 + Pr))) + (((1.0 / (Fcent * (1.0 + A * A / (B * B)))) - lnF_AB * (-2.90977303e-01 * B + 5.10817170e-01 * A) / Fcent) * (-5.00000000e+29 * exp(-T / 1.00000000e-30) - 5.00000000e-31 * exp(-T / 1.00000000e+30))) - lnF_AB * (4.34294482e-01 * B + 6.08012275e-02 * A) * (-2.15996700e+00 + (2.64188750e+02 / T) - 1.0) / T) * fwd_rxn_rates[29] + (pres_mod[9] / T) * (fwd_rxn_rates[29] * (-4.94330000e-02 + (2.52182000e+04 / T) + 1.0 - 1.0)));
  jac[1] += -1.0 * ((1.0 / T) * (fwd_rxn_rates[30] * (2.09000000e+00 + (-7.30167382e+02 / T) + 1.0 - 2.0)));
  jac[1] += ((1.0 / T) * (fwd_rxn_rates[31] * (2.60280000e+00 + (2.65552467e+04 / T) + 1.0 - 2.0)));
  jac[1] += -1.0 * ((1.0 / T) * (fwd_rxn_rates[32] * ((1.48448916e+02 / T) + 1.0 - 2.0)));
  jac[1] += ((1.0 / T) * (fwd_rxn_rates[33] * (8.64090000e-01 + (1.83206092e+04 / T) + 1.0 - 2.0)));
  jac[1] += -1.0 * ((1.0 / T) * (fwd_rxn_rates[44] * ((1.99777016e+03 / T) + 1.0 - 2.0)));
  jac[1] += ((1.0 / T) * (fwd_rxn_rates[45] * (1.28430000e+00 + (3.59925720e+04 / T) + 1.0 - 2.0)));
  jac[1] += -1.0 * ((1.0 / T) * (fwd_rxn_rates[46] * ((4.00057249e+03 / T) + 1.0 - 2.0)));
  jac[1] += ((1.0 / T) * (fwd_rxn_rates[47] * (7.47310000e-01 + (1.19941692e+04 / T) + 1.0 - 2.0)));
  jac[1] *= 1.00797000e+00 / rho;

  //partial of omega_H wrt Y_H;
  jac[15] = -1.0 * ((-mw_avg * fwd_rxn_rates[0] / 1.00797000e+00 + exp(3.22754120e+01 - (7.69216995e+03 / T)) * (rho / 1.00797000e+00) * conc[5]) + (-mw_avg * fwd_rxn_rates[0] / 1.00797000e+00));
  jac[15] += ((-mw_avg * fwd_rxn_rates[1] / 1.00797000e+00) + (-mw_avg * fwd_rxn_rates[1] / 1.00797000e+00));
  jac[15] += ((-mw_avg * fwd_rxn_rates[2] / 1.00797000e+00) + (-mw_avg * fwd_rxn_rates[2] / 1.00797000e+00));
  jac[15] += -1.0 * ((-mw_avg * fwd_rxn_rates[3] / 1.00797000e+00 + exp(2.85866095e+01 - 0.053533 * logT - (3.32706731e+03 / T)) * (rho / 1.00797000e+00) * conc[3]) + (-mw_avg * fwd_rxn_rates[3] / 1.00797000e+00));
  jac[15] += ((-mw_avg * fwd_rxn_rates[4] / 1.00797000e+00) + (-mw_avg * fwd_rxn_rates[4] / 1.00797000e+00));
  jac[15] += -1.0 * ((-mw_avg * fwd_rxn_rates[5] / 1.00797000e+00 + exp(3.40258987e+01 - 0.053533 * logT - (8.97436602e+03 / T)) * (rho / 1.00797000e+00) * conc[3]) + (-mw_avg * fwd_rxn_rates[5] / 1.00797000e+00));
  jac[15] += ((-mw_avg * fwd_rxn_rates[6] / 1.00797000e+00) + (-mw_avg * fwd_rxn_rates[6] / 1.00797000e+00));
  jac[15] += -1.0 * ((-mw_avg * fwd_rxn_rates[7] / 1.00797000e+00 + exp(2.34232839e+01 + 1.1829 * logT - (9.55507805e+03 / T)) * (rho / 1.00797000e+00) * conc[4]) + (-mw_avg * fwd_rxn_rates[7] / 1.00797000e+00));
  jac[15] += 2.0 * ((-mw_avg * pres_mod[0] / 1.00797000e+00 + rho / 1.00797000e+00) * fwd_rxn_rates[10] + pres_mod[0] * ((-mw_avg * fwd_rxn_rates[10] / 1.00797000e+00)));
  jac[15] += -2.0 * ((-mw_avg * pres_mod[1] / 1.00797000e+00 + rho / 1.00797000e+00) * fwd_rxn_rates[11] + pres_mod[1] * (2.0 * (-mw_avg * fwd_rxn_rates[11] / 1.00797000e+00 + exp(4.40606374e+01 - 1.4234 * logT - (2.21510944e+01 / T)) * (rho / 1.00797000e+00) * conc[0])));
  jac[15] += 2.0 * ((-mw_avg * fwd_rxn_rates[12] / 1.00797000e+00) + (-mw_avg * fwd_rxn_rates[12] / 1.00797000e+00));
  jac[15] += -2.0 * (2.0 * (-mw_avg * fwd_rxn_rates[13] / 1.00797000e+00 + exp(4.20017378e+01 - 1.1234 * logT - (2.21510944e+01 / T)) * (rho / 1.00797000e+00) * conc[0] * conc[9]) + (-mw_avg * fwd_rxn_rates[13] / 1.00797000e+00));
  jac[15] += 2.0 * ((-mw_avg * fwd_rxn_rates[14] / 1.00797000e+00) + (-mw_avg * fwd_rxn_rates[14] / 1.00797000e+00));
  jac[15] += -2.0 * (2.0 * (-mw_avg * fwd_rxn_rates[15] / 1.00797000e+00 + exp(4.20017378e+01 - 1.1234 * logT - (2.21510944e+01 / T)) * (rho / 1.00797000e+00) * conc[0] * conc[10]) + (-mw_avg * fwd_rxn_rates[15] / 1.00797000e+00));
  jac[15] += -1.0 * ((-mw_avg * pres_mod[4] / 1.00797000e+00 + rho / 1.00797000e+00) * fwd_rxn_rates[22] + pres_mod[4] * ((-mw_avg * fwd_rxn_rates[22] / 1.00797000e+00 + exp(4.29970685e+01 - 1.0 * logT) * (rho / 1.00797000e+00) * conc[2]) + (-mw_avg * fwd_rxn_rates[22] / 1.00797000e+00)));
  jac[15] += ((-mw_avg * pres_mod[5] / 1.00797000e+00 + rho / 1.00797000e+00) * fwd_rxn_rates[23] + pres_mod[5] * ((-mw_avg * fwd_rxn_rates[23] / 1.00797000e+00)));
  jac[15] += ((-mw_avg * pres_mod[6] / 1.00797000e+00 + rho / 1.00797000e+00) * fwd_rxn_rates[24] + pres_mod[6] * ((-mw_avg * fwd_rxn_rates[24] / 1.00797000e+00)));
  jac[15] += -1.0 * ((-mw_avg * pres_mod[7] / 1.00797000e+00 + rho / 1.00797000e+00) * fwd_rxn_rates[25] + pres_mod[7] * ((-mw_avg * fwd_rxn_rates[25] / 1.00797000e+00 + exp(5.85301272e+01 - 3.0183 * logT - (4.50771425e+02 / T)) * (rho / 1.00797000e+00) * conc[3]) + (-mw_avg * fwd_rxn_rates[25] / 1.00797000e+00)));
  jac[15] += (2.0 * (-mw_avg * fwd_rxn_rates[26] / 1.00797000e+00));
  jac[15] += -1.0 * ((-mw_avg * fwd_rxn_rates[27] / 1.00797000e+00 + exp(5.44311720e+01 - 2.1363 * logT - (1.43809259e+02 / T)) * (rho / 1.00797000e+00) * conc[3] * conc[4]) + (-mw_avg * fwd_rxn_rates[27] / 1.00797000e+00) + (-mw_avg * fwd_rxn_rates[27] / 1.00797000e+00));
  Pr = (m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * (exp(1.87346128e+01 - 2.16 * logT - (2.64088106e+02 / T)));
  Fcent = 5.0000e-01 * exp(-T / 1.0000e-30) + 5.0000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[15] += -1.0 * (pres_mod[8] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 1.00797000e+00 + rho / ((m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * 1.00797000e+00)) * fwd_rxn_rates[28] + pres_mod[8] * ((-mw_avg * fwd_rxn_rates[28] / 1.00797000e+00 + exp(2.91680604e+01 + 0.44 * logT) * (rho / 1.00797000e+00) * conc[5]) + (-mw_avg * fwd_rxn_rates[28] / 1.00797000e+00)));
  Pr = (m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * (exp(1.87345962e+01 - 2.159967 * logT - (2.64188750e+02 / T)));
  Fcent = 5.0000e-01 * exp(-T / 1.0000e-30) + 5.0000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[15] += (pres_mod[9] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 1.00797000e+00 + rho / ((m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * 1.00797000e+00)) * fwd_rxn_rates[29] + pres_mod[9] * ((-mw_avg * fwd_rxn_rates[29] / 1.00797000e+00)));
  jac[15] += -1.0 * ((-mw_avg * fwd_rxn_rates[30] / 1.00797000e+00 + exp(1.48271115e+01 + 2.09 * logT - (-7.30167382e+02 / T)) * (rho / 1.00797000e+00) * conc[6]) + (-mw_avg * fwd_rxn_rates[30] / 1.00797000e+00));
  jac[15] += ((-mw_avg * fwd_rxn_rates[31] / 1.00797000e+00) + (-mw_avg * fwd_rxn_rates[31] / 1.00797000e+00));
  jac[15] += -1.0 * ((-mw_avg * fwd_rxn_rates[32] / 1.00797000e+00 + exp(3.18907389e+01 - (1.48448916e+02 / T)) * (rho / 1.00797000e+00) * conc[6]) + (-mw_avg * fwd_rxn_rates[32] / 1.00797000e+00));
  jac[15] += (2.0 * (-mw_avg * fwd_rxn_rates[33] / 1.00797000e+00));
  jac[15] += -1.0 * ((-mw_avg * fwd_rxn_rates[44] / 1.00797000e+00 + exp(3.08132330e+01 - (1.99777016e+03 / T)) * (rho / 1.00797000e+00) * conc[7]) + (-mw_avg * fwd_rxn_rates[44] / 1.00797000e+00));
  jac[15] += ((-mw_avg * fwd_rxn_rates[45] / 1.00797000e+00) + (-mw_avg * fwd_rxn_rates[45] / 1.00797000e+00));
  jac[15] += -1.0 * ((-mw_avg * fwd_rxn_rates[46] / 1.00797000e+00 + exp(3.15063801e+01 - (4.00057249e+03 / T)) * (rho / 1.00797000e+00) * conc[7]) + (-mw_avg * fwd_rxn_rates[46] / 1.00797000e+00));
  jac[15] += ((-mw_avg * fwd_rxn_rates[47] / 1.00797000e+00) + (-mw_avg * fwd_rxn_rates[47] / 1.00797000e+00));
  jac[15] += sp_rates[0] * mw_avg / 1.00797000e+00;
  jac[15] *= 1.00797000e+00 / rho;

  //partial of omega_H wrt Y_H2;
  jac[29] = -1.0 * ((-mw_avg * fwd_rxn_rates[0] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[0] / 2.01594000e+00));
  jac[29] += ((-mw_avg * fwd_rxn_rates[1] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[1] / 2.01594000e+00));
  jac[29] += ((-mw_avg * fwd_rxn_rates[2] / 2.01594000e+00 + exp(2.89707478e+01 - (3.99956606e+03 / T)) * (rho / 2.01594000e+00) * conc[2]) + (-mw_avg * fwd_rxn_rates[2] / 2.01594000e+00));
  jac[29] += -1.0 * ((-mw_avg * fwd_rxn_rates[3] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[3] / 2.01594000e+00));
  jac[29] += ((-mw_avg * fwd_rxn_rates[4] / 2.01594000e+00 + exp(3.44100335e+01 - (9.64666348e+03 / T)) * (rho / 2.01594000e+00) * conc[2]) + (-mw_avg * fwd_rxn_rates[4] / 2.01594000e+00));
  jac[29] += -1.0 * ((-mw_avg * fwd_rxn_rates[5] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[5] / 2.01594000e+00));
  jac[29] += ((-mw_avg * fwd_rxn_rates[6] / 2.01594000e+00 + exp(1.91907890e+01 + 1.51 * logT - (1.72603316e+03 / T)) * (rho / 2.01594000e+00) * conc[3]) + (-mw_avg * fwd_rxn_rates[6] / 2.01594000e+00));
  jac[29] += -1.0 * ((-mw_avg * fwd_rxn_rates[7] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[7] / 2.01594000e+00));
  jac[29] += 2.0 * ((-mw_avg * pres_mod[0] / 2.01594000e+00 + 2.5 * rho / 2.01594000e+00) * fwd_rxn_rates[10] + pres_mod[0] * ((-mw_avg * fwd_rxn_rates[10] / 2.01594000e+00 + exp(4.52701605e+01 - 1.4 * logT - (5.25257556e+04 / T)) * (rho / 2.01594000e+00))));
  jac[29] += -2.0 * ((-mw_avg * pres_mod[1] / 2.01594000e+00 + 2.5 * rho / 2.01594000e+00) * fwd_rxn_rates[11] + pres_mod[1] * (2.0 * (-mw_avg * fwd_rxn_rates[11] / 2.01594000e+00)));
  jac[29] += 2.0 * ((-mw_avg * fwd_rxn_rates[12] / 2.01594000e+00 + exp(4.32112625e+01 - 1.1 * logT - (5.25257556e+04 / T)) * (rho / 2.01594000e+00) * conc[9]) + (-mw_avg * fwd_rxn_rates[12] / 2.01594000e+00));
  jac[29] += -2.0 * (2.0 * (-mw_avg * fwd_rxn_rates[13] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[13] / 2.01594000e+00));
  jac[29] += 2.0 * ((-mw_avg * fwd_rxn_rates[14] / 2.01594000e+00 + exp(4.32112625e+01 - 1.1 * logT - (5.25257556e+04 / T)) * (rho / 2.01594000e+00) * conc[10]) + (-mw_avg * fwd_rxn_rates[14] / 2.01594000e+00));
  jac[29] += -2.0 * (2.0 * (-mw_avg * fwd_rxn_rates[15] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[15] / 2.01594000e+00));
  jac[29] += -1.0 * ((-mw_avg * pres_mod[4] / 2.01594000e+00 + 2.5 * rho / 2.01594000e+00) * fwd_rxn_rates[22] + pres_mod[4] * ((-mw_avg * fwd_rxn_rates[22] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[22] / 2.01594000e+00)));
  jac[29] += ((-mw_avg * pres_mod[5] / 2.01594000e+00 + 2.5 * rho / 2.01594000e+00) * fwd_rxn_rates[23] + pres_mod[5] * ((-mw_avg * fwd_rxn_rates[23] / 2.01594000e+00)));
  jac[29] += ((-mw_avg * pres_mod[6] / 2.01594000e+00 + 3.0 * rho / 2.01594000e+00) * fwd_rxn_rates[24] + pres_mod[6] * ((-mw_avg * fwd_rxn_rates[24] / 2.01594000e+00)));
  jac[29] += -1.0 * ((-mw_avg * pres_mod[7] / 2.01594000e+00 + 3.0 * rho / 2.01594000e+00) * fwd_rxn_rates[25] + pres_mod[7] * ((-mw_avg * fwd_rxn_rates[25] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[25] / 2.01594000e+00)));
  jac[29] += (2.0 * (-mw_avg * fwd_rxn_rates[26] / 2.01594000e+00));
  jac[29] += -1.0 * ((-mw_avg * fwd_rxn_rates[27] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[27] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[27] / 2.01594000e+00));
  Pr = (m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * (exp(1.87346128e+01 - 2.16 * logT - (2.64088106e+02 / T)));
  Fcent = 5.0000e-01 * exp(-T / 1.0000e-30) + 5.0000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[29] += -1.0 * (pres_mod[8] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 2.01594000e+00 + rho * 2.0 / ((m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * 2.01594000e+00)) * fwd_rxn_rates[28] + pres_mod[8] * ((-mw_avg * fwd_rxn_rates[28] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[28] / 2.01594000e+00)));
  Pr = (m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * (exp(1.87345962e+01 - 2.159967 * logT - (2.64188750e+02 / T)));
  Fcent = 5.0000e-01 * exp(-T / 1.0000e-30) + 5.0000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[29] += (pres_mod[9] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 2.01594000e+00 + rho * 2.0 / ((m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * 2.01594000e+00)) * fwd_rxn_rates[29] + pres_mod[9] * ((-mw_avg * fwd_rxn_rates[29] / 2.01594000e+00)));
  jac[29] += -1.0 * ((-mw_avg * fwd_rxn_rates[30] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[30] / 2.01594000e+00));
  jac[29] += ((-mw_avg * fwd_rxn_rates[31] / 2.01594000e+00 + exp(1.17897235e+01 + 2.6028 * logT - (2.65552467e+04 / T)) * (rho / 2.01594000e+00) * conc[5]) + (-mw_avg * fwd_rxn_rates[31] / 2.01594000e+00));
  jac[29] += -1.0 * ((-mw_avg * fwd_rxn_rates[32] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[32] / 2.01594000e+00));
  jac[29] += (2.0 * (-mw_avg * fwd_rxn_rates[33] / 2.01594000e+00));
  jac[29] += -1.0 * ((-mw_avg * fwd_rxn_rates[44] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[44] / 2.01594000e+00));
  jac[29] += ((-mw_avg * fwd_rxn_rates[45] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[45] / 2.01594000e+00));
  jac[29] += -1.0 * ((-mw_avg * fwd_rxn_rates[46] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[46] / 2.01594000e+00));
  jac[29] += ((-mw_avg * fwd_rxn_rates[47] / 2.01594000e+00 + exp(2.47202181e+01 + 0.74731 * logT - (1.19941692e+04 / T)) * (rho / 2.01594000e+00) * conc[6]) + (-mw_avg * fwd_rxn_rates[47] / 2.01594000e+00));
  jac[29] += sp_rates[0] * mw_avg / 2.01594000e+00;
  jac[29] *= 1.00797000e+00 / rho;

  //partial of omega_H wrt Y_O;
  jac[43] = -1.0 * ((-mw_avg * fwd_rxn_rates[0] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[0] / 1.59994000e+01));
  jac[43] += ((-mw_avg * fwd_rxn_rates[1] / 1.59994000e+01 + exp(2.63075513e+01 + 0.40478 * logT - (-7.48736077e+02 / T)) * (rho / 1.59994000e+01) * conc[3]) + (-mw_avg * fwd_rxn_rates[1] / 1.59994000e+01));
  jac[43] += ((-mw_avg * fwd_rxn_rates[2] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[2] / 1.59994000e+01 + exp(2.89707478e+01 - (3.99956606e+03 / T)) * (rho / 1.59994000e+01) * conc[1]));
  jac[43] += -1.0 * ((-mw_avg * fwd_rxn_rates[3] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[3] / 1.59994000e+01));
  jac[43] += ((-mw_avg * fwd_rxn_rates[4] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[4] / 1.59994000e+01 + exp(3.44100335e+01 - (9.64666348e+03 / T)) * (rho / 1.59994000e+01) * conc[1]));
  jac[43] += -1.0 * ((-mw_avg * fwd_rxn_rates[5] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[5] / 1.59994000e+01));
  jac[43] += ((-mw_avg * fwd_rxn_rates[6] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[6] / 1.59994000e+01));
  jac[43] += -1.0 * ((-mw_avg * fwd_rxn_rates[7] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[7] / 1.59994000e+01));
  jac[43] += 2.0 * ((-mw_avg * pres_mod[0] / 1.59994000e+01 + rho / 1.59994000e+01) * fwd_rxn_rates[10] + pres_mod[0] * ((-mw_avg * fwd_rxn_rates[10] / 1.59994000e+01)));
  jac[43] += -2.0 * ((-mw_avg * pres_mod[1] / 1.59994000e+01 + rho / 1.59994000e+01) * fwd_rxn_rates[11] + pres_mod[1] * (2.0 * (-mw_avg * fwd_rxn_rates[11] / 1.59994000e+01)));
  jac[43] += 2.0 * ((-mw_avg * fwd_rxn_rates[12] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[12] / 1.59994000e+01));
  jac[43] += -2.0 * (2.0 * (-mw_avg * fwd_rxn_rates[13] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[13] / 1.59994000e+01));
  jac[43] += 2.0 * ((-mw_avg * fwd_rxn_rates[14] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[14] / 1.59994000e+01));
  jac[43] += -2.0 * (2.0 * (-mw_avg * fwd_rxn_rates[15] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[15] / 1.59994000e+01));
  jac[43] += -1.0 * ((-mw_avg * pres_mod[4] / 1.59994000e+01 + rho / 1.59994000e+01) * fwd_rxn_rates[22] + pres_mod[4] * ((-mw_avg * fwd_rxn_rates[22] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[22] / 1.59994000e+01 + exp(4.29970685e+01 - 1.0 * logT) * (rho / 1.59994000e+01) * conc[0])));
  jac[43] += ((-mw_avg * pres_mod[5] / 1.59994000e+01 + rho / 1.59994000e+01) * fwd_rxn_rates[23] + pres_mod[5] * ((-mw_avg * fwd_rxn_rates[23] / 1.59994000e+01)));
  jac[43] += ((-mw_avg * pres_mod[6] / 1.59994000e+01 + rho / 1.59994000e+01) * fwd_rxn_rates[24] + pres_mod[6] * ((-mw_avg * fwd_rxn_rates[24] / 1.59994000e+01)));
  jac[43] += -1.0 * ((-mw_avg * pres_mod[7] / 1.59994000e+01 + rho / 1.59994000e+01) * fwd_rxn_rates[25] + pres_mod[7] * ((-mw_avg * fwd_rxn_rates[25] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[25] / 1.59994000e+01)));
  jac[43] += (2.0 * (-mw_avg * fwd_rxn_rates[26] / 1.59994000e+01));
  jac[43] += -1.0 * ((-mw_avg * fwd_rxn_rates[27] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[27] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[27] / 1.59994000e+01));
  Pr = (m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * (exp(1.87346128e+01 - 2.16 * logT - (2.64088106e+02 / T)));
  Fcent = 5.0000e-01 * exp(-T / 1.0000e-30) + 5.0000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[43] += -1.0 * (pres_mod[8] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 1.59994000e+01 + rho / ((m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * 1.59994000e+01)) * fwd_rxn_rates[28] + pres_mod[8] * ((-mw_avg * fwd_rxn_rates[28] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[28] / 1.59994000e+01)));
  Pr = (m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * (exp(1.87345962e+01 - 2.159967 * logT - (2.64188750e+02 / T)));
  Fcent = 5.0000e-01 * exp(-T / 1.0000e-30) + 5.0000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[43] += (pres_mod[9] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 1.59994000e+01 + rho / ((m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * 1.59994000e+01)) * fwd_rxn_rates[29] + pres_mod[9] * ((-mw_avg * fwd_rxn_rates[29] / 1.59994000e+01)));
  jac[43] += -1.0 * ((-mw_avg * fwd_rxn_rates[30] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[30] / 1.59994000e+01));
  jac[43] += ((-mw_avg * fwd_rxn_rates[31] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[31] / 1.59994000e+01));
  jac[43] += -1.0 * ((-mw_avg * fwd_rxn_rates[32] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[32] / 1.59994000e+01));
  jac[43] += (2.0 * (-mw_avg * fwd_rxn_rates[33] / 1.59994000e+01));
  jac[43] += -1.0 * ((-mw_avg * fwd_rxn_rates[44] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[44] / 1.59994000e+01));
  jac[43] += ((-mw_avg * fwd_rxn_rates[45] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[45] / 1.59994000e+01));
  jac[43] += -1.0 * ((-mw_avg * fwd_rxn_rates[46] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[46] / 1.59994000e+01));
  jac[43] += ((-mw_avg * fwd_rxn_rates[47] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[47] / 1.59994000e+01));
  jac[43] += sp_rates[0] * mw_avg / 1.59994000e+01;
  jac[43] *= 1.00797000e+00 / rho;

  //partial of omega_H wrt Y_OH;
  jac[57] = -1.0 * ((-mw_avg * fwd_rxn_rates[0] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[0] / 1.70073700e+01));
  jac[57] += ((-mw_avg * fwd_rxn_rates[1] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[1] / 1.70073700e+01 + exp(2.63075513e+01 + 0.40478 * logT - (-7.48736077e+02 / T)) * (rho / 1.70073700e+01) * conc[2]));
  jac[57] += ((-mw_avg * fwd_rxn_rates[2] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[2] / 1.70073700e+01));
  jac[57] += -1.0 * ((-mw_avg * fwd_rxn_rates[3] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[3] / 1.70073700e+01 + exp(2.85866095e+01 - 0.053533 * logT - (3.32706731e+03 / T)) * (rho / 1.70073700e+01) * conc[0]));
  jac[57] += ((-mw_avg * fwd_rxn_rates[4] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[4] / 1.70073700e+01));
  jac[57] += -1.0 * ((-mw_avg * fwd_rxn_rates[5] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[5] / 1.70073700e+01 + exp(3.40258987e+01 - 0.053533 * logT - (8.97436602e+03 / T)) * (rho / 1.70073700e+01) * conc[0]));
  jac[57] += ((-mw_avg * fwd_rxn_rates[6] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[6] / 1.70073700e+01 + exp(1.91907890e+01 + 1.51 * logT - (1.72603316e+03 / T)) * (rho / 1.70073700e+01) * conc[1]));
  jac[57] += -1.0 * ((-mw_avg * fwd_rxn_rates[7] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[7] / 1.70073700e+01));
  jac[57] += 2.0 * ((-mw_avg * pres_mod[0] / 1.70073700e+01 + rho / 1.70073700e+01) * fwd_rxn_rates[10] + pres_mod[0] * ((-mw_avg * fwd_rxn_rates[10] / 1.70073700e+01)));
  jac[57] += -2.0 * ((-mw_avg * pres_mod[1] / 1.70073700e+01 + rho / 1.70073700e+01) * fwd_rxn_rates[11] + pres_mod[1] * (2.0 * (-mw_avg * fwd_rxn_rates[11] / 1.70073700e+01)));
  jac[57] += 2.0 * ((-mw_avg * fwd_rxn_rates[12] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[12] / 1.70073700e+01));
  jac[57] += -2.0 * (2.0 * (-mw_avg * fwd_rxn_rates[13] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[13] / 1.70073700e+01));
  jac[57] += 2.0 * ((-mw_avg * fwd_rxn_rates[14] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[14] / 1.70073700e+01));
  jac[57] += -2.0 * (2.0 * (-mw_avg * fwd_rxn_rates[15] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[15] / 1.70073700e+01));
  jac[57] += -1.0 * ((-mw_avg * pres_mod[4] / 1.70073700e+01 + rho / 1.70073700e+01) * fwd_rxn_rates[22] + pres_mod[4] * ((-mw_avg * fwd_rxn_rates[22] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[22] / 1.70073700e+01)));
  jac[57] += ((-mw_avg * pres_mod[5] / 1.70073700e+01 + rho / 1.70073700e+01) * fwd_rxn_rates[23] + pres_mod[5] * ((-mw_avg * fwd_rxn_rates[23] / 1.70073700e+01 + exp(4.38224602e+01 - 1.0301 * logT - (5.18313166e+04 / T)) * (rho / 1.70073700e+01))));
  jac[57] += ((-mw_avg * pres_mod[6] / 1.70073700e+01 + rho / 1.70073700e+01) * fwd_rxn_rates[24] + pres_mod[6] * ((-mw_avg * fwd_rxn_rates[24] / 1.70073700e+01)));
  jac[57] += -1.0 * ((-mw_avg * pres_mod[7] / 1.70073700e+01 + rho / 1.70073700e+01) * fwd_rxn_rates[25] + pres_mod[7] * ((-mw_avg * fwd_rxn_rates[25] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[25] / 1.70073700e+01 + exp(5.85301272e+01 - 3.0183 * logT - (4.50771425e+02 / T)) * (rho / 1.70073700e+01) * conc[0])));
  jac[57] += (2.0 * (-mw_avg * fwd_rxn_rates[26] / 1.70073700e+01));
  jac[57] += -1.0 * ((-mw_avg * fwd_rxn_rates[27] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[27] / 1.70073700e+01 + exp(5.44311720e+01 - 2.1363 * logT - (1.43809259e+02 / T)) * (rho / 1.70073700e+01) * conc[0] * conc[4]) + (-mw_avg * fwd_rxn_rates[27] / 1.70073700e+01));
  Pr = (m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * (exp(1.87346128e+01 - 2.16 * logT - (2.64088106e+02 / T)));
  Fcent = 5.0000e-01 * exp(-T / 1.0000e-30) + 5.0000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[57] += -1.0 * (pres_mod[8] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 1.70073700e+01 + rho / ((m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * 1.70073700e+01)) * fwd_rxn_rates[28] + pres_mod[8] * ((-mw_avg * fwd_rxn_rates[28] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[28] / 1.70073700e+01)));
  Pr = (m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * (exp(1.87345962e+01 - 2.159967 * logT - (2.64188750e+02 / T)));
  Fcent = 5.0000e-01 * exp(-T / 1.0000e-30) + 5.0000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[57] += (pres_mod[9] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 1.70073700e+01 + rho / ((m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * 1.70073700e+01)) * fwd_rxn_rates[29] + pres_mod[9] * ((-mw_avg * fwd_rxn_rates[29] / 1.70073700e+01)));
  jac[57] += -1.0 * ((-mw_avg * fwd_rxn_rates[30] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[30] / 1.70073700e+01));
  jac[57] += ((-mw_avg * fwd_rxn_rates[31] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[31] / 1.70073700e+01));
  jac[57] += -1.0 * ((-mw_avg * fwd_rxn_rates[32] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[32] / 1.70073700e+01));
  jac[57] += (2.0 * (-mw_avg * fwd_rxn_rates[33] / 1.70073700e+01 + exp(2.25013658e+01 + 0.86409 * logT - (1.83206092e+04 / T)) * (rho / 1.70073700e+01) * conc[3]));
  jac[57] += -1.0 * ((-mw_avg * fwd_rxn_rates[44] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[44] / 1.70073700e+01));
  jac[57] += ((-mw_avg * fwd_rxn_rates[45] / 1.70073700e+01 + exp(1.88701627e+01 + 1.2843 * logT - (3.59925720e+04 / T)) * (rho / 1.70073700e+01) * conc[4]) + (-mw_avg * fwd_rxn_rates[45] / 1.70073700e+01));
  jac[57] += -1.0 * ((-mw_avg * fwd_rxn_rates[46] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[46] / 1.70073700e+01));
  jac[57] += ((-mw_avg * fwd_rxn_rates[47] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[47] / 1.70073700e+01));
  jac[57] += sp_rates[0] * mw_avg / 1.70073700e+01;
  jac[57] *= 1.00797000e+00 / rho;

  //partial of omega_H wrt Y_H2O;
  jac[71] = -1.0 * ((-mw_avg * fwd_rxn_rates[0] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[0] / 1.80153400e+01));
  jac[71] += ((-mw_avg * fwd_rxn_rates[1] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[1] / 1.80153400e+01));
  jac[71] += ((-mw_avg * fwd_rxn_rates[2] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[2] / 1.80153400e+01));
  jac[71] += -1.0 * ((-mw_avg * fwd_rxn_rates[3] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[3] / 1.80153400e+01));
  jac[71] += ((-mw_avg * fwd_rxn_rates[4] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[4] / 1.80153400e+01));
  jac[71] += -1.0 * ((-mw_avg * fwd_rxn_rates[5] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[5] / 1.80153400e+01));
  jac[71] += ((-mw_avg * fwd_rxn_rates[6] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[6] / 1.80153400e+01));
  jac[71] += -1.0 * ((-mw_avg * fwd_rxn_rates[7] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[7] / 1.80153400e+01 + exp(2.34232839e+01 + 1.1829 * logT - (9.55507805e+03 / T)) * (rho / 1.80153400e+01) * conc[0]));
  jac[71] += 2.0 * ((-mw_avg * pres_mod[0] / 1.80153400e+01 + 12.0 * rho / 1.80153400e+01) * fwd_rxn_rates[10] + pres_mod[0] * ((-mw_avg * fwd_rxn_rates[10] / 1.80153400e+01)));
  jac[71] += -2.0 * ((-mw_avg * pres_mod[1] / 1.80153400e+01 + 12.0 * rho / 1.80153400e+01) * fwd_rxn_rates[11] + pres_mod[1] * (2.0 * (-mw_avg * fwd_rxn_rates[11] / 1.80153400e+01)));
  jac[71] += 2.0 * ((-mw_avg * fwd_rxn_rates[12] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[12] / 1.80153400e+01));
  jac[71] += -2.0 * (2.0 * (-mw_avg * fwd_rxn_rates[13] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[13] / 1.80153400e+01));
  jac[71] += 2.0 * ((-mw_avg * fwd_rxn_rates[14] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[14] / 1.80153400e+01));
  jac[71] += -2.0 * (2.0 * (-mw_avg * fwd_rxn_rates[15] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[15] / 1.80153400e+01));
  jac[71] += -1.0 * ((-mw_avg * pres_mod[4] / 1.80153400e+01 + 12.0 * rho / 1.80153400e+01) * fwd_rxn_rates[22] + pres_mod[4] * ((-mw_avg * fwd_rxn_rates[22] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[22] / 1.80153400e+01)));
  jac[71] += ((-mw_avg * pres_mod[5] / 1.80153400e+01 + 12.0 * rho / 1.80153400e+01) * fwd_rxn_rates[23] + pres_mod[5] * ((-mw_avg * fwd_rxn_rates[23] / 1.80153400e+01)));
  jac[71] += ((-mw_avg * pres_mod[6] / 1.80153400e+01) * fwd_rxn_rates[24] + pres_mod[6] * ((-mw_avg * fwd_rxn_rates[24] / 1.80153400e+01 + exp(6.39721672e+01 - 3.322 * logT - (6.07835411e+04 / T)) * (rho / 1.80153400e+01))));
  jac[71] += -1.0 * ((-mw_avg * pres_mod[7] / 1.80153400e+01) * fwd_rxn_rates[25] + pres_mod[7] * ((-mw_avg * fwd_rxn_rates[25] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[25] / 1.80153400e+01)));
  jac[71] += (2.0 * (-mw_avg * fwd_rxn_rates[26] / 1.80153400e+01 + exp(5.98731945e+01 - 2.44 * logT - (6.04765789e+04 / T)) * (rho / 1.80153400e+01) * conc[4]));
  jac[71] += -1.0 * ((-mw_avg * fwd_rxn_rates[27] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[27] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[27] / 1.80153400e+01 + exp(5.44311720e+01 - 2.1363 * logT - (1.43809259e+02 / T)) * (rho / 1.80153400e+01) * conc[0] * conc[3]));
  Pr = (m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * (exp(1.87346128e+01 - 2.16 * logT - (2.64088106e+02 / T)));
  Fcent = 5.0000e-01 * exp(-T / 1.0000e-30) + 5.0000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[71] += -1.0 * (pres_mod[8] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 1.80153400e+01 + rho * 14.0 / ((m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * 1.80153400e+01)) * fwd_rxn_rates[28] + pres_mod[8] * ((-mw_avg * fwd_rxn_rates[28] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[28] / 1.80153400e+01)));
  Pr = (m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * (exp(1.87345962e+01 - 2.159967 * logT - (2.64188750e+02 / T)));
  Fcent = 5.0000e-01 * exp(-T / 1.0000e-30) + 5.0000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[71] += (pres_mod[9] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 1.80153400e+01 + rho * 14.0 / ((m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * 1.80153400e+01)) * fwd_rxn_rates[29] + pres_mod[9] * ((-mw_avg * fwd_rxn_rates[29] / 1.80153400e+01)));
  jac[71] += -1.0 * ((-mw_avg * fwd_rxn_rates[30] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[30] / 1.80153400e+01));
  jac[71] += ((-mw_avg * fwd_rxn_rates[31] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[31] / 1.80153400e+01));
  jac[71] += -1.0 * ((-mw_avg * fwd_rxn_rates[32] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[32] / 1.80153400e+01));
  jac[71] += (2.0 * (-mw_avg * fwd_rxn_rates[33] / 1.80153400e+01));
  jac[71] += -1.0 * ((-mw_avg * fwd_rxn_rates[44] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[44] / 1.80153400e+01));
  jac[71] += ((-mw_avg * fwd_rxn_rates[45] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[45] / 1.80153400e+01 + exp(1.88701627e+01 + 1.2843 * logT - (3.59925720e+04 / T)) * (rho / 1.80153400e+01) * conc[3]));
  jac[71] += -1.0 * ((-mw_avg * fwd_rxn_rates[46] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[46] / 1.80153400e+01));
  jac[71] += ((-mw_avg * fwd_rxn_rates[47] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[47] / 1.80153400e+01));
  jac[71] += sp_rates[0] * mw_avg / 1.80153400e+01;
  jac[71] *= 1.00797000e+00 / rho;

  //partial of omega_H wrt Y_O2;
  jac[85] = -1.0 * ((-mw_avg * fwd_rxn_rates[0] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[0] / 3.19988000e+01 + exp(3.22754120e+01 - (7.69216995e+03 / T)) * (rho / 3.19988000e+01) * conc[0]));
  jac[85] += ((-mw_avg * fwd_rxn_rates[1] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[1] / 3.19988000e+01));
  jac[85] += ((-mw_avg * fwd_rxn_rates[2] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[2] / 3.19988000e+01));
  jac[85] += -1.0 * ((-mw_avg * fwd_rxn_rates[3] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[3] / 3.19988000e+01));
  jac[85] += ((-mw_avg * fwd_rxn_rates[4] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[4] / 3.19988000e+01));
  jac[85] += -1.0 * ((-mw_avg * fwd_rxn_rates[5] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[5] / 3.19988000e+01));
  jac[85] += ((-mw_avg * fwd_rxn_rates[6] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[6] / 3.19988000e+01));
  jac[85] += -1.0 * ((-mw_avg * fwd_rxn_rates[7] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[7] / 3.19988000e+01));
  jac[85] += 2.0 * ((-mw_avg * pres_mod[0] / 3.19988000e+01 + rho / 3.19988000e+01) * fwd_rxn_rates[10] + pres_mod[0] * ((-mw_avg * fwd_rxn_rates[10] / 3.19988000e+01)));
  jac[85] += -2.0 * ((-mw_avg * pres_mod[1] / 3.19988000e+01 + rho / 3.19988000e+01) * fwd_rxn_rates[11] + pres_mod[1] * (2.0 * (-mw_avg * fwd_rxn_rates[11] / 3.19988000e+01)));
  jac[85] += 2.0 * ((-mw_avg * fwd_rxn_rates[12] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[12] / 3.19988000e+01));
  jac[85] += -2.0 * (2.0 * (-mw_avg * fwd_rxn_rates[13] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[13] / 3.19988000e+01));
  jac[85] += 2.0 * ((-mw_avg * fwd_rxn_rates[14] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[14] / 3.19988000e+01));
  jac[85] += -2.0 * (2.0 * (-mw_avg * fwd_rxn_rates[15] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[15] / 3.19988000e+01));
  jac[85] += -1.0 * ((-mw_avg * pres_mod[4] / 3.19988000e+01 + rho / 3.19988000e+01) * fwd_rxn_rates[22] + pres_mod[4] * ((-mw_avg * fwd_rxn_rates[22] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[22] / 3.19988000e+01)));
  jac[85] += ((-mw_avg * pres_mod[5] / 3.19988000e+01 + rho / 3.19988000e+01) * fwd_rxn_rates[23] + pres_mod[5] * ((-mw_avg * fwd_rxn_rates[23] / 3.19988000e+01)));
  jac[85] += ((-mw_avg * pres_mod[6] / 3.19988000e+01 + 1.5 * rho / 3.19988000e+01) * fwd_rxn_rates[24] + pres_mod[6] * ((-mw_avg * fwd_rxn_rates[24] / 3.19988000e+01)));
  jac[85] += -1.0 * ((-mw_avg * pres_mod[7] / 3.19988000e+01 + 1.5 * rho / 3.19988000e+01) * fwd_rxn_rates[25] + pres_mod[7] * ((-mw_avg * fwd_rxn_rates[25] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[25] / 3.19988000e+01)));
  jac[85] += (2.0 * (-mw_avg * fwd_rxn_rates[26] / 3.19988000e+01));
  jac[85] += -1.0 * ((-mw_avg * fwd_rxn_rates[27] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[27] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[27] / 3.19988000e+01));
  Pr = (m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * (exp(1.87346128e+01 - 2.16 * logT - (2.64088106e+02 / T)));
  Fcent = 5.0000e-01 * exp(-T / 1.0000e-30) + 5.0000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[85] += -1.0 * (pres_mod[8] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 3.19988000e+01 + rho * 0.78 / ((m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * 3.19988000e+01)) * fwd_rxn_rates[28] + pres_mod[8] * ((-mw_avg * fwd_rxn_rates[28] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[28] / 3.19988000e+01 + exp(2.91680604e+01 + 0.44 * logT) * (rho / 3.19988000e+01) * conc[0])));
  Pr = (m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * (exp(1.87345962e+01 - 2.159967 * logT - (2.64188750e+02 / T)));
  Fcent = 5.0000e-01 * exp(-T / 1.0000e-30) + 5.0000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[85] += (pres_mod[9] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 3.19988000e+01 + rho * 0.78 / ((m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * 3.19988000e+01)) * fwd_rxn_rates[29] + pres_mod[9] * ((-mw_avg * fwd_rxn_rates[29] / 3.19988000e+01)));
  jac[85] += -1.0 * ((-mw_avg * fwd_rxn_rates[30] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[30] / 3.19988000e+01));
  jac[85] += ((-mw_avg * fwd_rxn_rates[31] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[31] / 3.19988000e+01 + exp(1.17897235e+01 + 2.6028 * logT - (2.65552467e+04 / T)) * (rho / 3.19988000e+01) * conc[1]));
  jac[85] += -1.0 * ((-mw_avg * fwd_rxn_rates[32] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[32] / 3.19988000e+01));
  jac[85] += (2.0 * (-mw_avg * fwd_rxn_rates[33] / 3.19988000e+01));
  jac[85] += -1.0 * ((-mw_avg * fwd_rxn_rates[44] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[44] / 3.19988000e+01));
  jac[85] += ((-mw_avg * fwd_rxn_rates[45] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[45] / 3.19988000e+01));
  jac[85] += -1.0 * ((-mw_avg * fwd_rxn_rates[46] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[46] / 3.19988000e+01));
  jac[85] += ((-mw_avg * fwd_rxn_rates[47] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[47] / 3.19988000e+01));
  jac[85] += sp_rates[0] * mw_avg / 3.19988000e+01;
  jac[85] *= 1.00797000e+00 / rho;

  //partial of omega_H wrt Y_HO2;
  jac[99] = -1.0 * ((-mw_avg * fwd_rxn_rates[0] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[0] / 3.30067700e+01));
  jac[99] += ((-mw_avg * fwd_rxn_rates[1] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[1] / 3.30067700e+01));
  jac[99] += ((-mw_avg * fwd_rxn_rates[2] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[2] / 3.30067700e+01));
  jac[99] += -1.0 * ((-mw_avg * fwd_rxn_rates[3] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[3] / 3.30067700e+01));
  jac[99] += ((-mw_avg * fwd_rxn_rates[4] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[4] / 3.30067700e+01));
  jac[99] += -1.0 * ((-mw_avg * fwd_rxn_rates[5] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[5] / 3.30067700e+01));
  jac[99] += ((-mw_avg * fwd_rxn_rates[6] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[6] / 3.30067700e+01));
  jac[99] += -1.0 * ((-mw_avg * fwd_rxn_rates[7] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[7] / 3.30067700e+01));
  jac[99] += 2.0 * ((-mw_avg * pres_mod[0] / 3.30067700e+01 + rho / 3.30067700e+01) * fwd_rxn_rates[10] + pres_mod[0] * ((-mw_avg * fwd_rxn_rates[10] / 3.30067700e+01)));
  jac[99] += -2.0 * ((-mw_avg * pres_mod[1] / 3.30067700e+01 + rho / 3.30067700e+01) * fwd_rxn_rates[11] + pres_mod[1] * (2.0 * (-mw_avg * fwd_rxn_rates[11] / 3.30067700e+01)));
  jac[99] += 2.0 * ((-mw_avg * fwd_rxn_rates[12] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[12] / 3.30067700e+01));
  jac[99] += -2.0 * (2.0 * (-mw_avg * fwd_rxn_rates[13] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[13] / 3.30067700e+01));
  jac[99] += 2.0 * ((-mw_avg * fwd_rxn_rates[14] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[14] / 3.30067700e+01));
  jac[99] += -2.0 * (2.0 * (-mw_avg * fwd_rxn_rates[15] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[15] / 3.30067700e+01));
  jac[99] += -1.0 * ((-mw_avg * pres_mod[4] / 3.30067700e+01 + rho / 3.30067700e+01) * fwd_rxn_rates[22] + pres_mod[4] * ((-mw_avg * fwd_rxn_rates[22] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[22] / 3.30067700e+01)));
  jac[99] += ((-mw_avg * pres_mod[5] / 3.30067700e+01 + rho / 3.30067700e+01) * fwd_rxn_rates[23] + pres_mod[5] * ((-mw_avg * fwd_rxn_rates[23] / 3.30067700e+01)));
  jac[99] += ((-mw_avg * pres_mod[6] / 3.30067700e+01 + rho / 3.30067700e+01) * fwd_rxn_rates[24] + pres_mod[6] * ((-mw_avg * fwd_rxn_rates[24] / 3.30067700e+01)));
  jac[99] += -1.0 * ((-mw_avg * pres_mod[7] / 3.30067700e+01 + rho / 3.30067700e+01) * fwd_rxn_rates[25] + pres_mod[7] * ((-mw_avg * fwd_rxn_rates[25] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[25] / 3.30067700e+01)));
  jac[99] += (2.0 * (-mw_avg * fwd_rxn_rates[26] / 3.30067700e+01));
  jac[99] += -1.0 * ((-mw_avg * fwd_rxn_rates[27] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[27] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[27] / 3.30067700e+01));
  Pr = (m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * (exp(1.87346128e+01 - 2.16 * logT - (2.64088106e+02 / T)));
  Fcent = 5.0000e-01 * exp(-T / 1.0000e-30) + 5.0000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[99] += -1.0 * (pres_mod[8] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 3.30067700e+01 + rho / ((m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * 3.30067700e+01)) * fwd_rxn_rates[28] + pres_mod[8] * ((-mw_avg * fwd_rxn_rates[28] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[28] / 3.30067700e+01)));
  Pr = (m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * (exp(1.87345962e+01 - 2.159967 * logT - (2.64188750e+02 / T)));
  Fcent = 5.0000e-01 * exp(-T / 1.0000e-30) + 5.0000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[99] += (pres_mod[9] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 3.30067700e+01 + rho / ((m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * 3.30067700e+01)) * fwd_rxn_rates[29] + pres_mod[9] * ((-mw_avg * fwd_rxn_rates[29] / 3.30067700e+01 + exp(3.34150001e+01 - 0.049433 * logT - (2.52182000e+04 / T)) * (rho / 3.30067700e+01))));
  jac[99] += -1.0 * ((-mw_avg * fwd_rxn_rates[30] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[30] / 3.30067700e+01 + exp(1.48271115e+01 + 2.09 * logT - (-7.30167382e+02 / T)) * (rho / 3.30067700e+01) * conc[0]));
  jac[99] += ((-mw_avg * fwd_rxn_rates[31] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[31] / 3.30067700e+01));
  jac[99] += -1.0 * ((-mw_avg * fwd_rxn_rates[32] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[32] / 3.30067700e+01 + exp(3.18907389e+01 - (1.48448916e+02 / T)) * (rho / 3.30067700e+01) * conc[0]));
  jac[99] += (2.0 * (-mw_avg * fwd_rxn_rates[33] / 3.30067700e+01));
  jac[99] += -1.0 * ((-mw_avg * fwd_rxn_rates[44] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[44] / 3.30067700e+01));
  jac[99] += ((-mw_avg * fwd_rxn_rates[45] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[45] / 3.30067700e+01));
  jac[99] += -1.0 * ((-mw_avg * fwd_rxn_rates[46] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[46] / 3.30067700e+01));
  jac[99] += ((-mw_avg * fwd_rxn_rates[47] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[47] / 3.30067700e+01 + exp(2.47202181e+01 + 0.74731 * logT - (1.19941692e+04 / T)) * (rho / 3.30067700e+01) * conc[1]));
  jac[99] += sp_rates[0] * mw_avg / 3.30067700e+01;
  jac[99] *= 1.00797000e+00 / rho;

  //partial of omega_H wrt Y_H2O2;
  jac[113] = -1.0 * ((-mw_avg * fwd_rxn_rates[0] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[0] / 3.40147400e+01));
  jac[113] += ((-mw_avg * fwd_rxn_rates[1] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[1] / 3.40147400e+01));
  jac[113] += ((-mw_avg * fwd_rxn_rates[2] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[2] / 3.40147400e+01));
  jac[113] += -1.0 * ((-mw_avg * fwd_rxn_rates[3] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[3] / 3.40147400e+01));
  jac[113] += ((-mw_avg * fwd_rxn_rates[4] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[4] / 3.40147400e+01));
  jac[113] += -1.0 * ((-mw_avg * fwd_rxn_rates[5] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[5] / 3.40147400e+01));
  jac[113] += ((-mw_avg * fwd_rxn_rates[6] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[6] / 3.40147400e+01));
  jac[113] += -1.0 * ((-mw_avg * fwd_rxn_rates[7] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[7] / 3.40147400e+01));
  jac[113] += 2.0 * ((-mw_avg * pres_mod[0] / 3.40147400e+01 + rho / 3.40147400e+01) * fwd_rxn_rates[10] + pres_mod[0] * ((-mw_avg * fwd_rxn_rates[10] / 3.40147400e+01)));
  jac[113] += -2.0 * ((-mw_avg * pres_mod[1] / 3.40147400e+01 + rho / 3.40147400e+01) * fwd_rxn_rates[11] + pres_mod[1] * (2.0 * (-mw_avg * fwd_rxn_rates[11] / 3.40147400e+01)));
  jac[113] += 2.0 * ((-mw_avg * fwd_rxn_rates[12] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[12] / 3.40147400e+01));
  jac[113] += -2.0 * (2.0 * (-mw_avg * fwd_rxn_rates[13] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[13] / 3.40147400e+01));
  jac[113] += 2.0 * ((-mw_avg * fwd_rxn_rates[14] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[14] / 3.40147400e+01));
  jac[113] += -2.0 * (2.0 * (-mw_avg * fwd_rxn_rates[15] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[15] / 3.40147400e+01));
  jac[113] += -1.0 * ((-mw_avg * pres_mod[4] / 3.40147400e+01 + rho / 3.40147400e+01) * fwd_rxn_rates[22] + pres_mod[4] * ((-mw_avg * fwd_rxn_rates[22] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[22] / 3.40147400e+01)));
  jac[113] += ((-mw_avg * pres_mod[5] / 3.40147400e+01 + rho / 3.40147400e+01) * fwd_rxn_rates[23] + pres_mod[5] * ((-mw_avg * fwd_rxn_rates[23] / 3.40147400e+01)));
  jac[113] += ((-mw_avg * pres_mod[6] / 3.40147400e+01 + rho / 3.40147400e+01) * fwd_rxn_rates[24] + pres_mod[6] * ((-mw_avg * fwd_rxn_rates[24] / 3.40147400e+01)));
  jac[113] += -1.0 * ((-mw_avg * pres_mod[7] / 3.40147400e+01 + rho / 3.40147400e+01) * fwd_rxn_rates[25] + pres_mod[7] * ((-mw_avg * fwd_rxn_rates[25] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[25] / 3.40147400e+01)));
  jac[113] += (2.0 * (-mw_avg * fwd_rxn_rates[26] / 3.40147400e+01));
  jac[113] += -1.0 * ((-mw_avg * fwd_rxn_rates[27] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[27] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[27] / 3.40147400e+01));
  Pr = (m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * (exp(1.87346128e+01 - 2.16 * logT - (2.64088106e+02 / T)));
  Fcent = 5.0000e-01 * exp(-T / 1.0000e-30) + 5.0000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[113] += -1.0 * (pres_mod[8] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 3.40147400e+01 + rho / ((m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * 3.40147400e+01)) * fwd_rxn_rates[28] + pres_mod[8] * ((-mw_avg * fwd_rxn_rates[28] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[28] / 3.40147400e+01)));
  Pr = (m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * (exp(1.87345962e+01 - 2.159967 * logT - (2.64188750e+02 / T)));
  Fcent = 5.0000e-01 * exp(-T / 1.0000e-30) + 5.0000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[113] += (pres_mod[9] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 3.40147400e+01 + rho / ((m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * 3.40147400e+01)) * fwd_rxn_rates[29] + pres_mod[9] * ((-mw_avg * fwd_rxn_rates[29] / 3.40147400e+01)));
  jac[113] += -1.0 * ((-mw_avg * fwd_rxn_rates[30] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[30] / 3.40147400e+01));
  jac[113] += ((-mw_avg * fwd_rxn_rates[31] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[31] / 3.40147400e+01));
  jac[113] += -1.0 * ((-mw_avg * fwd_rxn_rates[32] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[32] / 3.40147400e+01));
  jac[113] += (2.0 * (-mw_avg * fwd_rxn_rates[33] / 3.40147400e+01));
  jac[113] += -1.0 * ((-mw_avg * fwd_rxn_rates[44] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[44] / 3.40147400e+01 + exp(3.08132330e+01 - (1.99777016e+03 / T)) * (rho / 3.40147400e+01) * conc[0]));
  jac[113] += ((-mw_avg * fwd_rxn_rates[45] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[45] / 3.40147400e+01));
  jac[113] += -1.0 * ((-mw_avg * fwd_rxn_rates[46] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[46] / 3.40147400e+01 + exp(3.15063801e+01 - (4.00057249e+03 / T)) * (rho / 3.40147400e+01) * conc[0]));
  jac[113] += ((-mw_avg * fwd_rxn_rates[47] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[47] / 3.40147400e+01));
  jac[113] += sp_rates[0] * mw_avg / 3.40147400e+01;
  jac[113] *= 1.00797000e+00 / rho;

  //partial of omega_H wrt Y_N2;
  jac[127] = -1.0 * ((-mw_avg * fwd_rxn_rates[0] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[0] / 2.80134000e+01));
  jac[127] += ((-mw_avg * fwd_rxn_rates[1] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[1] / 2.80134000e+01));
  jac[127] += ((-mw_avg * fwd_rxn_rates[2] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[2] / 2.80134000e+01));
  jac[127] += -1.0 * ((-mw_avg * fwd_rxn_rates[3] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[3] / 2.80134000e+01));
  jac[127] += ((-mw_avg * fwd_rxn_rates[4] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[4] / 2.80134000e+01));
  jac[127] += -1.0 * ((-mw_avg * fwd_rxn_rates[5] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[5] / 2.80134000e+01));
  jac[127] += ((-mw_avg * fwd_rxn_rates[6] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[6] / 2.80134000e+01));
  jac[127] += -1.0 * ((-mw_avg * fwd_rxn_rates[7] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[7] / 2.80134000e+01));
  jac[127] += 2.0 * ((-mw_avg * pres_mod[0] / 2.80134000e+01 + rho / 2.80134000e+01) * fwd_rxn_rates[10] + pres_mod[0] * ((-mw_avg * fwd_rxn_rates[10] / 2.80134000e+01)));
  jac[127] += -2.0 * ((-mw_avg * pres_mod[1] / 2.80134000e+01 + rho / 2.80134000e+01) * fwd_rxn_rates[11] + pres_mod[1] * (2.0 * (-mw_avg * fwd_rxn_rates[11] / 2.80134000e+01)));
  jac[127] += 2.0 * ((-mw_avg * fwd_rxn_rates[12] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[12] / 2.80134000e+01));
  jac[127] += -2.0 * (2.0 * (-mw_avg * fwd_rxn_rates[13] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[13] / 2.80134000e+01));
  jac[127] += 2.0 * ((-mw_avg * fwd_rxn_rates[14] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[14] / 2.80134000e+01));
  jac[127] += -2.0 * (2.0 * (-mw_avg * fwd_rxn_rates[15] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[15] / 2.80134000e+01));
  jac[127] += -1.0 * ((-mw_avg * pres_mod[4] / 2.80134000e+01 + rho / 2.80134000e+01) * fwd_rxn_rates[22] + pres_mod[4] * ((-mw_avg * fwd_rxn_rates[22] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[22] / 2.80134000e+01)));
  jac[127] += ((-mw_avg * pres_mod[5] / 2.80134000e+01 + rho / 2.80134000e+01) * fwd_rxn_rates[23] + pres_mod[5] * ((-mw_avg * fwd_rxn_rates[23] / 2.80134000e+01)));
  jac[127] += ((-mw_avg * pres_mod[6] / 2.80134000e+01 + 2.0 * rho / 2.80134000e+01) * fwd_rxn_rates[24] + pres_mod[6] * ((-mw_avg * fwd_rxn_rates[24] / 2.80134000e+01)));
  jac[127] += -1.0 * ((-mw_avg * pres_mod[7] / 2.80134000e+01 + 2.0 * rho / 2.80134000e+01) * fwd_rxn_rates[25] + pres_mod[7] * ((-mw_avg * fwd_rxn_rates[25] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[25] / 2.80134000e+01)));
  jac[127] += (2.0 * (-mw_avg * fwd_rxn_rates[26] / 2.80134000e+01));
  jac[127] += -1.0 * ((-mw_avg * fwd_rxn_rates[27] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[27] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[27] / 2.80134000e+01));
  Pr = (m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * (exp(1.87346128e+01 - 2.16 * logT - (2.64088106e+02 / T)));
  Fcent = 5.0000e-01 * exp(-T / 1.0000e-30) + 5.0000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[127] += -1.0 * (pres_mod[8] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 2.80134000e+01 + rho / ((m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * 2.80134000e+01)) * fwd_rxn_rates[28] + pres_mod[8] * ((-mw_avg * fwd_rxn_rates[28] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[28] / 2.80134000e+01)));
  Pr = (m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * (exp(1.87345962e+01 - 2.159967 * logT - (2.64188750e+02 / T)));
  Fcent = 5.0000e-01 * exp(-T / 1.0000e-30) + 5.0000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[127] += (pres_mod[9] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 2.80134000e+01 + rho / ((m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * 2.80134000e+01)) * fwd_rxn_rates[29] + pres_mod[9] * ((-mw_avg * fwd_rxn_rates[29] / 2.80134000e+01)));
  jac[127] += -1.0 * ((-mw_avg * fwd_rxn_rates[30] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[30] / 2.80134000e+01));
  jac[127] += ((-mw_avg * fwd_rxn_rates[31] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[31] / 2.80134000e+01));
  jac[127] += -1.0 * ((-mw_avg * fwd_rxn_rates[32] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[32] / 2.80134000e+01));
  jac[127] += (2.0 * (-mw_avg * fwd_rxn_rates[33] / 2.80134000e+01));
  jac[127] += -1.0 * ((-mw_avg * fwd_rxn_rates[44] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[44] / 2.80134000e+01));
  jac[127] += ((-mw_avg * fwd_rxn_rates[45] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[45] / 2.80134000e+01));
  jac[127] += -1.0 * ((-mw_avg * fwd_rxn_rates[46] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[46] / 2.80134000e+01));
  jac[127] += ((-mw_avg * fwd_rxn_rates[47] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[47] / 2.80134000e+01));
  jac[127] += sp_rates[0] * mw_avg / 2.80134000e+01;
  jac[127] *= 1.00797000e+00 / rho;

  //partial of omega_H wrt Y_AR;
  jac[141] = -1.0 * ((-mw_avg * fwd_rxn_rates[0] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[0] / 3.99480000e+01));
  jac[141] += ((-mw_avg * fwd_rxn_rates[1] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[1] / 3.99480000e+01));
  jac[141] += ((-mw_avg * fwd_rxn_rates[2] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[2] / 3.99480000e+01));
  jac[141] += -1.0 * ((-mw_avg * fwd_rxn_rates[3] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[3] / 3.99480000e+01));
  jac[141] += ((-mw_avg * fwd_rxn_rates[4] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[4] / 3.99480000e+01));
  jac[141] += -1.0 * ((-mw_avg * fwd_rxn_rates[5] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[5] / 3.99480000e+01));
  jac[141] += ((-mw_avg * fwd_rxn_rates[6] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[6] / 3.99480000e+01));
  jac[141] += -1.0 * ((-mw_avg * fwd_rxn_rates[7] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[7] / 3.99480000e+01));
  jac[141] += 2.0 * ((-mw_avg * pres_mod[0] / 3.99480000e+01) * fwd_rxn_rates[10] + pres_mod[0] * ((-mw_avg * fwd_rxn_rates[10] / 3.99480000e+01)));
  jac[141] += -2.0 * ((-mw_avg * pres_mod[1] / 3.99480000e+01) * fwd_rxn_rates[11] + pres_mod[1] * (2.0 * (-mw_avg * fwd_rxn_rates[11] / 3.99480000e+01)));
  jac[141] += 2.0 * ((-mw_avg * fwd_rxn_rates[12] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[12] / 3.99480000e+01 + exp(4.32112625e+01 - 1.1 * logT - (5.25257556e+04 / T)) * (rho / 3.99480000e+01) * conc[1]));
  jac[141] += -2.0 * (2.0 * (-mw_avg * fwd_rxn_rates[13] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[13] / 3.99480000e+01 + exp(4.20017378e+01 - 1.1234 * logT - (2.21510944e+01 / T)) * (rho / 3.99480000e+01) * conc[0] * conc[0]));
  jac[141] += 2.0 * ((-mw_avg * fwd_rxn_rates[14] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[14] / 3.99480000e+01));
  jac[141] += -2.0 * (2.0 * (-mw_avg * fwd_rxn_rates[15] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[15] / 3.99480000e+01));
  jac[141] += -1.0 * ((-mw_avg * pres_mod[4] / 3.99480000e+01 + 0.75 * rho / 3.99480000e+01) * fwd_rxn_rates[22] + pres_mod[4] * ((-mw_avg * fwd_rxn_rates[22] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[22] / 3.99480000e+01)));
  jac[141] += ((-mw_avg * pres_mod[5] / 3.99480000e+01 + 0.75 * rho / 3.99480000e+01) * fwd_rxn_rates[23] + pres_mod[5] * ((-mw_avg * fwd_rxn_rates[23] / 3.99480000e+01)));
  jac[141] += ((-mw_avg * pres_mod[6] / 3.99480000e+01 + rho / 3.99480000e+01) * fwd_rxn_rates[24] + pres_mod[6] * ((-mw_avg * fwd_rxn_rates[24] / 3.99480000e+01)));
  jac[141] += -1.0 * ((-mw_avg * pres_mod[7] / 3.99480000e+01 + rho / 3.99480000e+01) * fwd_rxn_rates[25] + pres_mod[7] * ((-mw_avg * fwd_rxn_rates[25] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[25] / 3.99480000e+01)));
  jac[141] += (2.0 * (-mw_avg * fwd_rxn_rates[26] / 3.99480000e+01));
  jac[141] += -1.0 * ((-mw_avg * fwd_rxn_rates[27] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[27] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[27] / 3.99480000e+01));
  Pr = (m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * (exp(1.87346128e+01 - 2.16 * logT - (2.64088106e+02 / T)));
  Fcent = 5.0000e-01 * exp(-T / 1.0000e-30) + 5.0000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[141] += -1.0 * (pres_mod[8] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 3.99480000e+01 + rho * 0.67 / ((m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * 3.99480000e+01)) * fwd_rxn_rates[28] + pres_mod[8] * ((-mw_avg * fwd_rxn_rates[28] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[28] / 3.99480000e+01)));
  Pr = (m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * (exp(1.87345962e+01 - 2.159967 * logT - (2.64188750e+02 / T)));
  Fcent = 5.0000e-01 * exp(-T / 1.0000e-30) + 5.0000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[141] += (pres_mod[9] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 3.99480000e+01 + rho * 0.67 / ((m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * 3.99480000e+01)) * fwd_rxn_rates[29] + pres_mod[9] * ((-mw_avg * fwd_rxn_rates[29] / 3.99480000e+01)));
  jac[141] += -1.0 * ((-mw_avg * fwd_rxn_rates[30] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[30] / 3.99480000e+01));
  jac[141] += ((-mw_avg * fwd_rxn_rates[31] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[31] / 3.99480000e+01));
  jac[141] += -1.0 * ((-mw_avg * fwd_rxn_rates[32] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[32] / 3.99480000e+01));
  jac[141] += (2.0 * (-mw_avg * fwd_rxn_rates[33] / 3.99480000e+01));
  jac[141] += -1.0 * ((-mw_avg * fwd_rxn_rates[44] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[44] / 3.99480000e+01));
  jac[141] += ((-mw_avg * fwd_rxn_rates[45] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[45] / 3.99480000e+01));
  jac[141] += -1.0 * ((-mw_avg * fwd_rxn_rates[46] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[46] / 3.99480000e+01));
  jac[141] += ((-mw_avg * fwd_rxn_rates[47] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[47] / 3.99480000e+01));
  jac[141] += sp_rates[0] * mw_avg / 3.99480000e+01;
  jac[141] *= 1.00797000e+00 / rho;

  //partial of omega_H wrt Y_HE;
  jac[155] = -1.0 * ((-mw_avg * fwd_rxn_rates[0] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[0] / 4.00260000e+00));
  jac[155] += ((-mw_avg * fwd_rxn_rates[1] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[1] / 4.00260000e+00));
  jac[155] += ((-mw_avg * fwd_rxn_rates[2] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[2] / 4.00260000e+00));
  jac[155] += -1.0 * ((-mw_avg * fwd_rxn_rates[3] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[3] / 4.00260000e+00));
  jac[155] += ((-mw_avg * fwd_rxn_rates[4] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[4] / 4.00260000e+00));
  jac[155] += -1.0 * ((-mw_avg * fwd_rxn_rates[5] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[5] / 4.00260000e+00));
  jac[155] += ((-mw_avg * fwd_rxn_rates[6] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[6] / 4.00260000e+00));
  jac[155] += -1.0 * ((-mw_avg * fwd_rxn_rates[7] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[7] / 4.00260000e+00));
  jac[155] += 2.0 * ((-mw_avg * pres_mod[0] / 4.00260000e+00) * fwd_rxn_rates[10] + pres_mod[0] * ((-mw_avg * fwd_rxn_rates[10] / 4.00260000e+00)));
  jac[155] += -2.0 * ((-mw_avg * pres_mod[1] / 4.00260000e+00) * fwd_rxn_rates[11] + pres_mod[1] * (2.0 * (-mw_avg * fwd_rxn_rates[11] / 4.00260000e+00)));
  jac[155] += 2.0 * ((-mw_avg * fwd_rxn_rates[12] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[12] / 4.00260000e+00));
  jac[155] += -2.0 * (2.0 * (-mw_avg * fwd_rxn_rates[13] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[13] / 4.00260000e+00));
  jac[155] += 2.0 * ((-mw_avg * fwd_rxn_rates[14] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[14] / 4.00260000e+00 + exp(4.32112625e+01 - 1.1 * logT - (5.25257556e+04 / T)) * (rho / 4.00260000e+00) * conc[1]));
  jac[155] += -2.0 * (2.0 * (-mw_avg * fwd_rxn_rates[15] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[15] / 4.00260000e+00 + exp(4.20017378e+01 - 1.1234 * logT - (2.21510944e+01 / T)) * (rho / 4.00260000e+00) * conc[0] * conc[0]));
  jac[155] += -1.0 * ((-mw_avg * pres_mod[4] / 4.00260000e+00 + 0.75 * rho / 4.00260000e+00) * fwd_rxn_rates[22] + pres_mod[4] * ((-mw_avg * fwd_rxn_rates[22] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[22] / 4.00260000e+00)));
  jac[155] += ((-mw_avg * pres_mod[5] / 4.00260000e+00 + 0.75 * rho / 4.00260000e+00) * fwd_rxn_rates[23] + pres_mod[5] * ((-mw_avg * fwd_rxn_rates[23] / 4.00260000e+00)));
  jac[155] += ((-mw_avg * pres_mod[6] / 4.00260000e+00 + 1.1 * rho / 4.00260000e+00) * fwd_rxn_rates[24] + pres_mod[6] * ((-mw_avg * fwd_rxn_rates[24] / 4.00260000e+00)));
  jac[155] += -1.0 * ((-mw_avg * pres_mod[7] / 4.00260000e+00 + 1.1 * rho / 4.00260000e+00) * fwd_rxn_rates[25] + pres_mod[7] * ((-mw_avg * fwd_rxn_rates[25] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[25] / 4.00260000e+00)));
  jac[155] += (2.0 * (-mw_avg * fwd_rxn_rates[26] / 4.00260000e+00));
  jac[155] += -1.0 * ((-mw_avg * fwd_rxn_rates[27] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[27] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[27] / 4.00260000e+00));
  Pr = (m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * (exp(1.87346128e+01 - 2.16 * logT - (2.64088106e+02 / T)));
  Fcent = 5.0000e-01 * exp(-T / 1.0000e-30) + 5.0000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[155] += -1.0 * (pres_mod[8] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 4.00260000e+00 + rho * 0.8 / ((m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * 4.00260000e+00)) * fwd_rxn_rates[28] + pres_mod[8] * ((-mw_avg * fwd_rxn_rates[28] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[28] / 4.00260000e+00)));
  Pr = (m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * (exp(1.87345962e+01 - 2.159967 * logT - (2.64188750e+02 / T)));
  Fcent = 5.0000e-01 * exp(-T / 1.0000e-30) + 5.0000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[155] += (pres_mod[9] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 4.00260000e+00 + rho * 0.8 / ((m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * 4.00260000e+00)) * fwd_rxn_rates[29] + pres_mod[9] * ((-mw_avg * fwd_rxn_rates[29] / 4.00260000e+00)));
  jac[155] += -1.0 * ((-mw_avg * fwd_rxn_rates[30] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[30] / 4.00260000e+00));
  jac[155] += ((-mw_avg * fwd_rxn_rates[31] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[31] / 4.00260000e+00));
  jac[155] += -1.0 * ((-mw_avg * fwd_rxn_rates[32] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[32] / 4.00260000e+00));
  jac[155] += (2.0 * (-mw_avg * fwd_rxn_rates[33] / 4.00260000e+00));
  jac[155] += -1.0 * ((-mw_avg * fwd_rxn_rates[44] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[44] / 4.00260000e+00));
  jac[155] += ((-mw_avg * fwd_rxn_rates[45] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[45] / 4.00260000e+00));
  jac[155] += -1.0 * ((-mw_avg * fwd_rxn_rates[46] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[46] / 4.00260000e+00));
  jac[155] += ((-mw_avg * fwd_rxn_rates[47] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[47] / 4.00260000e+00));
  jac[155] += sp_rates[0] * mw_avg / 4.00260000e+00;
  jac[155] *= 1.00797000e+00 / rho;

  //partial of omega_H wrt Y_CO;
  jac[169] = -1.0 * ((-mw_avg * fwd_rxn_rates[0] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[0] / 2.80105500e+01));
  jac[169] += ((-mw_avg * fwd_rxn_rates[1] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[1] / 2.80105500e+01));
  jac[169] += ((-mw_avg * fwd_rxn_rates[2] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[2] / 2.80105500e+01));
  jac[169] += -1.0 * ((-mw_avg * fwd_rxn_rates[3] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[3] / 2.80105500e+01));
  jac[169] += ((-mw_avg * fwd_rxn_rates[4] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[4] / 2.80105500e+01));
  jac[169] += -1.0 * ((-mw_avg * fwd_rxn_rates[5] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[5] / 2.80105500e+01));
  jac[169] += ((-mw_avg * fwd_rxn_rates[6] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[6] / 2.80105500e+01));
  jac[169] += -1.0 * ((-mw_avg * fwd_rxn_rates[7] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[7] / 2.80105500e+01));
  jac[169] += 2.0 * ((-mw_avg * pres_mod[0] / 2.80105500e+01 + 1.9 * rho / 2.80105500e+01) * fwd_rxn_rates[10] + pres_mod[0] * ((-mw_avg * fwd_rxn_rates[10] / 2.80105500e+01)));
  jac[169] += -2.0 * ((-mw_avg * pres_mod[1] / 2.80105500e+01 + 1.9 * rho / 2.80105500e+01) * fwd_rxn_rates[11] + pres_mod[1] * (2.0 * (-mw_avg * fwd_rxn_rates[11] / 2.80105500e+01)));
  jac[169] += 2.0 * ((-mw_avg * fwd_rxn_rates[12] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[12] / 2.80105500e+01));
  jac[169] += -2.0 * (2.0 * (-mw_avg * fwd_rxn_rates[13] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[13] / 2.80105500e+01));
  jac[169] += 2.0 * ((-mw_avg * fwd_rxn_rates[14] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[14] / 2.80105500e+01));
  jac[169] += -2.0 * (2.0 * (-mw_avg * fwd_rxn_rates[15] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[15] / 2.80105500e+01));
  jac[169] += -1.0 * ((-mw_avg * pres_mod[4] / 2.80105500e+01 + 1.9 * rho / 2.80105500e+01) * fwd_rxn_rates[22] + pres_mod[4] * ((-mw_avg * fwd_rxn_rates[22] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[22] / 2.80105500e+01)));
  jac[169] += ((-mw_avg * pres_mod[5] / 2.80105500e+01 + 1.9 * rho / 2.80105500e+01) * fwd_rxn_rates[23] + pres_mod[5] * ((-mw_avg * fwd_rxn_rates[23] / 2.80105500e+01)));
  jac[169] += ((-mw_avg * pres_mod[6] / 2.80105500e+01 + 1.9 * rho / 2.80105500e+01) * fwd_rxn_rates[24] + pres_mod[6] * ((-mw_avg * fwd_rxn_rates[24] / 2.80105500e+01)));
  jac[169] += -1.0 * ((-mw_avg * pres_mod[7] / 2.80105500e+01 + 1.9 * rho / 2.80105500e+01) * fwd_rxn_rates[25] + pres_mod[7] * ((-mw_avg * fwd_rxn_rates[25] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[25] / 2.80105500e+01)));
  jac[169] += (2.0 * (-mw_avg * fwd_rxn_rates[26] / 2.80105500e+01));
  jac[169] += -1.0 * ((-mw_avg * fwd_rxn_rates[27] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[27] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[27] / 2.80105500e+01));
  Pr = (m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * (exp(1.87346128e+01 - 2.16 * logT - (2.64088106e+02 / T)));
  Fcent = 5.0000e-01 * exp(-T / 1.0000e-30) + 5.0000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[169] += -1.0 * (pres_mod[8] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 2.80105500e+01 + rho * 1.9 / ((m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * 2.80105500e+01)) * fwd_rxn_rates[28] + pres_mod[8] * ((-mw_avg * fwd_rxn_rates[28] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[28] / 2.80105500e+01)));
  Pr = (m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * (exp(1.87345962e+01 - 2.159967 * logT - (2.64188750e+02 / T)));
  Fcent = 5.0000e-01 * exp(-T / 1.0000e-30) + 5.0000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[169] += (pres_mod[9] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 2.80105500e+01 + rho * 1.9 / ((m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * 2.80105500e+01)) * fwd_rxn_rates[29] + pres_mod[9] * ((-mw_avg * fwd_rxn_rates[29] / 2.80105500e+01)));
  jac[169] += -1.0 * ((-mw_avg * fwd_rxn_rates[30] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[30] / 2.80105500e+01));
  jac[169] += ((-mw_avg * fwd_rxn_rates[31] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[31] / 2.80105500e+01));
  jac[169] += -1.0 * ((-mw_avg * fwd_rxn_rates[32] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[32] / 2.80105500e+01));
  jac[169] += (2.0 * (-mw_avg * fwd_rxn_rates[33] / 2.80105500e+01));
  jac[169] += -1.0 * ((-mw_avg * fwd_rxn_rates[44] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[44] / 2.80105500e+01));
  jac[169] += ((-mw_avg * fwd_rxn_rates[45] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[45] / 2.80105500e+01));
  jac[169] += -1.0 * ((-mw_avg * fwd_rxn_rates[46] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[46] / 2.80105500e+01));
  jac[169] += ((-mw_avg * fwd_rxn_rates[47] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[47] / 2.80105500e+01));
  jac[169] += sp_rates[0] * mw_avg / 2.80105500e+01;
  jac[169] *= 1.00797000e+00 / rho;

  //partial of omega_H wrt Y_CO2;
  jac[183] = -1.0 * ((-mw_avg * fwd_rxn_rates[0] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[0] / 4.40099500e+01));
  jac[183] += ((-mw_avg * fwd_rxn_rates[1] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[1] / 4.40099500e+01));
  jac[183] += ((-mw_avg * fwd_rxn_rates[2] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[2] / 4.40099500e+01));
  jac[183] += -1.0 * ((-mw_avg * fwd_rxn_rates[3] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[3] / 4.40099500e+01));
  jac[183] += ((-mw_avg * fwd_rxn_rates[4] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[4] / 4.40099500e+01));
  jac[183] += -1.0 * ((-mw_avg * fwd_rxn_rates[5] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[5] / 4.40099500e+01));
  jac[183] += ((-mw_avg * fwd_rxn_rates[6] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[6] / 4.40099500e+01));
  jac[183] += -1.0 * ((-mw_avg * fwd_rxn_rates[7] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[7] / 4.40099500e+01));
  jac[183] += 2.0 * ((-mw_avg * pres_mod[0] / 4.40099500e+01 + 3.8 * rho / 4.40099500e+01) * fwd_rxn_rates[10] + pres_mod[0] * ((-mw_avg * fwd_rxn_rates[10] / 4.40099500e+01)));
  jac[183] += -2.0 * ((-mw_avg * pres_mod[1] / 4.40099500e+01 + 3.8 * rho / 4.40099500e+01) * fwd_rxn_rates[11] + pres_mod[1] * (2.0 * (-mw_avg * fwd_rxn_rates[11] / 4.40099500e+01)));
  jac[183] += 2.0 * ((-mw_avg * fwd_rxn_rates[12] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[12] / 4.40099500e+01));
  jac[183] += -2.0 * (2.0 * (-mw_avg * fwd_rxn_rates[13] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[13] / 4.40099500e+01));
  jac[183] += 2.0 * ((-mw_avg * fwd_rxn_rates[14] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[14] / 4.40099500e+01));
  jac[183] += -2.0 * (2.0 * (-mw_avg * fwd_rxn_rates[15] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[15] / 4.40099500e+01));
  jac[183] += -1.0 * ((-mw_avg * pres_mod[4] / 4.40099500e+01 + 3.8 * rho / 4.40099500e+01) * fwd_rxn_rates[22] + pres_mod[4] * ((-mw_avg * fwd_rxn_rates[22] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[22] / 4.40099500e+01)));
  jac[183] += ((-mw_avg * pres_mod[5] / 4.40099500e+01 + 3.8 * rho / 4.40099500e+01) * fwd_rxn_rates[23] + pres_mod[5] * ((-mw_avg * fwd_rxn_rates[23] / 4.40099500e+01)));
  jac[183] += ((-mw_avg * pres_mod[6] / 4.40099500e+01 + 3.8 * rho / 4.40099500e+01) * fwd_rxn_rates[24] + pres_mod[6] * ((-mw_avg * fwd_rxn_rates[24] / 4.40099500e+01)));
  jac[183] += -1.0 * ((-mw_avg * pres_mod[7] / 4.40099500e+01 + 3.8 * rho / 4.40099500e+01) * fwd_rxn_rates[25] + pres_mod[7] * ((-mw_avg * fwd_rxn_rates[25] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[25] / 4.40099500e+01)));
  jac[183] += (2.0 * (-mw_avg * fwd_rxn_rates[26] / 4.40099500e+01));
  jac[183] += -1.0 * ((-mw_avg * fwd_rxn_rates[27] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[27] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[27] / 4.40099500e+01));
  Pr = (m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * (exp(1.87346128e+01 - 2.16 * logT - (2.64088106e+02 / T)));
  Fcent = 5.0000e-01 * exp(-T / 1.0000e-30) + 5.0000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[183] += -1.0 * (pres_mod[8] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 4.40099500e+01 + rho * 3.8 / ((m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * 4.40099500e+01)) * fwd_rxn_rates[28] + pres_mod[8] * ((-mw_avg * fwd_rxn_rates[28] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[28] / 4.40099500e+01)));
  Pr = (m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * (exp(1.87345962e+01 - 2.159967 * logT - (2.64188750e+02 / T)));
  Fcent = 5.0000e-01 * exp(-T / 1.0000e-30) + 5.0000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[183] += (pres_mod[9] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 4.40099500e+01 + rho * 3.8 / ((m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * 4.40099500e+01)) * fwd_rxn_rates[29] + pres_mod[9] * ((-mw_avg * fwd_rxn_rates[29] / 4.40099500e+01)));
  jac[183] += -1.0 * ((-mw_avg * fwd_rxn_rates[30] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[30] / 4.40099500e+01));
  jac[183] += ((-mw_avg * fwd_rxn_rates[31] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[31] / 4.40099500e+01));
  jac[183] += -1.0 * ((-mw_avg * fwd_rxn_rates[32] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[32] / 4.40099500e+01));
  jac[183] += (2.0 * (-mw_avg * fwd_rxn_rates[33] / 4.40099500e+01));
  jac[183] += -1.0 * ((-mw_avg * fwd_rxn_rates[44] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[44] / 4.40099500e+01));
  jac[183] += ((-mw_avg * fwd_rxn_rates[45] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[45] / 4.40099500e+01));
  jac[183] += -1.0 * ((-mw_avg * fwd_rxn_rates[46] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[46] / 4.40099500e+01));
  jac[183] += ((-mw_avg * fwd_rxn_rates[47] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[47] / 4.40099500e+01));
  jac[183] += sp_rates[0] * mw_avg / 4.40099500e+01;
  jac[183] *= 1.00797000e+00 / rho;

  //partial of omega_H2 wrt T;
  jac[2] = -1.0 * ((1.0 / T) * (fwd_rxn_rates[2] * ((3.99956606e+03 / T) + 1.0 - 2.0)));
  jac[2] += ((1.0 / T) * (fwd_rxn_rates[3] * (-5.35330000e-02 + (3.32706731e+03 / T) + 1.0 - 2.0)));
  jac[2] += -1.0 * ((1.0 / T) * (fwd_rxn_rates[4] * ((9.64666348e+03 / T) + 1.0 - 2.0)));
  jac[2] += ((1.0 / T) * (fwd_rxn_rates[5] * (-5.35330000e-02 + (8.97436602e+03 / T) + 1.0 - 2.0)));
  jac[2] += -1.0 * ((1.0 / T) * (fwd_rxn_rates[6] * (1.51000000e+00 + (1.72603316e+03 / T) + 1.0 - 2.0)));
  jac[2] += ((1.0 / T) * (fwd_rxn_rates[7] * (1.18290000e+00 + (9.55507805e+03 / T) + 1.0 - 2.0)));
  jac[2] += -1.0 * ((-pres_mod[0] * fwd_rxn_rates[10] / T) + (pres_mod[0] / T) * (fwd_rxn_rates[10] * (-1.40000000e+00 + (5.25257556e+04 / T) + 1.0 - 1.0)));
  jac[2] += ((-pres_mod[1] * fwd_rxn_rates[11] / T) + (pres_mod[1] / T) * (fwd_rxn_rates[11] * (-1.42340000e+00 + (2.21510944e+01 / T) + 1.0 - 2.0)));
  jac[2] += -1.0 * ((1.0 / T) * (fwd_rxn_rates[12] * (-1.10000000e+00 + (5.25257556e+04 / T) + 1.0 - 2.0)));
  jac[2] += ((1.0 / T) * (fwd_rxn_rates[13] * (-1.12340000e+00 + (2.21510944e+01 / T) + 1.0 - 3.0)));
  jac[2] += -1.0 * ((1.0 / T) * (fwd_rxn_rates[14] * (-1.10000000e+00 + (5.25257556e+04 / T) + 1.0 - 2.0)));
  jac[2] += ((1.0 / T) * (fwd_rxn_rates[15] * (-1.12340000e+00 + (2.21510944e+01 / T) + 1.0 - 3.0)));
  jac[2] += ((1.0 / T) * (fwd_rxn_rates[30] * (2.09000000e+00 + (-7.30167382e+02 / T) + 1.0 - 2.0)));
  jac[2] += -1.0 * ((1.0 / T) * (fwd_rxn_rates[31] * (2.60280000e+00 + (2.65552467e+04 / T) + 1.0 - 2.0)));
  jac[2] += ((1.0 / T) * (fwd_rxn_rates[46] * ((4.00057249e+03 / T) + 1.0 - 2.0)));
  jac[2] += -1.0 * ((1.0 / T) * (fwd_rxn_rates[47] * (7.47310000e-01 + (1.19941692e+04 / T) + 1.0 - 2.0)));
  jac[2] *= 2.01594000e+00 / rho;

  //partial of omega_H2 wrt Y_H;
  jac[16] = -1.0 * ((-mw_avg * fwd_rxn_rates[2] / 1.00797000e+00) + (-mw_avg * fwd_rxn_rates[2] / 1.00797000e+00));
  jac[16] += ((-mw_avg * fwd_rxn_rates[3] / 1.00797000e+00 + exp(2.85866095e+01 - 0.053533 * logT - (3.32706731e+03 / T)) * (rho / 1.00797000e+00) * conc[3]) + (-mw_avg * fwd_rxn_rates[3] / 1.00797000e+00));
  jac[16] += -1.0 * ((-mw_avg * fwd_rxn_rates[4] / 1.00797000e+00) + (-mw_avg * fwd_rxn_rates[4] / 1.00797000e+00));
  jac[16] += ((-mw_avg * fwd_rxn_rates[5] / 1.00797000e+00 + exp(3.40258987e+01 - 0.053533 * logT - (8.97436602e+03 / T)) * (rho / 1.00797000e+00) * conc[3]) + (-mw_avg * fwd_rxn_rates[5] / 1.00797000e+00));
  jac[16] += -1.0 * ((-mw_avg * fwd_rxn_rates[6] / 1.00797000e+00) + (-mw_avg * fwd_rxn_rates[6] / 1.00797000e+00));
  jac[16] += ((-mw_avg * fwd_rxn_rates[7] / 1.00797000e+00 + exp(2.34232839e+01 + 1.1829 * logT - (9.55507805e+03 / T)) * (rho / 1.00797000e+00) * conc[4]) + (-mw_avg * fwd_rxn_rates[7] / 1.00797000e+00));
  jac[16] += -1.0 * ((-mw_avg * pres_mod[0] / 1.00797000e+00 + rho / 1.00797000e+00) * fwd_rxn_rates[10] + pres_mod[0] * ((-mw_avg * fwd_rxn_rates[10] / 1.00797000e+00)));
  jac[16] += ((-mw_avg * pres_mod[1] / 1.00797000e+00 + rho / 1.00797000e+00) * fwd_rxn_rates[11] + pres_mod[1] * (2.0 * (-mw_avg * fwd_rxn_rates[11] / 1.00797000e+00 + exp(4.40606374e+01 - 1.4234 * logT - (2.21510944e+01 / T)) * (rho / 1.00797000e+00) * conc[0])));
  jac[16] += -1.0 * ((-mw_avg * fwd_rxn_rates[12] / 1.00797000e+00) + (-mw_avg * fwd_rxn_rates[12] / 1.00797000e+00));
  jac[16] += (2.0 * (-mw_avg * fwd_rxn_rates[13] / 1.00797000e+00 + exp(4.20017378e+01 - 1.1234 * logT - (2.21510944e+01 / T)) * (rho / 1.00797000e+00) * conc[0] * conc[9]) + (-mw_avg * fwd_rxn_rates[13] / 1.00797000e+00));
  jac[16] += -1.0 * ((-mw_avg * fwd_rxn_rates[14] / 1.00797000e+00) + (-mw_avg * fwd_rxn_rates[14] / 1.00797000e+00));
  jac[16] += (2.0 * (-mw_avg * fwd_rxn_rates[15] / 1.00797000e+00 + exp(4.20017378e+01 - 1.1234 * logT - (2.21510944e+01 / T)) * (rho / 1.00797000e+00) * conc[0] * conc[10]) + (-mw_avg * fwd_rxn_rates[15] / 1.00797000e+00));
  jac[16] += ((-mw_avg * fwd_rxn_rates[30] / 1.00797000e+00 + exp(1.48271115e+01 + 2.09 * logT - (-7.30167382e+02 / T)) * (rho / 1.00797000e+00) * conc[6]) + (-mw_avg * fwd_rxn_rates[30] / 1.00797000e+00));
  jac[16] += -1.0 * ((-mw_avg * fwd_rxn_rates[31] / 1.00797000e+00) + (-mw_avg * fwd_rxn_rates[31] / 1.00797000e+00));
  jac[16] += ((-mw_avg * fwd_rxn_rates[46] / 1.00797000e+00 + exp(3.15063801e+01 - (4.00057249e+03 / T)) * (rho / 1.00797000e+00) * conc[7]) + (-mw_avg * fwd_rxn_rates[46] / 1.00797000e+00));
  jac[16] += -1.0 * ((-mw_avg * fwd_rxn_rates[47] / 1.00797000e+00) + (-mw_avg * fwd_rxn_rates[47] / 1.00797000e+00));
  jac[16] += sp_rates[1] * mw_avg / 1.00797000e+00;
  jac[16] *= 2.01594000e+00 / rho;

  //partial of omega_H2 wrt Y_H2;
  jac[30] = -1.0 * ((-mw_avg * fwd_rxn_rates[2] / 2.01594000e+00 + exp(2.89707478e+01 - (3.99956606e+03 / T)) * (rho / 2.01594000e+00) * conc[2]) + (-mw_avg * fwd_rxn_rates[2] / 2.01594000e+00));
  jac[30] += ((-mw_avg * fwd_rxn_rates[3] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[3] / 2.01594000e+00));
  jac[30] += -1.0 * ((-mw_avg * fwd_rxn_rates[4] / 2.01594000e+00 + exp(3.44100335e+01 - (9.64666348e+03 / T)) * (rho / 2.01594000e+00) * conc[2]) + (-mw_avg * fwd_rxn_rates[4] / 2.01594000e+00));
  jac[30] += ((-mw_avg * fwd_rxn_rates[5] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[5] / 2.01594000e+00));
  jac[30] += -1.0 * ((-mw_avg * fwd_rxn_rates[6] / 2.01594000e+00 + exp(1.91907890e+01 + 1.51 * logT - (1.72603316e+03 / T)) * (rho / 2.01594000e+00) * conc[3]) + (-mw_avg * fwd_rxn_rates[6] / 2.01594000e+00));
  jac[30] += ((-mw_avg * fwd_rxn_rates[7] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[7] / 2.01594000e+00));
  jac[30] += -1.0 * ((-mw_avg * pres_mod[0] / 2.01594000e+00 + 2.5 * rho / 2.01594000e+00) * fwd_rxn_rates[10] + pres_mod[0] * ((-mw_avg * fwd_rxn_rates[10] / 2.01594000e+00 + exp(4.52701605e+01 - 1.4 * logT - (5.25257556e+04 / T)) * (rho / 2.01594000e+00))));
  jac[30] += ((-mw_avg * pres_mod[1] / 2.01594000e+00 + 2.5 * rho / 2.01594000e+00) * fwd_rxn_rates[11] + pres_mod[1] * (2.0 * (-mw_avg * fwd_rxn_rates[11] / 2.01594000e+00)));
  jac[30] += -1.0 * ((-mw_avg * fwd_rxn_rates[12] / 2.01594000e+00 + exp(4.32112625e+01 - 1.1 * logT - (5.25257556e+04 / T)) * (rho / 2.01594000e+00) * conc[9]) + (-mw_avg * fwd_rxn_rates[12] / 2.01594000e+00));
  jac[30] += (2.0 * (-mw_avg * fwd_rxn_rates[13] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[13] / 2.01594000e+00));
  jac[30] += -1.0 * ((-mw_avg * fwd_rxn_rates[14] / 2.01594000e+00 + exp(4.32112625e+01 - 1.1 * logT - (5.25257556e+04 / T)) * (rho / 2.01594000e+00) * conc[10]) + (-mw_avg * fwd_rxn_rates[14] / 2.01594000e+00));
  jac[30] += (2.0 * (-mw_avg * fwd_rxn_rates[15] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[15] / 2.01594000e+00));
  jac[30] += ((-mw_avg * fwd_rxn_rates[30] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[30] / 2.01594000e+00));
  jac[30] += -1.0 * ((-mw_avg * fwd_rxn_rates[31] / 2.01594000e+00 + exp(1.17897235e+01 + 2.6028 * logT - (2.65552467e+04 / T)) * (rho / 2.01594000e+00) * conc[5]) + (-mw_avg * fwd_rxn_rates[31] / 2.01594000e+00));
  jac[30] += ((-mw_avg * fwd_rxn_rates[46] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[46] / 2.01594000e+00));
  jac[30] += -1.0 * ((-mw_avg * fwd_rxn_rates[47] / 2.01594000e+00 + exp(2.47202181e+01 + 0.74731 * logT - (1.19941692e+04 / T)) * (rho / 2.01594000e+00) * conc[6]) + (-mw_avg * fwd_rxn_rates[47] / 2.01594000e+00));
  jac[30] += sp_rates[1] * mw_avg / 2.01594000e+00;
  jac[30] *= 2.01594000e+00 / rho;

  //partial of omega_H2 wrt Y_O;
  jac[44] = -1.0 * ((-mw_avg * fwd_rxn_rates[2] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[2] / 1.59994000e+01 + exp(2.89707478e+01 - (3.99956606e+03 / T)) * (rho / 1.59994000e+01) * conc[1]));
  jac[44] += ((-mw_avg * fwd_rxn_rates[3] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[3] / 1.59994000e+01));
  jac[44] += -1.0 * ((-mw_avg * fwd_rxn_rates[4] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[4] / 1.59994000e+01 + exp(3.44100335e+01 - (9.64666348e+03 / T)) * (rho / 1.59994000e+01) * conc[1]));
  jac[44] += ((-mw_avg * fwd_rxn_rates[5] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[5] / 1.59994000e+01));
  jac[44] += -1.0 * ((-mw_avg * fwd_rxn_rates[6] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[6] / 1.59994000e+01));
  jac[44] += ((-mw_avg * fwd_rxn_rates[7] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[7] / 1.59994000e+01));
  jac[44] += -1.0 * ((-mw_avg * pres_mod[0] / 1.59994000e+01 + rho / 1.59994000e+01) * fwd_rxn_rates[10] + pres_mod[0] * ((-mw_avg * fwd_rxn_rates[10] / 1.59994000e+01)));
  jac[44] += ((-mw_avg * pres_mod[1] / 1.59994000e+01 + rho / 1.59994000e+01) * fwd_rxn_rates[11] + pres_mod[1] * (2.0 * (-mw_avg * fwd_rxn_rates[11] / 1.59994000e+01)));
  jac[44] += -1.0 * ((-mw_avg * fwd_rxn_rates[12] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[12] / 1.59994000e+01));
  jac[44] += (2.0 * (-mw_avg * fwd_rxn_rates[13] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[13] / 1.59994000e+01));
  jac[44] += -1.0 * ((-mw_avg * fwd_rxn_rates[14] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[14] / 1.59994000e+01));
  jac[44] += (2.0 * (-mw_avg * fwd_rxn_rates[15] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[15] / 1.59994000e+01));
  jac[44] += ((-mw_avg * fwd_rxn_rates[30] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[30] / 1.59994000e+01));
  jac[44] += -1.0 * ((-mw_avg * fwd_rxn_rates[31] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[31] / 1.59994000e+01));
  jac[44] += ((-mw_avg * fwd_rxn_rates[46] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[46] / 1.59994000e+01));
  jac[44] += -1.0 * ((-mw_avg * fwd_rxn_rates[47] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[47] / 1.59994000e+01));
  jac[44] += sp_rates[1] * mw_avg / 1.59994000e+01;
  jac[44] *= 2.01594000e+00 / rho;

  //partial of omega_H2 wrt Y_OH;
  jac[58] = -1.0 * ((-mw_avg * fwd_rxn_rates[2] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[2] / 1.70073700e+01));
  jac[58] += ((-mw_avg * fwd_rxn_rates[3] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[3] / 1.70073700e+01 + exp(2.85866095e+01 - 0.053533 * logT - (3.32706731e+03 / T)) * (rho / 1.70073700e+01) * conc[0]));
  jac[58] += -1.0 * ((-mw_avg * fwd_rxn_rates[4] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[4] / 1.70073700e+01));
  jac[58] += ((-mw_avg * fwd_rxn_rates[5] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[5] / 1.70073700e+01 + exp(3.40258987e+01 - 0.053533 * logT - (8.97436602e+03 / T)) * (rho / 1.70073700e+01) * conc[0]));
  jac[58] += -1.0 * ((-mw_avg * fwd_rxn_rates[6] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[6] / 1.70073700e+01 + exp(1.91907890e+01 + 1.51 * logT - (1.72603316e+03 / T)) * (rho / 1.70073700e+01) * conc[1]));
  jac[58] += ((-mw_avg * fwd_rxn_rates[7] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[7] / 1.70073700e+01));
  jac[58] += -1.0 * ((-mw_avg * pres_mod[0] / 1.70073700e+01 + rho / 1.70073700e+01) * fwd_rxn_rates[10] + pres_mod[0] * ((-mw_avg * fwd_rxn_rates[10] / 1.70073700e+01)));
  jac[58] += ((-mw_avg * pres_mod[1] / 1.70073700e+01 + rho / 1.70073700e+01) * fwd_rxn_rates[11] + pres_mod[1] * (2.0 * (-mw_avg * fwd_rxn_rates[11] / 1.70073700e+01)));
  jac[58] += -1.0 * ((-mw_avg * fwd_rxn_rates[12] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[12] / 1.70073700e+01));
  jac[58] += (2.0 * (-mw_avg * fwd_rxn_rates[13] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[13] / 1.70073700e+01));
  jac[58] += -1.0 * ((-mw_avg * fwd_rxn_rates[14] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[14] / 1.70073700e+01));
  jac[58] += (2.0 * (-mw_avg * fwd_rxn_rates[15] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[15] / 1.70073700e+01));
  jac[58] += ((-mw_avg * fwd_rxn_rates[30] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[30] / 1.70073700e+01));
  jac[58] += -1.0 * ((-mw_avg * fwd_rxn_rates[31] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[31] / 1.70073700e+01));
  jac[58] += ((-mw_avg * fwd_rxn_rates[46] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[46] / 1.70073700e+01));
  jac[58] += -1.0 * ((-mw_avg * fwd_rxn_rates[47] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[47] / 1.70073700e+01));
  jac[58] += sp_rates[1] * mw_avg / 1.70073700e+01;
  jac[58] *= 2.01594000e+00 / rho;

  //partial of omega_H2 wrt Y_H2O;
  jac[72] = -1.0 * ((-mw_avg * fwd_rxn_rates[2] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[2] / 1.80153400e+01));
  jac[72] += ((-mw_avg * fwd_rxn_rates[3] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[3] / 1.80153400e+01));
  jac[72] += -1.0 * ((-mw_avg * fwd_rxn_rates[4] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[4] / 1.80153400e+01));
  jac[72] += ((-mw_avg * fwd_rxn_rates[5] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[5] / 1.80153400e+01));
  jac[72] += -1.0 * ((-mw_avg * fwd_rxn_rates[6] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[6] / 1.80153400e+01));
  jac[72] += ((-mw_avg * fwd_rxn_rates[7] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[7] / 1.80153400e+01 + exp(2.34232839e+01 + 1.1829 * logT - (9.55507805e+03 / T)) * (rho / 1.80153400e+01) * conc[0]));
  jac[72] += -1.0 * ((-mw_avg * pres_mod[0] / 1.80153400e+01 + 12.0 * rho / 1.80153400e+01) * fwd_rxn_rates[10] + pres_mod[0] * ((-mw_avg * fwd_rxn_rates[10] / 1.80153400e+01)));
  jac[72] += ((-mw_avg * pres_mod[1] / 1.80153400e+01 + 12.0 * rho / 1.80153400e+01) * fwd_rxn_rates[11] + pres_mod[1] * (2.0 * (-mw_avg * fwd_rxn_rates[11] / 1.80153400e+01)));
  jac[72] += -1.0 * ((-mw_avg * fwd_rxn_rates[12] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[12] / 1.80153400e+01));
  jac[72] += (2.0 * (-mw_avg * fwd_rxn_rates[13] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[13] / 1.80153400e+01));
  jac[72] += -1.0 * ((-mw_avg * fwd_rxn_rates[14] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[14] / 1.80153400e+01));
  jac[72] += (2.0 * (-mw_avg * fwd_rxn_rates[15] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[15] / 1.80153400e+01));
  jac[72] += ((-mw_avg * fwd_rxn_rates[30] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[30] / 1.80153400e+01));
  jac[72] += -1.0 * ((-mw_avg * fwd_rxn_rates[31] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[31] / 1.80153400e+01));
  jac[72] += ((-mw_avg * fwd_rxn_rates[46] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[46] / 1.80153400e+01));
  jac[72] += -1.0 * ((-mw_avg * fwd_rxn_rates[47] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[47] / 1.80153400e+01));
  jac[72] += sp_rates[1] * mw_avg / 1.80153400e+01;
  jac[72] *= 2.01594000e+00 / rho;

  //partial of omega_H2 wrt Y_O2;
  jac[86] = -1.0 * ((-mw_avg * fwd_rxn_rates[2] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[2] / 3.19988000e+01));
  jac[86] += ((-mw_avg * fwd_rxn_rates[3] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[3] / 3.19988000e+01));
  jac[86] += -1.0 * ((-mw_avg * fwd_rxn_rates[4] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[4] / 3.19988000e+01));
  jac[86] += ((-mw_avg * fwd_rxn_rates[5] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[5] / 3.19988000e+01));
  jac[86] += -1.0 * ((-mw_avg * fwd_rxn_rates[6] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[6] / 3.19988000e+01));
  jac[86] += ((-mw_avg * fwd_rxn_rates[7] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[7] / 3.19988000e+01));
  jac[86] += -1.0 * ((-mw_avg * pres_mod[0] / 3.19988000e+01 + rho / 3.19988000e+01) * fwd_rxn_rates[10] + pres_mod[0] * ((-mw_avg * fwd_rxn_rates[10] / 3.19988000e+01)));
  jac[86] += ((-mw_avg * pres_mod[1] / 3.19988000e+01 + rho / 3.19988000e+01) * fwd_rxn_rates[11] + pres_mod[1] * (2.0 * (-mw_avg * fwd_rxn_rates[11] / 3.19988000e+01)));
  jac[86] += -1.0 * ((-mw_avg * fwd_rxn_rates[12] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[12] / 3.19988000e+01));
  jac[86] += (2.0 * (-mw_avg * fwd_rxn_rates[13] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[13] / 3.19988000e+01));
  jac[86] += -1.0 * ((-mw_avg * fwd_rxn_rates[14] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[14] / 3.19988000e+01));
  jac[86] += (2.0 * (-mw_avg * fwd_rxn_rates[15] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[15] / 3.19988000e+01));
  jac[86] += ((-mw_avg * fwd_rxn_rates[30] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[30] / 3.19988000e+01));
  jac[86] += -1.0 * ((-mw_avg * fwd_rxn_rates[31] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[31] / 3.19988000e+01 + exp(1.17897235e+01 + 2.6028 * logT - (2.65552467e+04 / T)) * (rho / 3.19988000e+01) * conc[1]));
  jac[86] += ((-mw_avg * fwd_rxn_rates[46] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[46] / 3.19988000e+01));
  jac[86] += -1.0 * ((-mw_avg * fwd_rxn_rates[47] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[47] / 3.19988000e+01));
  jac[86] += sp_rates[1] * mw_avg / 3.19988000e+01;
  jac[86] *= 2.01594000e+00 / rho;

  //partial of omega_H2 wrt Y_HO2;
  jac[100] = -1.0 * ((-mw_avg * fwd_rxn_rates[2] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[2] / 3.30067700e+01));
  jac[100] += ((-mw_avg * fwd_rxn_rates[3] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[3] / 3.30067700e+01));
  jac[100] += -1.0 * ((-mw_avg * fwd_rxn_rates[4] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[4] / 3.30067700e+01));
  jac[100] += ((-mw_avg * fwd_rxn_rates[5] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[5] / 3.30067700e+01));
  jac[100] += -1.0 * ((-mw_avg * fwd_rxn_rates[6] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[6] / 3.30067700e+01));
  jac[100] += ((-mw_avg * fwd_rxn_rates[7] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[7] / 3.30067700e+01));
  jac[100] += -1.0 * ((-mw_avg * pres_mod[0] / 3.30067700e+01 + rho / 3.30067700e+01) * fwd_rxn_rates[10] + pres_mod[0] * ((-mw_avg * fwd_rxn_rates[10] / 3.30067700e+01)));
  jac[100] += ((-mw_avg * pres_mod[1] / 3.30067700e+01 + rho / 3.30067700e+01) * fwd_rxn_rates[11] + pres_mod[1] * (2.0 * (-mw_avg * fwd_rxn_rates[11] / 3.30067700e+01)));
  jac[100] += -1.0 * ((-mw_avg * fwd_rxn_rates[12] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[12] / 3.30067700e+01));
  jac[100] += (2.0 * (-mw_avg * fwd_rxn_rates[13] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[13] / 3.30067700e+01));
  jac[100] += -1.0 * ((-mw_avg * fwd_rxn_rates[14] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[14] / 3.30067700e+01));
  jac[100] += (2.0 * (-mw_avg * fwd_rxn_rates[15] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[15] / 3.30067700e+01));
  jac[100] += ((-mw_avg * fwd_rxn_rates[30] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[30] / 3.30067700e+01 + exp(1.48271115e+01 + 2.09 * logT - (-7.30167382e+02 / T)) * (rho / 3.30067700e+01) * conc[0]));
  jac[100] += -1.0 * ((-mw_avg * fwd_rxn_rates[31] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[31] / 3.30067700e+01));
  jac[100] += ((-mw_avg * fwd_rxn_rates[46] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[46] / 3.30067700e+01));
  jac[100] += -1.0 * ((-mw_avg * fwd_rxn_rates[47] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[47] / 3.30067700e+01 + exp(2.47202181e+01 + 0.74731 * logT - (1.19941692e+04 / T)) * (rho / 3.30067700e+01) * conc[1]));
  jac[100] += sp_rates[1] * mw_avg / 3.30067700e+01;
  jac[100] *= 2.01594000e+00 / rho;

  //partial of omega_H2 wrt Y_H2O2;
  jac[114] = -1.0 * ((-mw_avg * fwd_rxn_rates[2] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[2] / 3.40147400e+01));
  jac[114] += ((-mw_avg * fwd_rxn_rates[3] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[3] / 3.40147400e+01));
  jac[114] += -1.0 * ((-mw_avg * fwd_rxn_rates[4] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[4] / 3.40147400e+01));
  jac[114] += ((-mw_avg * fwd_rxn_rates[5] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[5] / 3.40147400e+01));
  jac[114] += -1.0 * ((-mw_avg * fwd_rxn_rates[6] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[6] / 3.40147400e+01));
  jac[114] += ((-mw_avg * fwd_rxn_rates[7] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[7] / 3.40147400e+01));
  jac[114] += -1.0 * ((-mw_avg * pres_mod[0] / 3.40147400e+01 + rho / 3.40147400e+01) * fwd_rxn_rates[10] + pres_mod[0] * ((-mw_avg * fwd_rxn_rates[10] / 3.40147400e+01)));
  jac[114] += ((-mw_avg * pres_mod[1] / 3.40147400e+01 + rho / 3.40147400e+01) * fwd_rxn_rates[11] + pres_mod[1] * (2.0 * (-mw_avg * fwd_rxn_rates[11] / 3.40147400e+01)));
  jac[114] += -1.0 * ((-mw_avg * fwd_rxn_rates[12] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[12] / 3.40147400e+01));
  jac[114] += (2.0 * (-mw_avg * fwd_rxn_rates[13] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[13] / 3.40147400e+01));
  jac[114] += -1.0 * ((-mw_avg * fwd_rxn_rates[14] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[14] / 3.40147400e+01));
  jac[114] += (2.0 * (-mw_avg * fwd_rxn_rates[15] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[15] / 3.40147400e+01));
  jac[114] += ((-mw_avg * fwd_rxn_rates[30] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[30] / 3.40147400e+01));
  jac[114] += -1.0 * ((-mw_avg * fwd_rxn_rates[31] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[31] / 3.40147400e+01));
  jac[114] += ((-mw_avg * fwd_rxn_rates[46] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[46] / 3.40147400e+01 + exp(3.15063801e+01 - (4.00057249e+03 / T)) * (rho / 3.40147400e+01) * conc[0]));
  jac[114] += -1.0 * ((-mw_avg * fwd_rxn_rates[47] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[47] / 3.40147400e+01));
  jac[114] += sp_rates[1] * mw_avg / 3.40147400e+01;
  jac[114] *= 2.01594000e+00 / rho;

  //partial of omega_H2 wrt Y_N2;
  jac[128] = -1.0 * ((-mw_avg * fwd_rxn_rates[2] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[2] / 2.80134000e+01));
  jac[128] += ((-mw_avg * fwd_rxn_rates[3] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[3] / 2.80134000e+01));
  jac[128] += -1.0 * ((-mw_avg * fwd_rxn_rates[4] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[4] / 2.80134000e+01));
  jac[128] += ((-mw_avg * fwd_rxn_rates[5] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[5] / 2.80134000e+01));
  jac[128] += -1.0 * ((-mw_avg * fwd_rxn_rates[6] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[6] / 2.80134000e+01));
  jac[128] += ((-mw_avg * fwd_rxn_rates[7] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[7] / 2.80134000e+01));
  jac[128] += -1.0 * ((-mw_avg * pres_mod[0] / 2.80134000e+01 + rho / 2.80134000e+01) * fwd_rxn_rates[10] + pres_mod[0] * ((-mw_avg * fwd_rxn_rates[10] / 2.80134000e+01)));
  jac[128] += ((-mw_avg * pres_mod[1] / 2.80134000e+01 + rho / 2.80134000e+01) * fwd_rxn_rates[11] + pres_mod[1] * (2.0 * (-mw_avg * fwd_rxn_rates[11] / 2.80134000e+01)));
  jac[128] += -1.0 * ((-mw_avg * fwd_rxn_rates[12] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[12] / 2.80134000e+01));
  jac[128] += (2.0 * (-mw_avg * fwd_rxn_rates[13] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[13] / 2.80134000e+01));
  jac[128] += -1.0 * ((-mw_avg * fwd_rxn_rates[14] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[14] / 2.80134000e+01));
  jac[128] += (2.0 * (-mw_avg * fwd_rxn_rates[15] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[15] / 2.80134000e+01));
  jac[128] += ((-mw_avg * fwd_rxn_rates[30] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[30] / 2.80134000e+01));
  jac[128] += -1.0 * ((-mw_avg * fwd_rxn_rates[31] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[31] / 2.80134000e+01));
  jac[128] += ((-mw_avg * fwd_rxn_rates[46] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[46] / 2.80134000e+01));
  jac[128] += -1.0 * ((-mw_avg * fwd_rxn_rates[47] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[47] / 2.80134000e+01));
  jac[128] += sp_rates[1] * mw_avg / 2.80134000e+01;
  jac[128] *= 2.01594000e+00 / rho;

  //partial of omega_H2 wrt Y_AR;
  jac[142] = -1.0 * ((-mw_avg * fwd_rxn_rates[2] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[2] / 3.99480000e+01));
  jac[142] += ((-mw_avg * fwd_rxn_rates[3] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[3] / 3.99480000e+01));
  jac[142] += -1.0 * ((-mw_avg * fwd_rxn_rates[4] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[4] / 3.99480000e+01));
  jac[142] += ((-mw_avg * fwd_rxn_rates[5] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[5] / 3.99480000e+01));
  jac[142] += -1.0 * ((-mw_avg * fwd_rxn_rates[6] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[6] / 3.99480000e+01));
  jac[142] += ((-mw_avg * fwd_rxn_rates[7] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[7] / 3.99480000e+01));
  jac[142] += -1.0 * ((-mw_avg * pres_mod[0] / 3.99480000e+01) * fwd_rxn_rates[10] + pres_mod[0] * ((-mw_avg * fwd_rxn_rates[10] / 3.99480000e+01)));
  jac[142] += ((-mw_avg * pres_mod[1] / 3.99480000e+01) * fwd_rxn_rates[11] + pres_mod[1] * (2.0 * (-mw_avg * fwd_rxn_rates[11] / 3.99480000e+01)));
  jac[142] += -1.0 * ((-mw_avg * fwd_rxn_rates[12] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[12] / 3.99480000e+01 + exp(4.32112625e+01 - 1.1 * logT - (5.25257556e+04 / T)) * (rho / 3.99480000e+01) * conc[1]));
  jac[142] += (2.0 * (-mw_avg * fwd_rxn_rates[13] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[13] / 3.99480000e+01 + exp(4.20017378e+01 - 1.1234 * logT - (2.21510944e+01 / T)) * (rho / 3.99480000e+01) * conc[0] * conc[0]));
  jac[142] += -1.0 * ((-mw_avg * fwd_rxn_rates[14] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[14] / 3.99480000e+01));
  jac[142] += (2.0 * (-mw_avg * fwd_rxn_rates[15] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[15] / 3.99480000e+01));
  jac[142] += ((-mw_avg * fwd_rxn_rates[30] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[30] / 3.99480000e+01));
  jac[142] += -1.0 * ((-mw_avg * fwd_rxn_rates[31] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[31] / 3.99480000e+01));
  jac[142] += ((-mw_avg * fwd_rxn_rates[46] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[46] / 3.99480000e+01));
  jac[142] += -1.0 * ((-mw_avg * fwd_rxn_rates[47] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[47] / 3.99480000e+01));
  jac[142] += sp_rates[1] * mw_avg / 3.99480000e+01;
  jac[142] *= 2.01594000e+00 / rho;

  //partial of omega_H2 wrt Y_HE;
  jac[156] = -1.0 * ((-mw_avg * fwd_rxn_rates[2] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[2] / 4.00260000e+00));
  jac[156] += ((-mw_avg * fwd_rxn_rates[3] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[3] / 4.00260000e+00));
  jac[156] += -1.0 * ((-mw_avg * fwd_rxn_rates[4] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[4] / 4.00260000e+00));
  jac[156] += ((-mw_avg * fwd_rxn_rates[5] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[5] / 4.00260000e+00));
  jac[156] += -1.0 * ((-mw_avg * fwd_rxn_rates[6] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[6] / 4.00260000e+00));
  jac[156] += ((-mw_avg * fwd_rxn_rates[7] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[7] / 4.00260000e+00));
  jac[156] += -1.0 * ((-mw_avg * pres_mod[0] / 4.00260000e+00) * fwd_rxn_rates[10] + pres_mod[0] * ((-mw_avg * fwd_rxn_rates[10] / 4.00260000e+00)));
  jac[156] += ((-mw_avg * pres_mod[1] / 4.00260000e+00) * fwd_rxn_rates[11] + pres_mod[1] * (2.0 * (-mw_avg * fwd_rxn_rates[11] / 4.00260000e+00)));
  jac[156] += -1.0 * ((-mw_avg * fwd_rxn_rates[12] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[12] / 4.00260000e+00));
  jac[156] += (2.0 * (-mw_avg * fwd_rxn_rates[13] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[13] / 4.00260000e+00));
  jac[156] += -1.0 * ((-mw_avg * fwd_rxn_rates[14] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[14] / 4.00260000e+00 + exp(4.32112625e+01 - 1.1 * logT - (5.25257556e+04 / T)) * (rho / 4.00260000e+00) * conc[1]));
  jac[156] += (2.0 * (-mw_avg * fwd_rxn_rates[15] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[15] / 4.00260000e+00 + exp(4.20017378e+01 - 1.1234 * logT - (2.21510944e+01 / T)) * (rho / 4.00260000e+00) * conc[0] * conc[0]));
  jac[156] += ((-mw_avg * fwd_rxn_rates[30] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[30] / 4.00260000e+00));
  jac[156] += -1.0 * ((-mw_avg * fwd_rxn_rates[31] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[31] / 4.00260000e+00));
  jac[156] += ((-mw_avg * fwd_rxn_rates[46] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[46] / 4.00260000e+00));
  jac[156] += -1.0 * ((-mw_avg * fwd_rxn_rates[47] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[47] / 4.00260000e+00));
  jac[156] += sp_rates[1] * mw_avg / 4.00260000e+00;
  jac[156] *= 2.01594000e+00 / rho;

  //partial of omega_H2 wrt Y_CO;
  jac[170] = -1.0 * ((-mw_avg * fwd_rxn_rates[2] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[2] / 2.80105500e+01));
  jac[170] += ((-mw_avg * fwd_rxn_rates[3] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[3] / 2.80105500e+01));
  jac[170] += -1.0 * ((-mw_avg * fwd_rxn_rates[4] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[4] / 2.80105500e+01));
  jac[170] += ((-mw_avg * fwd_rxn_rates[5] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[5] / 2.80105500e+01));
  jac[170] += -1.0 * ((-mw_avg * fwd_rxn_rates[6] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[6] / 2.80105500e+01));
  jac[170] += ((-mw_avg * fwd_rxn_rates[7] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[7] / 2.80105500e+01));
  jac[170] += -1.0 * ((-mw_avg * pres_mod[0] / 2.80105500e+01 + 1.9 * rho / 2.80105500e+01) * fwd_rxn_rates[10] + pres_mod[0] * ((-mw_avg * fwd_rxn_rates[10] / 2.80105500e+01)));
  jac[170] += ((-mw_avg * pres_mod[1] / 2.80105500e+01 + 1.9 * rho / 2.80105500e+01) * fwd_rxn_rates[11] + pres_mod[1] * (2.0 * (-mw_avg * fwd_rxn_rates[11] / 2.80105500e+01)));
  jac[170] += -1.0 * ((-mw_avg * fwd_rxn_rates[12] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[12] / 2.80105500e+01));
  jac[170] += (2.0 * (-mw_avg * fwd_rxn_rates[13] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[13] / 2.80105500e+01));
  jac[170] += -1.0 * ((-mw_avg * fwd_rxn_rates[14] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[14] / 2.80105500e+01));
  jac[170] += (2.0 * (-mw_avg * fwd_rxn_rates[15] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[15] / 2.80105500e+01));
  jac[170] += ((-mw_avg * fwd_rxn_rates[30] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[30] / 2.80105500e+01));
  jac[170] += -1.0 * ((-mw_avg * fwd_rxn_rates[31] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[31] / 2.80105500e+01));
  jac[170] += ((-mw_avg * fwd_rxn_rates[46] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[46] / 2.80105500e+01));
  jac[170] += -1.0 * ((-mw_avg * fwd_rxn_rates[47] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[47] / 2.80105500e+01));
  jac[170] += sp_rates[1] * mw_avg / 2.80105500e+01;
  jac[170] *= 2.01594000e+00 / rho;

  //partial of omega_H2 wrt Y_CO2;
  jac[184] = -1.0 * ((-mw_avg * fwd_rxn_rates[2] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[2] / 4.40099500e+01));
  jac[184] += ((-mw_avg * fwd_rxn_rates[3] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[3] / 4.40099500e+01));
  jac[184] += -1.0 * ((-mw_avg * fwd_rxn_rates[4] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[4] / 4.40099500e+01));
  jac[184] += ((-mw_avg * fwd_rxn_rates[5] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[5] / 4.40099500e+01));
  jac[184] += -1.0 * ((-mw_avg * fwd_rxn_rates[6] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[6] / 4.40099500e+01));
  jac[184] += ((-mw_avg * fwd_rxn_rates[7] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[7] / 4.40099500e+01));
  jac[184] += -1.0 * ((-mw_avg * pres_mod[0] / 4.40099500e+01 + 3.8 * rho / 4.40099500e+01) * fwd_rxn_rates[10] + pres_mod[0] * ((-mw_avg * fwd_rxn_rates[10] / 4.40099500e+01)));
  jac[184] += ((-mw_avg * pres_mod[1] / 4.40099500e+01 + 3.8 * rho / 4.40099500e+01) * fwd_rxn_rates[11] + pres_mod[1] * (2.0 * (-mw_avg * fwd_rxn_rates[11] / 4.40099500e+01)));
  jac[184] += -1.0 * ((-mw_avg * fwd_rxn_rates[12] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[12] / 4.40099500e+01));
  jac[184] += (2.0 * (-mw_avg * fwd_rxn_rates[13] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[13] / 4.40099500e+01));
  jac[184] += -1.0 * ((-mw_avg * fwd_rxn_rates[14] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[14] / 4.40099500e+01));
  jac[184] += (2.0 * (-mw_avg * fwd_rxn_rates[15] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[15] / 4.40099500e+01));
  jac[184] += ((-mw_avg * fwd_rxn_rates[30] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[30] / 4.40099500e+01));
  jac[184] += -1.0 * ((-mw_avg * fwd_rxn_rates[31] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[31] / 4.40099500e+01));
  jac[184] += ((-mw_avg * fwd_rxn_rates[46] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[46] / 4.40099500e+01));
  jac[184] += -1.0 * ((-mw_avg * fwd_rxn_rates[47] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[47] / 4.40099500e+01));
  jac[184] += sp_rates[1] * mw_avg / 4.40099500e+01;
  jac[184] *= 2.01594000e+00 / rho;

  //partial of omega_O wrt T;
  jac[3] = ((1.0 / T) * (fwd_rxn_rates[0] * ((7.69216995e+03 / T) + 1.0 - 2.0)));
  jac[3] += -1.0 * ((1.0 / T) * (fwd_rxn_rates[1] * (4.04780000e-01 + (-7.48736077e+02 / T) + 1.0 - 2.0)));
  jac[3] += -1.0 * ((1.0 / T) * (fwd_rxn_rates[2] * ((3.99956606e+03 / T) + 1.0 - 2.0)));
  jac[3] += ((1.0 / T) * (fwd_rxn_rates[3] * (-5.35330000e-02 + (3.32706731e+03 / T) + 1.0 - 2.0)));
  jac[3] += -1.0 * ((1.0 / T) * (fwd_rxn_rates[4] * ((9.64666348e+03 / T) + 1.0 - 2.0)));
  jac[3] += ((1.0 / T) * (fwd_rxn_rates[5] * (-5.35330000e-02 + (8.97436602e+03 / T) + 1.0 - 2.0)));
  jac[3] += ((1.0 / T) * (fwd_rxn_rates[8] * (2.42000000e+00 + (-9.71208165e+02 / T) + 1.0 - 2.0)));
  jac[3] += -1.0 * ((1.0 / T) * (fwd_rxn_rates[9] * (2.14640000e+00 + (7.53063740e+03 / T) + 1.0 - 2.0)));
  jac[3] += -2.0 * ((-pres_mod[2] * fwd_rxn_rates[16] / T) + (pres_mod[2] / T) * (fwd_rxn_rates[16] * (-5.00000000e-01 + 1.0 - 2.0)));
  jac[3] += 2.0 * ((-pres_mod[3] * fwd_rxn_rates[17] / T) + (pres_mod[3] / T) * (fwd_rxn_rates[17] * (-9.34910000e-01 + (6.02702601e+04 / T) + 1.0 - 1.0)));
  jac[3] += -2.0 * ((1.0 / T) * (fwd_rxn_rates[18] * ((-8.99751398e+02 / T) + 1.0 - 3.0)));
  jac[3] += 2.0 * ((1.0 / T) * (fwd_rxn_rates[19] * (-4.34910000e-01 + (5.93745344e+04 / T) + 1.0 - 2.0)));
  jac[3] += -2.0 * ((1.0 / T) * (fwd_rxn_rates[20] * ((-8.99751398e+02 / T) + 1.0 - 3.0)));
  jac[3] += 2.0 * ((1.0 / T) * (fwd_rxn_rates[21] * (-4.34910000e-01 + (5.93745344e+04 / T) + 1.0 - 2.0)));
  jac[3] += -1.0 * ((-pres_mod[4] * fwd_rxn_rates[22] / T) + (pres_mod[4] / T) * (fwd_rxn_rates[22] * (-1.00000000e+00 + 1.0 - 2.0)));
  jac[3] += ((-pres_mod[5] * fwd_rxn_rates[23] / T) + (pres_mod[5] / T) * (fwd_rxn_rates[23] * (-1.03010000e+00 + (5.18313166e+04 / T) + 1.0 - 1.0)));
  jac[3] += -1.0 * ((1.0 / T) * (fwd_rxn_rates[34] * (1.00000000e+00 + (-3.64293641e+02 / T) + 1.0 - 2.0)));
  jac[3] += ((1.0 / T) * (fwd_rxn_rates[35] * (1.45930000e+00 + (2.62487877e+04 / T) + 1.0 - 2.0)));
  jac[3] += -1.0 * ((1.0 / T) * (fwd_rxn_rates[48] * (2.00000000e+00 + (1.99777016e+03 / T) + 1.0 - 2.0)));
  jac[3] += ((1.0 / T) * (fwd_rxn_rates[49] * (2.69380000e+00 + (9.31906943e+03 / T) + 1.0 - 2.0)));
  jac[3] *= 1.59994000e+01 / rho;

  //partial of omega_O wrt Y_H;
  jac[17] = ((-mw_avg * fwd_rxn_rates[0] / 1.00797000e+00 + exp(3.22754120e+01 - (7.69216995e+03 / T)) * (rho / 1.00797000e+00) * conc[5]) + (-mw_avg * fwd_rxn_rates[0] / 1.00797000e+00));
  jac[17] += -1.0 * ((-mw_avg * fwd_rxn_rates[1] / 1.00797000e+00) + (-mw_avg * fwd_rxn_rates[1] / 1.00797000e+00));
  jac[17] += -1.0 * ((-mw_avg * fwd_rxn_rates[2] / 1.00797000e+00) + (-mw_avg * fwd_rxn_rates[2] / 1.00797000e+00));
  jac[17] += ((-mw_avg * fwd_rxn_rates[3] / 1.00797000e+00 + exp(2.85866095e+01 - 0.053533 * logT - (3.32706731e+03 / T)) * (rho / 1.00797000e+00) * conc[3]) + (-mw_avg * fwd_rxn_rates[3] / 1.00797000e+00));
  jac[17] += -1.0 * ((-mw_avg * fwd_rxn_rates[4] / 1.00797000e+00) + (-mw_avg * fwd_rxn_rates[4] / 1.00797000e+00));
  jac[17] += ((-mw_avg * fwd_rxn_rates[5] / 1.00797000e+00 + exp(3.40258987e+01 - 0.053533 * logT - (8.97436602e+03 / T)) * (rho / 1.00797000e+00) * conc[3]) + (-mw_avg * fwd_rxn_rates[5] / 1.00797000e+00));
  jac[17] += (2.0 * (-mw_avg * fwd_rxn_rates[8] / 1.00797000e+00));
  jac[17] += -1.0 * ((-mw_avg * fwd_rxn_rates[9] / 1.00797000e+00) + (-mw_avg * fwd_rxn_rates[9] / 1.00797000e+00));
  jac[17] += -2.0 * ((-mw_avg * pres_mod[2] / 1.00797000e+00 + rho / 1.00797000e+00) * fwd_rxn_rates[16] + pres_mod[2] * (2.0 * (-mw_avg * fwd_rxn_rates[16] / 1.00797000e+00)));
  jac[17] += 2.0 * ((-mw_avg * pres_mod[3] / 1.00797000e+00 + rho / 1.00797000e+00) * fwd_rxn_rates[17] + pres_mod[3] * ((-mw_avg * fwd_rxn_rates[17] / 1.00797000e+00)));
  jac[17] += -2.0 * (2.0 * (-mw_avg * fwd_rxn_rates[18] / 1.00797000e+00) + (-mw_avg * fwd_rxn_rates[18] / 1.00797000e+00));
  jac[17] += 2.0 * ((-mw_avg * fwd_rxn_rates[19] / 1.00797000e+00) + (-mw_avg * fwd_rxn_rates[19] / 1.00797000e+00));
  jac[17] += -2.0 * (2.0 * (-mw_avg * fwd_rxn_rates[20] / 1.00797000e+00) + (-mw_avg * fwd_rxn_rates[20] / 1.00797000e+00));
  jac[17] += 2.0 * ((-mw_avg * fwd_rxn_rates[21] / 1.00797000e+00) + (-mw_avg * fwd_rxn_rates[21] / 1.00797000e+00));
  jac[17] += -1.0 * ((-mw_avg * pres_mod[4] / 1.00797000e+00 + rho / 1.00797000e+00) * fwd_rxn_rates[22] + pres_mod[4] * ((-mw_avg * fwd_rxn_rates[22] / 1.00797000e+00 + exp(4.29970685e+01 - 1.0 * logT) * (rho / 1.00797000e+00) * conc[2]) + (-mw_avg * fwd_rxn_rates[22] / 1.00797000e+00)));
  jac[17] += ((-mw_avg * pres_mod[5] / 1.00797000e+00 + rho / 1.00797000e+00) * fwd_rxn_rates[23] + pres_mod[5] * ((-mw_avg * fwd_rxn_rates[23] / 1.00797000e+00)));
  jac[17] += -1.0 * ((-mw_avg * fwd_rxn_rates[34] / 1.00797000e+00) + (-mw_avg * fwd_rxn_rates[34] / 1.00797000e+00));
  jac[17] += ((-mw_avg * fwd_rxn_rates[35] / 1.00797000e+00) + (-mw_avg * fwd_rxn_rates[35] / 1.00797000e+00));
  jac[17] += -1.0 * ((-mw_avg * fwd_rxn_rates[48] / 1.00797000e+00) + (-mw_avg * fwd_rxn_rates[48] / 1.00797000e+00));
  jac[17] += ((-mw_avg * fwd_rxn_rates[49] / 1.00797000e+00) + (-mw_avg * fwd_rxn_rates[49] / 1.00797000e+00));
  jac[17] += sp_rates[2] * mw_avg / 1.00797000e+00;
  jac[17] *= 1.59994000e+01 / rho;

  //partial of omega_O wrt Y_H2;
  jac[31] = ((-mw_avg * fwd_rxn_rates[0] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[0] / 2.01594000e+00));
  jac[31] += -1.0 * ((-mw_avg * fwd_rxn_rates[1] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[1] / 2.01594000e+00));
  jac[31] += -1.0 * ((-mw_avg * fwd_rxn_rates[2] / 2.01594000e+00 + exp(2.89707478e+01 - (3.99956606e+03 / T)) * (rho / 2.01594000e+00) * conc[2]) + (-mw_avg * fwd_rxn_rates[2] / 2.01594000e+00));
  jac[31] += ((-mw_avg * fwd_rxn_rates[3] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[3] / 2.01594000e+00));
  jac[31] += -1.0 * ((-mw_avg * fwd_rxn_rates[4] / 2.01594000e+00 + exp(3.44100335e+01 - (9.64666348e+03 / T)) * (rho / 2.01594000e+00) * conc[2]) + (-mw_avg * fwd_rxn_rates[4] / 2.01594000e+00));
  jac[31] += ((-mw_avg * fwd_rxn_rates[5] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[5] / 2.01594000e+00));
  jac[31] += (2.0 * (-mw_avg * fwd_rxn_rates[8] / 2.01594000e+00));
  jac[31] += -1.0 * ((-mw_avg * fwd_rxn_rates[9] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[9] / 2.01594000e+00));
  jac[31] += -2.0 * ((-mw_avg * pres_mod[2] / 2.01594000e+00 + 2.5 * rho / 2.01594000e+00) * fwd_rxn_rates[16] + pres_mod[2] * (2.0 * (-mw_avg * fwd_rxn_rates[16] / 2.01594000e+00)));
  jac[31] += 2.0 * ((-mw_avg * pres_mod[3] / 2.01594000e+00 + 2.5 * rho / 2.01594000e+00) * fwd_rxn_rates[17] + pres_mod[3] * ((-mw_avg * fwd_rxn_rates[17] / 2.01594000e+00)));
  jac[31] += -2.0 * (2.0 * (-mw_avg * fwd_rxn_rates[18] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[18] / 2.01594000e+00));
  jac[31] += 2.0 * ((-mw_avg * fwd_rxn_rates[19] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[19] / 2.01594000e+00));
  jac[31] += -2.0 * (2.0 * (-mw_avg * fwd_rxn_rates[20] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[20] / 2.01594000e+00));
  jac[31] += 2.0 * ((-mw_avg * fwd_rxn_rates[21] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[21] / 2.01594000e+00));
  jac[31] += -1.0 * ((-mw_avg * pres_mod[4] / 2.01594000e+00 + 2.5 * rho / 2.01594000e+00) * fwd_rxn_rates[22] + pres_mod[4] * ((-mw_avg * fwd_rxn_rates[22] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[22] / 2.01594000e+00)));
  jac[31] += ((-mw_avg * pres_mod[5] / 2.01594000e+00 + 2.5 * rho / 2.01594000e+00) * fwd_rxn_rates[23] + pres_mod[5] * ((-mw_avg * fwd_rxn_rates[23] / 2.01594000e+00)));
  jac[31] += -1.0 * ((-mw_avg * fwd_rxn_rates[34] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[34] / 2.01594000e+00));
  jac[31] += ((-mw_avg * fwd_rxn_rates[35] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[35] / 2.01594000e+00));
  jac[31] += -1.0 * ((-mw_avg * fwd_rxn_rates[48] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[48] / 2.01594000e+00));
  jac[31] += ((-mw_avg * fwd_rxn_rates[49] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[49] / 2.01594000e+00));
  jac[31] += sp_rates[2] * mw_avg / 2.01594000e+00;
  jac[31] *= 1.59994000e+01 / rho;

  //partial of omega_O wrt Y_O;
  jac[45] = ((-mw_avg * fwd_rxn_rates[0] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[0] / 1.59994000e+01));
  jac[45] += -1.0 * ((-mw_avg * fwd_rxn_rates[1] / 1.59994000e+01 + exp(2.63075513e+01 + 0.40478 * logT - (-7.48736077e+02 / T)) * (rho / 1.59994000e+01) * conc[3]) + (-mw_avg * fwd_rxn_rates[1] / 1.59994000e+01));
  jac[45] += -1.0 * ((-mw_avg * fwd_rxn_rates[2] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[2] / 1.59994000e+01 + exp(2.89707478e+01 - (3.99956606e+03 / T)) * (rho / 1.59994000e+01) * conc[1]));
  jac[45] += ((-mw_avg * fwd_rxn_rates[3] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[3] / 1.59994000e+01));
  jac[45] += -1.0 * ((-mw_avg * fwd_rxn_rates[4] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[4] / 1.59994000e+01 + exp(3.44100335e+01 - (9.64666348e+03 / T)) * (rho / 1.59994000e+01) * conc[1]));
  jac[45] += ((-mw_avg * fwd_rxn_rates[5] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[5] / 1.59994000e+01));
  jac[45] += (2.0 * (-mw_avg * fwd_rxn_rates[8] / 1.59994000e+01));
  jac[45] += -1.0 * ((-mw_avg * fwd_rxn_rates[9] / 1.59994000e+01 + exp(1.50329128e+01 + 2.1464 * logT - (7.53063740e+03 / T)) * (rho / 1.59994000e+01) * conc[4]) + (-mw_avg * fwd_rxn_rates[9] / 1.59994000e+01));
  jac[45] += -2.0 * ((-mw_avg * pres_mod[2] / 1.59994000e+01 + rho / 1.59994000e+01) * fwd_rxn_rates[16] + pres_mod[2] * (2.0 * (-mw_avg * fwd_rxn_rates[16] / 1.59994000e+01 + exp(3.63576645e+01 - 0.5 * logT) * (rho / 1.59994000e+01) * conc[2])));
  jac[45] += 2.0 * ((-mw_avg * pres_mod[3] / 1.59994000e+01 + rho / 1.59994000e+01) * fwd_rxn_rates[17] + pres_mod[3] * ((-mw_avg * fwd_rxn_rates[17] / 1.59994000e+01)));
  jac[45] += -2.0 * (2.0 * (-mw_avg * fwd_rxn_rates[18] / 1.59994000e+01 + exp(3.05680644e+01 - (-8.99751398e+02 / T)) * (rho / 1.59994000e+01) * conc[2] * conc[9]) + (-mw_avg * fwd_rxn_rates[18] / 1.59994000e+01));
  jac[45] += 2.0 * ((-mw_avg * fwd_rxn_rates[19] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[19] / 1.59994000e+01));
  jac[45] += -2.0 * (2.0 * (-mw_avg * fwd_rxn_rates[20] / 1.59994000e+01 + exp(3.05680644e+01 - (-8.99751398e+02 / T)) * (rho / 1.59994000e+01) * conc[2] * conc[10]) + (-mw_avg * fwd_rxn_rates[20] / 1.59994000e+01));
  jac[45] += 2.0 * ((-mw_avg * fwd_rxn_rates[21] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[21] / 1.59994000e+01));
  jac[45] += -1.0 * ((-mw_avg * pres_mod[4] / 1.59994000e+01 + rho / 1.59994000e+01) * fwd_rxn_rates[22] + pres_mod[4] * ((-mw_avg * fwd_rxn_rates[22] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[22] / 1.59994000e+01 + exp(4.29970685e+01 - 1.0 * logT) * (rho / 1.59994000e+01) * conc[0])));
  jac[45] += ((-mw_avg * pres_mod[5] / 1.59994000e+01 + rho / 1.59994000e+01) * fwd_rxn_rates[23] + pres_mod[5] * ((-mw_avg * fwd_rxn_rates[23] / 1.59994000e+01)));
  jac[45] += -1.0 * ((-mw_avg * fwd_rxn_rates[34] / 1.59994000e+01 + exp(2.40731699e+01 + 1.0 * logT - (-3.64293641e+02 / T)) * (rho / 1.59994000e+01) * conc[6]) + (-mw_avg * fwd_rxn_rates[34] / 1.59994000e+01));
  jac[45] += ((-mw_avg * fwd_rxn_rates[35] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[35] / 1.59994000e+01));
  jac[45] += -1.0 * ((-mw_avg * fwd_rxn_rates[48] / 1.59994000e+01 + exp(1.60720517e+01 + 2.0 * logT - (1.99777016e+03 / T)) * (rho / 1.59994000e+01) * conc[7]) + (-mw_avg * fwd_rxn_rates[48] / 1.59994000e+01));
  jac[45] += ((-mw_avg * fwd_rxn_rates[49] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[49] / 1.59994000e+01));
  jac[45] += sp_rates[2] * mw_avg / 1.59994000e+01;
  jac[45] *= 1.59994000e+01 / rho;

  //partial of omega_O wrt Y_OH;
  jac[59] = ((-mw_avg * fwd_rxn_rates[0] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[0] / 1.70073700e+01));
  jac[59] += -1.0 * ((-mw_avg * fwd_rxn_rates[1] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[1] / 1.70073700e+01 + exp(2.63075513e+01 + 0.40478 * logT - (-7.48736077e+02 / T)) * (rho / 1.70073700e+01) * conc[2]));
  jac[59] += -1.0 * ((-mw_avg * fwd_rxn_rates[2] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[2] / 1.70073700e+01));
  jac[59] += ((-mw_avg * fwd_rxn_rates[3] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[3] / 1.70073700e+01 + exp(2.85866095e+01 - 0.053533 * logT - (3.32706731e+03 / T)) * (rho / 1.70073700e+01) * conc[0]));
  jac[59] += -1.0 * ((-mw_avg * fwd_rxn_rates[4] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[4] / 1.70073700e+01));
  jac[59] += ((-mw_avg * fwd_rxn_rates[5] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[5] / 1.70073700e+01 + exp(3.40258987e+01 - 0.053533 * logT - (8.97436602e+03 / T)) * (rho / 1.70073700e+01) * conc[0]));
  jac[59] += (2.0 * (-mw_avg * fwd_rxn_rates[8] / 1.70073700e+01 + exp(1.04163112e+01 + 2.42 * logT - (-9.71208165e+02 / T)) * (rho / 1.70073700e+01) * conc[3]));
  jac[59] += -1.0 * ((-mw_avg * fwd_rxn_rates[9] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[9] / 1.70073700e+01));
  jac[59] += -2.0 * ((-mw_avg * pres_mod[2] / 1.70073700e+01 + rho / 1.70073700e+01) * fwd_rxn_rates[16] + pres_mod[2] * (2.0 * (-mw_avg * fwd_rxn_rates[16] / 1.70073700e+01)));
  jac[59] += 2.0 * ((-mw_avg * pres_mod[3] / 1.70073700e+01 + rho / 1.70073700e+01) * fwd_rxn_rates[17] + pres_mod[3] * ((-mw_avg * fwd_rxn_rates[17] / 1.70073700e+01)));
  jac[59] += -2.0 * (2.0 * (-mw_avg * fwd_rxn_rates[18] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[18] / 1.70073700e+01));
  jac[59] += 2.0 * ((-mw_avg * fwd_rxn_rates[19] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[19] / 1.70073700e+01));
  jac[59] += -2.0 * (2.0 * (-mw_avg * fwd_rxn_rates[20] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[20] / 1.70073700e+01));
  jac[59] += 2.0 * ((-mw_avg * fwd_rxn_rates[21] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[21] / 1.70073700e+01));
  jac[59] += -1.0 * ((-mw_avg * pres_mod[4] / 1.70073700e+01 + rho / 1.70073700e+01) * fwd_rxn_rates[22] + pres_mod[4] * ((-mw_avg * fwd_rxn_rates[22] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[22] / 1.70073700e+01)));
  jac[59] += ((-mw_avg * pres_mod[5] / 1.70073700e+01 + rho / 1.70073700e+01) * fwd_rxn_rates[23] + pres_mod[5] * ((-mw_avg * fwd_rxn_rates[23] / 1.70073700e+01 + exp(4.38224602e+01 - 1.0301 * logT - (5.18313166e+04 / T)) * (rho / 1.70073700e+01))));
  jac[59] += -1.0 * ((-mw_avg * fwd_rxn_rates[34] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[34] / 1.70073700e+01));
  jac[59] += ((-mw_avg * fwd_rxn_rates[35] / 1.70073700e+01 + exp(2.06516517e+01 + 1.4593 * logT - (2.62487877e+04 / T)) * (rho / 1.70073700e+01) * conc[5]) + (-mw_avg * fwd_rxn_rates[35] / 1.70073700e+01));
  jac[59] += -1.0 * ((-mw_avg * fwd_rxn_rates[48] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[48] / 1.70073700e+01));
  jac[59] += ((-mw_avg * fwd_rxn_rates[49] / 1.70073700e+01 + exp(8.90174786e+00 + 2.6938 * logT - (9.31906943e+03 / T)) * (rho / 1.70073700e+01) * conc[6]) + (-mw_avg * fwd_rxn_rates[49] / 1.70073700e+01));
  jac[59] += sp_rates[2] * mw_avg / 1.70073700e+01;
  jac[59] *= 1.59994000e+01 / rho;

  //partial of omega_O wrt Y_H2O;
  jac[73] = ((-mw_avg * fwd_rxn_rates[0] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[0] / 1.80153400e+01));
  jac[73] += -1.0 * ((-mw_avg * fwd_rxn_rates[1] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[1] / 1.80153400e+01));
  jac[73] += -1.0 * ((-mw_avg * fwd_rxn_rates[2] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[2] / 1.80153400e+01));
  jac[73] += ((-mw_avg * fwd_rxn_rates[3] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[3] / 1.80153400e+01));
  jac[73] += -1.0 * ((-mw_avg * fwd_rxn_rates[4] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[4] / 1.80153400e+01));
  jac[73] += ((-mw_avg * fwd_rxn_rates[5] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[5] / 1.80153400e+01));
  jac[73] += (2.0 * (-mw_avg * fwd_rxn_rates[8] / 1.80153400e+01));
  jac[73] += -1.0 * ((-mw_avg * fwd_rxn_rates[9] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[9] / 1.80153400e+01 + exp(1.50329128e+01 + 2.1464 * logT - (7.53063740e+03 / T)) * (rho / 1.80153400e+01) * conc[2]));
  jac[73] += -2.0 * ((-mw_avg * pres_mod[2] / 1.80153400e+01 + 12.0 * rho / 1.80153400e+01) * fwd_rxn_rates[16] + pres_mod[2] * (2.0 * (-mw_avg * fwd_rxn_rates[16] / 1.80153400e+01)));
  jac[73] += 2.0 * ((-mw_avg * pres_mod[3] / 1.80153400e+01 + 12.0 * rho / 1.80153400e+01) * fwd_rxn_rates[17] + pres_mod[3] * ((-mw_avg * fwd_rxn_rates[17] / 1.80153400e+01)));
  jac[73] += -2.0 * (2.0 * (-mw_avg * fwd_rxn_rates[18] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[18] / 1.80153400e+01));
  jac[73] += 2.0 * ((-mw_avg * fwd_rxn_rates[19] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[19] / 1.80153400e+01));
  jac[73] += -2.0 * (2.0 * (-mw_avg * fwd_rxn_rates[20] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[20] / 1.80153400e+01));
  jac[73] += 2.0 * ((-mw_avg * fwd_rxn_rates[21] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[21] / 1.80153400e+01));
  jac[73] += -1.0 * ((-mw_avg * pres_mod[4] / 1.80153400e+01 + 12.0 * rho / 1.80153400e+01) * fwd_rxn_rates[22] + pres_mod[4] * ((-mw_avg * fwd_rxn_rates[22] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[22] / 1.80153400e+01)));
  jac[73] += ((-mw_avg * pres_mod[5] / 1.80153400e+01 + 12.0 * rho / 1.80153400e+01) * fwd_rxn_rates[23] + pres_mod[5] * ((-mw_avg * fwd_rxn_rates[23] / 1.80153400e+01)));
  jac[73] += -1.0 * ((-mw_avg * fwd_rxn_rates[34] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[34] / 1.80153400e+01));
  jac[73] += ((-mw_avg * fwd_rxn_rates[35] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[35] / 1.80153400e+01));
  jac[73] += -1.0 * ((-mw_avg * fwd_rxn_rates[48] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[48] / 1.80153400e+01));
  jac[73] += ((-mw_avg * fwd_rxn_rates[49] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[49] / 1.80153400e+01));
  jac[73] += sp_rates[2] * mw_avg / 1.80153400e+01;
  jac[73] *= 1.59994000e+01 / rho;

  //partial of omega_O wrt Y_O2;
  jac[87] = ((-mw_avg * fwd_rxn_rates[0] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[0] / 3.19988000e+01 + exp(3.22754120e+01 - (7.69216995e+03 / T)) * (rho / 3.19988000e+01) * conc[0]));
  jac[87] += -1.0 * ((-mw_avg * fwd_rxn_rates[1] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[1] / 3.19988000e+01));
  jac[87] += -1.0 * ((-mw_avg * fwd_rxn_rates[2] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[2] / 3.19988000e+01));
  jac[87] += ((-mw_avg * fwd_rxn_rates[3] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[3] / 3.19988000e+01));
  jac[87] += -1.0 * ((-mw_avg * fwd_rxn_rates[4] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[4] / 3.19988000e+01));
  jac[87] += ((-mw_avg * fwd_rxn_rates[5] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[5] / 3.19988000e+01));
  jac[87] += (2.0 * (-mw_avg * fwd_rxn_rates[8] / 3.19988000e+01));
  jac[87] += -1.0 * ((-mw_avg * fwd_rxn_rates[9] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[9] / 3.19988000e+01));
  jac[87] += -2.0 * ((-mw_avg * pres_mod[2] / 3.19988000e+01 + rho / 3.19988000e+01) * fwd_rxn_rates[16] + pres_mod[2] * (2.0 * (-mw_avg * fwd_rxn_rates[16] / 3.19988000e+01)));
  jac[87] += 2.0 * ((-mw_avg * pres_mod[3] / 3.19988000e+01 + rho / 3.19988000e+01) * fwd_rxn_rates[17] + pres_mod[3] * ((-mw_avg * fwd_rxn_rates[17] / 3.19988000e+01 + exp(4.31509343e+01 - 0.93491 * logT - (6.02702601e+04 / T)) * (rho / 3.19988000e+01))));
  jac[87] += -2.0 * (2.0 * (-mw_avg * fwd_rxn_rates[18] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[18] / 3.19988000e+01));
  jac[87] += 2.0 * ((-mw_avg * fwd_rxn_rates[19] / 3.19988000e+01 + exp(3.73613450e+01 - 0.43491 * logT - (5.93745344e+04 / T)) * (rho / 3.19988000e+01) * conc[9]) + (-mw_avg * fwd_rxn_rates[19] / 3.19988000e+01));
  jac[87] += -2.0 * (2.0 * (-mw_avg * fwd_rxn_rates[20] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[20] / 3.19988000e+01));
  jac[87] += 2.0 * ((-mw_avg * fwd_rxn_rates[21] / 3.19988000e+01 + exp(3.73613450e+01 - 0.43491 * logT - (5.93745344e+04 / T)) * (rho / 3.19988000e+01) * conc[10]) + (-mw_avg * fwd_rxn_rates[21] / 3.19988000e+01));
  jac[87] += -1.0 * ((-mw_avg * pres_mod[4] / 3.19988000e+01 + rho / 3.19988000e+01) * fwd_rxn_rates[22] + pres_mod[4] * ((-mw_avg * fwd_rxn_rates[22] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[22] / 3.19988000e+01)));
  jac[87] += ((-mw_avg * pres_mod[5] / 3.19988000e+01 + rho / 3.19988000e+01) * fwd_rxn_rates[23] + pres_mod[5] * ((-mw_avg * fwd_rxn_rates[23] / 3.19988000e+01)));
  jac[87] += -1.0 * ((-mw_avg * fwd_rxn_rates[34] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[34] / 3.19988000e+01));
  jac[87] += ((-mw_avg * fwd_rxn_rates[35] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[35] / 3.19988000e+01 + exp(2.06516517e+01 + 1.4593 * logT - (2.62487877e+04 / T)) * (rho / 3.19988000e+01) * conc[3]));
  jac[87] += -1.0 * ((-mw_avg * fwd_rxn_rates[48] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[48] / 3.19988000e+01));
  jac[87] += ((-mw_avg * fwd_rxn_rates[49] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[49] / 3.19988000e+01));
  jac[87] += sp_rates[2] * mw_avg / 3.19988000e+01;
  jac[87] *= 1.59994000e+01 / rho;

  //partial of omega_O wrt Y_HO2;
  jac[101] = ((-mw_avg * fwd_rxn_rates[0] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[0] / 3.30067700e+01));
  jac[101] += -1.0 * ((-mw_avg * fwd_rxn_rates[1] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[1] / 3.30067700e+01));
  jac[101] += -1.0 * ((-mw_avg * fwd_rxn_rates[2] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[2] / 3.30067700e+01));
  jac[101] += ((-mw_avg * fwd_rxn_rates[3] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[3] / 3.30067700e+01));
  jac[101] += -1.0 * ((-mw_avg * fwd_rxn_rates[4] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[4] / 3.30067700e+01));
  jac[101] += ((-mw_avg * fwd_rxn_rates[5] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[5] / 3.30067700e+01));
  jac[101] += (2.0 * (-mw_avg * fwd_rxn_rates[8] / 3.30067700e+01));
  jac[101] += -1.0 * ((-mw_avg * fwd_rxn_rates[9] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[9] / 3.30067700e+01));
  jac[101] += -2.0 * ((-mw_avg * pres_mod[2] / 3.30067700e+01 + rho / 3.30067700e+01) * fwd_rxn_rates[16] + pres_mod[2] * (2.0 * (-mw_avg * fwd_rxn_rates[16] / 3.30067700e+01)));
  jac[101] += 2.0 * ((-mw_avg * pres_mod[3] / 3.30067700e+01 + rho / 3.30067700e+01) * fwd_rxn_rates[17] + pres_mod[3] * ((-mw_avg * fwd_rxn_rates[17] / 3.30067700e+01)));
  jac[101] += -2.0 * (2.0 * (-mw_avg * fwd_rxn_rates[18] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[18] / 3.30067700e+01));
  jac[101] += 2.0 * ((-mw_avg * fwd_rxn_rates[19] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[19] / 3.30067700e+01));
  jac[101] += -2.0 * (2.0 * (-mw_avg * fwd_rxn_rates[20] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[20] / 3.30067700e+01));
  jac[101] += 2.0 * ((-mw_avg * fwd_rxn_rates[21] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[21] / 3.30067700e+01));
  jac[101] += -1.0 * ((-mw_avg * pres_mod[4] / 3.30067700e+01 + rho / 3.30067700e+01) * fwd_rxn_rates[22] + pres_mod[4] * ((-mw_avg * fwd_rxn_rates[22] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[22] / 3.30067700e+01)));
  jac[101] += ((-mw_avg * pres_mod[5] / 3.30067700e+01 + rho / 3.30067700e+01) * fwd_rxn_rates[23] + pres_mod[5] * ((-mw_avg * fwd_rxn_rates[23] / 3.30067700e+01)));
  jac[101] += -1.0 * ((-mw_avg * fwd_rxn_rates[34] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[34] / 3.30067700e+01 + exp(2.40731699e+01 + 1.0 * logT - (-3.64293641e+02 / T)) * (rho / 3.30067700e+01) * conc[2]));
  jac[101] += ((-mw_avg * fwd_rxn_rates[35] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[35] / 3.30067700e+01));
  jac[101] += -1.0 * ((-mw_avg * fwd_rxn_rates[48] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[48] / 3.30067700e+01));
  jac[101] += ((-mw_avg * fwd_rxn_rates[49] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[49] / 3.30067700e+01 + exp(8.90174786e+00 + 2.6938 * logT - (9.31906943e+03 / T)) * (rho / 3.30067700e+01) * conc[3]));
  jac[101] += sp_rates[2] * mw_avg / 3.30067700e+01;
  jac[101] *= 1.59994000e+01 / rho;

  //partial of omega_O wrt Y_H2O2;
  jac[115] = ((-mw_avg * fwd_rxn_rates[0] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[0] / 3.40147400e+01));
  jac[115] += -1.0 * ((-mw_avg * fwd_rxn_rates[1] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[1] / 3.40147400e+01));
  jac[115] += -1.0 * ((-mw_avg * fwd_rxn_rates[2] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[2] / 3.40147400e+01));
  jac[115] += ((-mw_avg * fwd_rxn_rates[3] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[3] / 3.40147400e+01));
  jac[115] += -1.0 * ((-mw_avg * fwd_rxn_rates[4] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[4] / 3.40147400e+01));
  jac[115] += ((-mw_avg * fwd_rxn_rates[5] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[5] / 3.40147400e+01));
  jac[115] += (2.0 * (-mw_avg * fwd_rxn_rates[8] / 3.40147400e+01));
  jac[115] += -1.0 * ((-mw_avg * fwd_rxn_rates[9] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[9] / 3.40147400e+01));
  jac[115] += -2.0 * ((-mw_avg * pres_mod[2] / 3.40147400e+01 + rho / 3.40147400e+01) * fwd_rxn_rates[16] + pres_mod[2] * (2.0 * (-mw_avg * fwd_rxn_rates[16] / 3.40147400e+01)));
  jac[115] += 2.0 * ((-mw_avg * pres_mod[3] / 3.40147400e+01 + rho / 3.40147400e+01) * fwd_rxn_rates[17] + pres_mod[3] * ((-mw_avg * fwd_rxn_rates[17] / 3.40147400e+01)));
  jac[115] += -2.0 * (2.0 * (-mw_avg * fwd_rxn_rates[18] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[18] / 3.40147400e+01));
  jac[115] += 2.0 * ((-mw_avg * fwd_rxn_rates[19] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[19] / 3.40147400e+01));
  jac[115] += -2.0 * (2.0 * (-mw_avg * fwd_rxn_rates[20] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[20] / 3.40147400e+01));
  jac[115] += 2.0 * ((-mw_avg * fwd_rxn_rates[21] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[21] / 3.40147400e+01));
  jac[115] += -1.0 * ((-mw_avg * pres_mod[4] / 3.40147400e+01 + rho / 3.40147400e+01) * fwd_rxn_rates[22] + pres_mod[4] * ((-mw_avg * fwd_rxn_rates[22] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[22] / 3.40147400e+01)));
  jac[115] += ((-mw_avg * pres_mod[5] / 3.40147400e+01 + rho / 3.40147400e+01) * fwd_rxn_rates[23] + pres_mod[5] * ((-mw_avg * fwd_rxn_rates[23] / 3.40147400e+01)));
  jac[115] += -1.0 * ((-mw_avg * fwd_rxn_rates[34] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[34] / 3.40147400e+01));
  jac[115] += ((-mw_avg * fwd_rxn_rates[35] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[35] / 3.40147400e+01));
  jac[115] += -1.0 * ((-mw_avg * fwd_rxn_rates[48] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[48] / 3.40147400e+01 + exp(1.60720517e+01 + 2.0 * logT - (1.99777016e+03 / T)) * (rho / 3.40147400e+01) * conc[2]));
  jac[115] += ((-mw_avg * fwd_rxn_rates[49] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[49] / 3.40147400e+01));
  jac[115] += sp_rates[2] * mw_avg / 3.40147400e+01;
  jac[115] *= 1.59994000e+01 / rho;

  //partial of omega_O wrt Y_N2;
  jac[129] = ((-mw_avg * fwd_rxn_rates[0] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[0] / 2.80134000e+01));
  jac[129] += -1.0 * ((-mw_avg * fwd_rxn_rates[1] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[1] / 2.80134000e+01));
  jac[129] += -1.0 * ((-mw_avg * fwd_rxn_rates[2] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[2] / 2.80134000e+01));
  jac[129] += ((-mw_avg * fwd_rxn_rates[3] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[3] / 2.80134000e+01));
  jac[129] += -1.0 * ((-mw_avg * fwd_rxn_rates[4] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[4] / 2.80134000e+01));
  jac[129] += ((-mw_avg * fwd_rxn_rates[5] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[5] / 2.80134000e+01));
  jac[129] += (2.0 * (-mw_avg * fwd_rxn_rates[8] / 2.80134000e+01));
  jac[129] += -1.0 * ((-mw_avg * fwd_rxn_rates[9] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[9] / 2.80134000e+01));
  jac[129] += -2.0 * ((-mw_avg * pres_mod[2] / 2.80134000e+01 + rho / 2.80134000e+01) * fwd_rxn_rates[16] + pres_mod[2] * (2.0 * (-mw_avg * fwd_rxn_rates[16] / 2.80134000e+01)));
  jac[129] += 2.0 * ((-mw_avg * pres_mod[3] / 2.80134000e+01 + rho / 2.80134000e+01) * fwd_rxn_rates[17] + pres_mod[3] * ((-mw_avg * fwd_rxn_rates[17] / 2.80134000e+01)));
  jac[129] += -2.0 * (2.0 * (-mw_avg * fwd_rxn_rates[18] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[18] / 2.80134000e+01));
  jac[129] += 2.0 * ((-mw_avg * fwd_rxn_rates[19] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[19] / 2.80134000e+01));
  jac[129] += -2.0 * (2.0 * (-mw_avg * fwd_rxn_rates[20] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[20] / 2.80134000e+01));
  jac[129] += 2.0 * ((-mw_avg * fwd_rxn_rates[21] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[21] / 2.80134000e+01));
  jac[129] += -1.0 * ((-mw_avg * pres_mod[4] / 2.80134000e+01 + rho / 2.80134000e+01) * fwd_rxn_rates[22] + pres_mod[4] * ((-mw_avg * fwd_rxn_rates[22] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[22] / 2.80134000e+01)));
  jac[129] += ((-mw_avg * pres_mod[5] / 2.80134000e+01 + rho / 2.80134000e+01) * fwd_rxn_rates[23] + pres_mod[5] * ((-mw_avg * fwd_rxn_rates[23] / 2.80134000e+01)));
  jac[129] += -1.0 * ((-mw_avg * fwd_rxn_rates[34] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[34] / 2.80134000e+01));
  jac[129] += ((-mw_avg * fwd_rxn_rates[35] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[35] / 2.80134000e+01));
  jac[129] += -1.0 * ((-mw_avg * fwd_rxn_rates[48] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[48] / 2.80134000e+01));
  jac[129] += ((-mw_avg * fwd_rxn_rates[49] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[49] / 2.80134000e+01));
  jac[129] += sp_rates[2] * mw_avg / 2.80134000e+01;
  jac[129] *= 1.59994000e+01 / rho;

  //partial of omega_O wrt Y_AR;
  jac[143] = ((-mw_avg * fwd_rxn_rates[0] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[0] / 3.99480000e+01));
  jac[143] += -1.0 * ((-mw_avg * fwd_rxn_rates[1] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[1] / 3.99480000e+01));
  jac[143] += -1.0 * ((-mw_avg * fwd_rxn_rates[2] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[2] / 3.99480000e+01));
  jac[143] += ((-mw_avg * fwd_rxn_rates[3] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[3] / 3.99480000e+01));
  jac[143] += -1.0 * ((-mw_avg * fwd_rxn_rates[4] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[4] / 3.99480000e+01));
  jac[143] += ((-mw_avg * fwd_rxn_rates[5] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[5] / 3.99480000e+01));
  jac[143] += (2.0 * (-mw_avg * fwd_rxn_rates[8] / 3.99480000e+01));
  jac[143] += -1.0 * ((-mw_avg * fwd_rxn_rates[9] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[9] / 3.99480000e+01));
  jac[143] += -2.0 * ((-mw_avg * pres_mod[2] / 3.99480000e+01) * fwd_rxn_rates[16] + pres_mod[2] * (2.0 * (-mw_avg * fwd_rxn_rates[16] / 3.99480000e+01)));
  jac[143] += 2.0 * ((-mw_avg * pres_mod[3] / 3.99480000e+01) * fwd_rxn_rates[17] + pres_mod[3] * ((-mw_avg * fwd_rxn_rates[17] / 3.99480000e+01)));
  jac[143] += -2.0 * (2.0 * (-mw_avg * fwd_rxn_rates[18] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[18] / 3.99480000e+01 + exp(3.05680644e+01 - (-8.99751398e+02 / T)) * (rho / 3.99480000e+01) * conc[2] * conc[2]));
  jac[143] += 2.0 * ((-mw_avg * fwd_rxn_rates[19] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[19] / 3.99480000e+01 + exp(3.73613450e+01 - 0.43491 * logT - (5.93745344e+04 / T)) * (rho / 3.99480000e+01) * conc[5]));
  jac[143] += -2.0 * (2.0 * (-mw_avg * fwd_rxn_rates[20] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[20] / 3.99480000e+01));
  jac[143] += 2.0 * ((-mw_avg * fwd_rxn_rates[21] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[21] / 3.99480000e+01));
  jac[143] += -1.0 * ((-mw_avg * pres_mod[4] / 3.99480000e+01 + 0.75 * rho / 3.99480000e+01) * fwd_rxn_rates[22] + pres_mod[4] * ((-mw_avg * fwd_rxn_rates[22] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[22] / 3.99480000e+01)));
  jac[143] += ((-mw_avg * pres_mod[5] / 3.99480000e+01 + 0.75 * rho / 3.99480000e+01) * fwd_rxn_rates[23] + pres_mod[5] * ((-mw_avg * fwd_rxn_rates[23] / 3.99480000e+01)));
  jac[143] += -1.0 * ((-mw_avg * fwd_rxn_rates[34] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[34] / 3.99480000e+01));
  jac[143] += ((-mw_avg * fwd_rxn_rates[35] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[35] / 3.99480000e+01));
  jac[143] += -1.0 * ((-mw_avg * fwd_rxn_rates[48] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[48] / 3.99480000e+01));
  jac[143] += ((-mw_avg * fwd_rxn_rates[49] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[49] / 3.99480000e+01));
  jac[143] += sp_rates[2] * mw_avg / 3.99480000e+01;
  jac[143] *= 1.59994000e+01 / rho;

  //partial of omega_O wrt Y_HE;
  jac[157] = ((-mw_avg * fwd_rxn_rates[0] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[0] / 4.00260000e+00));
  jac[157] += -1.0 * ((-mw_avg * fwd_rxn_rates[1] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[1] / 4.00260000e+00));
  jac[157] += -1.0 * ((-mw_avg * fwd_rxn_rates[2] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[2] / 4.00260000e+00));
  jac[157] += ((-mw_avg * fwd_rxn_rates[3] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[3] / 4.00260000e+00));
  jac[157] += -1.0 * ((-mw_avg * fwd_rxn_rates[4] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[4] / 4.00260000e+00));
  jac[157] += ((-mw_avg * fwd_rxn_rates[5] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[5] / 4.00260000e+00));
  jac[157] += (2.0 * (-mw_avg * fwd_rxn_rates[8] / 4.00260000e+00));
  jac[157] += -1.0 * ((-mw_avg * fwd_rxn_rates[9] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[9] / 4.00260000e+00));
  jac[157] += -2.0 * ((-mw_avg * pres_mod[2] / 4.00260000e+00) * fwd_rxn_rates[16] + pres_mod[2] * (2.0 * (-mw_avg * fwd_rxn_rates[16] / 4.00260000e+00)));
  jac[157] += 2.0 * ((-mw_avg * pres_mod[3] / 4.00260000e+00) * fwd_rxn_rates[17] + pres_mod[3] * ((-mw_avg * fwd_rxn_rates[17] / 4.00260000e+00)));
  jac[157] += -2.0 * (2.0 * (-mw_avg * fwd_rxn_rates[18] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[18] / 4.00260000e+00));
  jac[157] += 2.0 * ((-mw_avg * fwd_rxn_rates[19] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[19] / 4.00260000e+00));
  jac[157] += -2.0 * (2.0 * (-mw_avg * fwd_rxn_rates[20] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[20] / 4.00260000e+00 + exp(3.05680644e+01 - (-8.99751398e+02 / T)) * (rho / 4.00260000e+00) * conc[2] * conc[2]));
  jac[157] += 2.0 * ((-mw_avg * fwd_rxn_rates[21] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[21] / 4.00260000e+00 + exp(3.73613450e+01 - 0.43491 * logT - (5.93745344e+04 / T)) * (rho / 4.00260000e+00) * conc[5]));
  jac[157] += -1.0 * ((-mw_avg * pres_mod[4] / 4.00260000e+00 + 0.75 * rho / 4.00260000e+00) * fwd_rxn_rates[22] + pres_mod[4] * ((-mw_avg * fwd_rxn_rates[22] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[22] / 4.00260000e+00)));
  jac[157] += ((-mw_avg * pres_mod[5] / 4.00260000e+00 + 0.75 * rho / 4.00260000e+00) * fwd_rxn_rates[23] + pres_mod[5] * ((-mw_avg * fwd_rxn_rates[23] / 4.00260000e+00)));
  jac[157] += -1.0 * ((-mw_avg * fwd_rxn_rates[34] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[34] / 4.00260000e+00));
  jac[157] += ((-mw_avg * fwd_rxn_rates[35] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[35] / 4.00260000e+00));
  jac[157] += -1.0 * ((-mw_avg * fwd_rxn_rates[48] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[48] / 4.00260000e+00));
  jac[157] += ((-mw_avg * fwd_rxn_rates[49] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[49] / 4.00260000e+00));
  jac[157] += sp_rates[2] * mw_avg / 4.00260000e+00;
  jac[157] *= 1.59994000e+01 / rho;

  //partial of omega_O wrt Y_CO;
  jac[171] = ((-mw_avg * fwd_rxn_rates[0] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[0] / 2.80105500e+01));
  jac[171] += -1.0 * ((-mw_avg * fwd_rxn_rates[1] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[1] / 2.80105500e+01));
  jac[171] += -1.0 * ((-mw_avg * fwd_rxn_rates[2] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[2] / 2.80105500e+01));
  jac[171] += ((-mw_avg * fwd_rxn_rates[3] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[3] / 2.80105500e+01));
  jac[171] += -1.0 * ((-mw_avg * fwd_rxn_rates[4] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[4] / 2.80105500e+01));
  jac[171] += ((-mw_avg * fwd_rxn_rates[5] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[5] / 2.80105500e+01));
  jac[171] += (2.0 * (-mw_avg * fwd_rxn_rates[8] / 2.80105500e+01));
  jac[171] += -1.0 * ((-mw_avg * fwd_rxn_rates[9] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[9] / 2.80105500e+01));
  jac[171] += -2.0 * ((-mw_avg * pres_mod[2] / 2.80105500e+01 + 1.9 * rho / 2.80105500e+01) * fwd_rxn_rates[16] + pres_mod[2] * (2.0 * (-mw_avg * fwd_rxn_rates[16] / 2.80105500e+01)));
  jac[171] += 2.0 * ((-mw_avg * pres_mod[3] / 2.80105500e+01 + 1.9 * rho / 2.80105500e+01) * fwd_rxn_rates[17] + pres_mod[3] * ((-mw_avg * fwd_rxn_rates[17] / 2.80105500e+01)));
  jac[171] += -2.0 * (2.0 * (-mw_avg * fwd_rxn_rates[18] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[18] / 2.80105500e+01));
  jac[171] += 2.0 * ((-mw_avg * fwd_rxn_rates[19] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[19] / 2.80105500e+01));
  jac[171] += -2.0 * (2.0 * (-mw_avg * fwd_rxn_rates[20] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[20] / 2.80105500e+01));
  jac[171] += 2.0 * ((-mw_avg * fwd_rxn_rates[21] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[21] / 2.80105500e+01));
  jac[171] += -1.0 * ((-mw_avg * pres_mod[4] / 2.80105500e+01 + 1.9 * rho / 2.80105500e+01) * fwd_rxn_rates[22] + pres_mod[4] * ((-mw_avg * fwd_rxn_rates[22] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[22] / 2.80105500e+01)));
  jac[171] += ((-mw_avg * pres_mod[5] / 2.80105500e+01 + 1.9 * rho / 2.80105500e+01) * fwd_rxn_rates[23] + pres_mod[5] * ((-mw_avg * fwd_rxn_rates[23] / 2.80105500e+01)));
  jac[171] += -1.0 * ((-mw_avg * fwd_rxn_rates[34] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[34] / 2.80105500e+01));
  jac[171] += ((-mw_avg * fwd_rxn_rates[35] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[35] / 2.80105500e+01));
  jac[171] += -1.0 * ((-mw_avg * fwd_rxn_rates[48] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[48] / 2.80105500e+01));
  jac[171] += ((-mw_avg * fwd_rxn_rates[49] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[49] / 2.80105500e+01));
  jac[171] += sp_rates[2] * mw_avg / 2.80105500e+01;
  jac[171] *= 1.59994000e+01 / rho;

  //partial of omega_O wrt Y_CO2;
  jac[185] = ((-mw_avg * fwd_rxn_rates[0] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[0] / 4.40099500e+01));
  jac[185] += -1.0 * ((-mw_avg * fwd_rxn_rates[1] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[1] / 4.40099500e+01));
  jac[185] += -1.0 * ((-mw_avg * fwd_rxn_rates[2] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[2] / 4.40099500e+01));
  jac[185] += ((-mw_avg * fwd_rxn_rates[3] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[3] / 4.40099500e+01));
  jac[185] += -1.0 * ((-mw_avg * fwd_rxn_rates[4] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[4] / 4.40099500e+01));
  jac[185] += ((-mw_avg * fwd_rxn_rates[5] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[5] / 4.40099500e+01));
  jac[185] += (2.0 * (-mw_avg * fwd_rxn_rates[8] / 4.40099500e+01));
  jac[185] += -1.0 * ((-mw_avg * fwd_rxn_rates[9] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[9] / 4.40099500e+01));
  jac[185] += -2.0 * ((-mw_avg * pres_mod[2] / 4.40099500e+01 + 3.8 * rho / 4.40099500e+01) * fwd_rxn_rates[16] + pres_mod[2] * (2.0 * (-mw_avg * fwd_rxn_rates[16] / 4.40099500e+01)));
  jac[185] += 2.0 * ((-mw_avg * pres_mod[3] / 4.40099500e+01 + 3.8 * rho / 4.40099500e+01) * fwd_rxn_rates[17] + pres_mod[3] * ((-mw_avg * fwd_rxn_rates[17] / 4.40099500e+01)));
  jac[185] += -2.0 * (2.0 * (-mw_avg * fwd_rxn_rates[18] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[18] / 4.40099500e+01));
  jac[185] += 2.0 * ((-mw_avg * fwd_rxn_rates[19] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[19] / 4.40099500e+01));
  jac[185] += -2.0 * (2.0 * (-mw_avg * fwd_rxn_rates[20] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[20] / 4.40099500e+01));
  jac[185] += 2.0 * ((-mw_avg * fwd_rxn_rates[21] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[21] / 4.40099500e+01));
  jac[185] += -1.0 * ((-mw_avg * pres_mod[4] / 4.40099500e+01 + 3.8 * rho / 4.40099500e+01) * fwd_rxn_rates[22] + pres_mod[4] * ((-mw_avg * fwd_rxn_rates[22] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[22] / 4.40099500e+01)));
  jac[185] += ((-mw_avg * pres_mod[5] / 4.40099500e+01 + 3.8 * rho / 4.40099500e+01) * fwd_rxn_rates[23] + pres_mod[5] * ((-mw_avg * fwd_rxn_rates[23] / 4.40099500e+01)));
  jac[185] += -1.0 * ((-mw_avg * fwd_rxn_rates[34] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[34] / 4.40099500e+01));
  jac[185] += ((-mw_avg * fwd_rxn_rates[35] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[35] / 4.40099500e+01));
  jac[185] += -1.0 * ((-mw_avg * fwd_rxn_rates[48] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[48] / 4.40099500e+01));
  jac[185] += ((-mw_avg * fwd_rxn_rates[49] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[49] / 4.40099500e+01));
  jac[185] += sp_rates[2] * mw_avg / 4.40099500e+01;
  jac[185] *= 1.59994000e+01 / rho;

  //partial of omega_OH wrt T;
  jac[4] = ((1.0 / T) * (fwd_rxn_rates[0] * ((7.69216995e+03 / T) + 1.0 - 2.0)));
  jac[4] += -1.0 * ((1.0 / T) * (fwd_rxn_rates[1] * (4.04780000e-01 + (-7.48736077e+02 / T) + 1.0 - 2.0)));
  jac[4] += ((1.0 / T) * (fwd_rxn_rates[2] * ((3.99956606e+03 / T) + 1.0 - 2.0)));
  jac[4] += -1.0 * ((1.0 / T) * (fwd_rxn_rates[3] * (-5.35330000e-02 + (3.32706731e+03 / T) + 1.0 - 2.0)));
  jac[4] += ((1.0 / T) * (fwd_rxn_rates[4] * ((9.64666348e+03 / T) + 1.0 - 2.0)));
  jac[4] += -1.0 * ((1.0 / T) * (fwd_rxn_rates[5] * (-5.35330000e-02 + (8.97436602e+03 / T) + 1.0 - 2.0)));
  jac[4] += -1.0 * ((1.0 / T) * (fwd_rxn_rates[6] * (1.51000000e+00 + (1.72603316e+03 / T) + 1.0 - 2.0)));
  jac[4] += ((1.0 / T) * (fwd_rxn_rates[7] * (1.18290000e+00 + (9.55507805e+03 / T) + 1.0 - 2.0)));
  jac[4] += -2.0 * ((1.0 / T) * (fwd_rxn_rates[8] * (2.42000000e+00 + (-9.71208165e+02 / T) + 1.0 - 2.0)));
  jac[4] += 2.0 * ((1.0 / T) * (fwd_rxn_rates[9] * (2.14640000e+00 + (7.53063740e+03 / T) + 1.0 - 2.0)));
  jac[4] += ((-pres_mod[4] * fwd_rxn_rates[22] / T) + (pres_mod[4] / T) * (fwd_rxn_rates[22] * (-1.00000000e+00 + 1.0 - 2.0)));
  jac[4] += -1.0 * ((-pres_mod[5] * fwd_rxn_rates[23] / T) + (pres_mod[5] / T) * (fwd_rxn_rates[23] * (-1.03010000e+00 + (5.18313166e+04 / T) + 1.0 - 1.0)));
  jac[4] += ((-pres_mod[6] * fwd_rxn_rates[24] / T) + (pres_mod[6] / T) * (fwd_rxn_rates[24] * (-3.32200000e+00 + (6.07835411e+04 / T) + 1.0 - 1.0)));
  jac[4] += -1.0 * ((-pres_mod[7] * fwd_rxn_rates[25] / T) + (pres_mod[7] / T) * (fwd_rxn_rates[25] * (-3.01830000e+00 + (4.50771425e+02 / T) + 1.0 - 2.0)));
  jac[4] += ((1.0 / T) * (fwd_rxn_rates[26] * (-2.44000000e+00 + (6.04765789e+04 / T) + 1.0 - 2.0)));
  jac[4] += -1.0 * ((1.0 / T) * (fwd_rxn_rates[27] * (-2.13630000e+00 + (1.43809259e+02 / T) + 1.0 - 3.0)));
  jac[4] += 2.0 * ((1.0 / T) * (fwd_rxn_rates[32] * ((1.48448916e+02 / T) + 1.0 - 2.0)));
  jac[4] += -2.0 * ((1.0 / T) * (fwd_rxn_rates[33] * (8.64090000e-01 + (1.83206092e+04 / T) + 1.0 - 2.0)));
  jac[4] += ((1.0 / T) * (fwd_rxn_rates[34] * (1.00000000e+00 + (-3.64293641e+02 / T) + 1.0 - 2.0)));
  jac[4] += -1.0 * ((1.0 / T) * (fwd_rxn_rates[35] * (1.45930000e+00 + (2.62487877e+04 / T) + 1.0 - 2.0)));
  jac[4] += -1.0 * ((1.0 / T) * (fwd_rxn_rates[36] * ((-2.50098683e+02 / T) + 1.0 - 2.0)));
  jac[4] += ((1.0 / T) * (fwd_rxn_rates[37] * (1.85740000e-01 + (3.48643603e+04 / T) + 1.0 - 2.0)));
  Pr = (m + 6.5 * conc[4] + 0.6 * conc[12] + 0.5 * conc[8] + 0.2 * conc[5] - 0.35 * conc[10] + 6.7 * conc[7] + 2.7 * conc[1] + 1.8 * conc[11]) * (exp(2.78501566e+01 - 3.2 * logT));
  Fcent = 5.70000000e-01 * exp(-T / 1.00000000e-30) + 4.30000000e-01 * exp(T / 1.00000000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  lnF_AB = 2.0 * log(Fcent) * A / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)));
  jac[4] += 2.0 * (pres_mod[10]* (((-3.2000e+00 + (0.00000000e+00 / T) - 1.0) / (T * (1.0 + Pr))) + (((1.0 / (Fcent * (1.0 + A * A / (B * B)))) - lnF_AB * (-2.90977303e-01 * B + 5.10817170e-01 * A) / Fcent) * (-5.70000000e+29 * exp(-T / 1.00000000e-30) - 4.30000000e-31 * exp(-T / 1.00000000e+30))) - lnF_AB * (4.34294482e-01 * B + 6.08012275e-02 * A) * (-3.20000000e+00 + (0.00000000e+00 / T) - 1.0) / T) * fwd_rxn_rates[42] + (pres_mod[10] / T) * (fwd_rxn_rates[42] * (9.00000000e-01 + (2.45313092e+04 / T) + 1.0 - 1.0)));
  Pr = (m + 6.5 * conc[4] + 0.6 * conc[12] + 0.5 * conc[8] + 0.2 * conc[5] - 0.35 * conc[10] + 6.7 * conc[7] + 2.7 * conc[1] + 1.8 * conc[11]) * (exp(2.78501521e+01 - 3.20001 * logT));
  Fcent = 5.70000000e-01 * exp(-T / 1.00000000e-30) + 4.30000000e-01 * exp(T / 1.00000000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  lnF_AB = 2.0 * log(Fcent) * A / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)));
  jac[4] += -2.0 * (pres_mod[11]* (((-3.2000e+00 + (0.00000000e+00 / T) - 1.0) / (T * (1.0 + Pr))) + (((1.0 / (Fcent * (1.0 + A * A / (B * B)))) - lnF_AB * (-2.90977303e-01 * B + 5.10817170e-01 * A) / Fcent) * (-5.70000000e+29 * exp(-T / 1.00000000e-30) - 4.30000000e-31 * exp(-T / 1.00000000e+30))) - lnF_AB * (4.34294482e-01 * B + 6.08012275e-02 * A) * (-3.20001000e+00 + (0.00000000e+00 / T) - 1.0) / T) * fwd_rxn_rates[43] + (pres_mod[11] / T) * (fwd_rxn_rates[43] * (2.48800000e+00 + (-1.80654783e+03 / T) + 1.0 - 2.0)));
  jac[4] += ((1.0 / T) * (fwd_rxn_rates[44] * ((1.99777016e+03 / T) + 1.0 - 2.0)));
  jac[4] += -1.0 * ((1.0 / T) * (fwd_rxn_rates[45] * (1.28430000e+00 + (3.59925720e+04 / T) + 1.0 - 2.0)));
  jac[4] += ((1.0 / T) * (fwd_rxn_rates[48] * (2.00000000e+00 + (1.99777016e+03 / T) + 1.0 - 2.0)));
  jac[4] += -1.0 * ((1.0 / T) * (fwd_rxn_rates[49] * (2.69380000e+00 + (9.31906943e+03 / T) + 1.0 - 2.0)));
  jac[4] += -1.0 * ((1.0 / T) * (fwd_rxn_rates[50] * ((1.60022900e+02 / T) + 1.0 - 2.0)));
  jac[4] += ((1.0 / T) * (fwd_rxn_rates[51] * (4.20210000e-01 + (1.59826645e+04 / T) + 1.0 - 2.0)));
  jac[4] += -1.0 * ((1.0 / T) * (fwd_rxn_rates[52] * ((3.65838516e+03 / T) + 1.0 - 2.0)));
  jac[4] += ((1.0 / T) * (fwd_rxn_rates[53] * (4.20210000e-01 + (1.94810268e+04 / T) + 1.0 - 2.0)));
  jac[4] *= 1.70073700e+01 / rho;

  //partial of omega_OH wrt Y_H;
  jac[18] = ((-mw_avg * fwd_rxn_rates[0] / 1.00797000e+00 + exp(3.22754120e+01 - (7.69216995e+03 / T)) * (rho / 1.00797000e+00) * conc[5]) + (-mw_avg * fwd_rxn_rates[0] / 1.00797000e+00));
  jac[18] += -1.0 * ((-mw_avg * fwd_rxn_rates[1] / 1.00797000e+00) + (-mw_avg * fwd_rxn_rates[1] / 1.00797000e+00));
  jac[18] += ((-mw_avg * fwd_rxn_rates[2] / 1.00797000e+00) + (-mw_avg * fwd_rxn_rates[2] / 1.00797000e+00));
  jac[18] += -1.0 * ((-mw_avg * fwd_rxn_rates[3] / 1.00797000e+00 + exp(2.85866095e+01 - 0.053533 * logT - (3.32706731e+03 / T)) * (rho / 1.00797000e+00) * conc[3]) + (-mw_avg * fwd_rxn_rates[3] / 1.00797000e+00));
  jac[18] += ((-mw_avg * fwd_rxn_rates[4] / 1.00797000e+00) + (-mw_avg * fwd_rxn_rates[4] / 1.00797000e+00));
  jac[18] += -1.0 * ((-mw_avg * fwd_rxn_rates[5] / 1.00797000e+00 + exp(3.40258987e+01 - 0.053533 * logT - (8.97436602e+03 / T)) * (rho / 1.00797000e+00) * conc[3]) + (-mw_avg * fwd_rxn_rates[5] / 1.00797000e+00));
  jac[18] += -1.0 * ((-mw_avg * fwd_rxn_rates[6] / 1.00797000e+00) + (-mw_avg * fwd_rxn_rates[6] / 1.00797000e+00));
  jac[18] += ((-mw_avg * fwd_rxn_rates[7] / 1.00797000e+00 + exp(2.34232839e+01 + 1.1829 * logT - (9.55507805e+03 / T)) * (rho / 1.00797000e+00) * conc[4]) + (-mw_avg * fwd_rxn_rates[7] / 1.00797000e+00));
  jac[18] += -2.0 * (2.0 * (-mw_avg * fwd_rxn_rates[8] / 1.00797000e+00));
  jac[18] += 2.0 * ((-mw_avg * fwd_rxn_rates[9] / 1.00797000e+00) + (-mw_avg * fwd_rxn_rates[9] / 1.00797000e+00));
  jac[18] += ((-mw_avg * pres_mod[4] / 1.00797000e+00 + rho / 1.00797000e+00) * fwd_rxn_rates[22] + pres_mod[4] * ((-mw_avg * fwd_rxn_rates[22] / 1.00797000e+00 + exp(4.29970685e+01 - 1.0 * logT) * (rho / 1.00797000e+00) * conc[2]) + (-mw_avg * fwd_rxn_rates[22] / 1.00797000e+00)));
  jac[18] += -1.0 * ((-mw_avg * pres_mod[5] / 1.00797000e+00 + rho / 1.00797000e+00) * fwd_rxn_rates[23] + pres_mod[5] * ((-mw_avg * fwd_rxn_rates[23] / 1.00797000e+00)));
  jac[18] += ((-mw_avg * pres_mod[6] / 1.00797000e+00 + rho / 1.00797000e+00) * fwd_rxn_rates[24] + pres_mod[6] * ((-mw_avg * fwd_rxn_rates[24] / 1.00797000e+00)));
  jac[18] += -1.0 * ((-mw_avg * pres_mod[7] / 1.00797000e+00 + rho / 1.00797000e+00) * fwd_rxn_rates[25] + pres_mod[7] * ((-mw_avg * fwd_rxn_rates[25] / 1.00797000e+00 + exp(5.85301272e+01 - 3.0183 * logT - (4.50771425e+02 / T)) * (rho / 1.00797000e+00) * conc[3]) + (-mw_avg * fwd_rxn_rates[25] / 1.00797000e+00)));
  jac[18] += (2.0 * (-mw_avg * fwd_rxn_rates[26] / 1.00797000e+00));
  jac[18] += -1.0 * ((-mw_avg * fwd_rxn_rates[27] / 1.00797000e+00 + exp(5.44311720e+01 - 2.1363 * logT - (1.43809259e+02 / T)) * (rho / 1.00797000e+00) * conc[3] * conc[4]) + (-mw_avg * fwd_rxn_rates[27] / 1.00797000e+00) + (-mw_avg * fwd_rxn_rates[27] / 1.00797000e+00));
  jac[18] += 2.0 * ((-mw_avg * fwd_rxn_rates[32] / 1.00797000e+00 + exp(3.18907389e+01 - (1.48448916e+02 / T)) * (rho / 1.00797000e+00) * conc[6]) + (-mw_avg * fwd_rxn_rates[32] / 1.00797000e+00));
  jac[18] += -2.0 * (2.0 * (-mw_avg * fwd_rxn_rates[33] / 1.00797000e+00));
  jac[18] += ((-mw_avg * fwd_rxn_rates[34] / 1.00797000e+00) + (-mw_avg * fwd_rxn_rates[34] / 1.00797000e+00));
  jac[18] += -1.0 * ((-mw_avg * fwd_rxn_rates[35] / 1.00797000e+00) + (-mw_avg * fwd_rxn_rates[35] / 1.00797000e+00));
  jac[18] += -1.0 * ((-mw_avg * fwd_rxn_rates[36] / 1.00797000e+00) + (-mw_avg * fwd_rxn_rates[36] / 1.00797000e+00));
  jac[18] += ((-mw_avg * fwd_rxn_rates[37] / 1.00797000e+00) + (-mw_avg * fwd_rxn_rates[37] / 1.00797000e+00));
  Pr = (m + 6.5 * conc[4] + 0.6 * conc[12] + 0.5 * conc[8] + 0.2 * conc[5] - 0.35 * conc[10] + 6.7 * conc[7] + 2.7 * conc[1] + 1.8 * conc[11]) * (exp(2.78501566e+01 - 3.2 * logT));
  Fcent = 5.7000e-01 * exp(-T / 1.0000e-30) + 4.3000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[18] += 2.0 * (pres_mod[10] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 1.00797000e+00 + rho / ((m + 6.5 * conc[4] + 0.6 * conc[12] + 0.5 * conc[8] + 0.2 * conc[5] - 0.35 * conc[10] + 6.7 * conc[7] + 2.7 * conc[1] + 1.8 * conc[11]) * 1.00797000e+00)) * fwd_rxn_rates[42] + pres_mod[10] * ((-mw_avg * fwd_rxn_rates[42] / 1.00797000e+00)));
  Pr = (m + 6.5 * conc[4] + 0.6 * conc[12] + 0.5 * conc[8] + 0.2 * conc[5] - 0.35 * conc[10] + 6.7 * conc[7] + 2.7 * conc[1] + 1.8 * conc[11]) * (exp(2.78501521e+01 - 3.20001 * logT));
  Fcent = 5.7000e-01 * exp(-T / 1.0000e-30) + 4.3000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[18] += -2.0 * (pres_mod[11] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 1.00797000e+00 + rho / ((m + 6.5 * conc[4] + 0.6 * conc[12] + 0.5 * conc[8] + 0.2 * conc[5] - 0.35 * conc[10] + 6.7 * conc[7] + 2.7 * conc[1] + 1.8 * conc[11]) * 1.00797000e+00)) * fwd_rxn_rates[43] + pres_mod[11] * (2.0 * (-mw_avg * fwd_rxn_rates[43] / 1.00797000e+00)));
  jac[18] += ((-mw_avg * fwd_rxn_rates[44] / 1.00797000e+00 + exp(3.08132330e+01 - (1.99777016e+03 / T)) * (rho / 1.00797000e+00) * conc[7]) + (-mw_avg * fwd_rxn_rates[44] / 1.00797000e+00));
  jac[18] += -1.0 * ((-mw_avg * fwd_rxn_rates[45] / 1.00797000e+00) + (-mw_avg * fwd_rxn_rates[45] / 1.00797000e+00));
  jac[18] += ((-mw_avg * fwd_rxn_rates[48] / 1.00797000e+00) + (-mw_avg * fwd_rxn_rates[48] / 1.00797000e+00));
  jac[18] += -1.0 * ((-mw_avg * fwd_rxn_rates[49] / 1.00797000e+00) + (-mw_avg * fwd_rxn_rates[49] / 1.00797000e+00));
  jac[18] += -1.0 * ((-mw_avg * fwd_rxn_rates[50] / 1.00797000e+00) + (-mw_avg * fwd_rxn_rates[50] / 1.00797000e+00));
  jac[18] += ((-mw_avg * fwd_rxn_rates[51] / 1.00797000e+00) + (-mw_avg * fwd_rxn_rates[51] / 1.00797000e+00));
  jac[18] += -1.0 * ((-mw_avg * fwd_rxn_rates[52] / 1.00797000e+00) + (-mw_avg * fwd_rxn_rates[52] / 1.00797000e+00));
  jac[18] += ((-mw_avg * fwd_rxn_rates[53] / 1.00797000e+00) + (-mw_avg * fwd_rxn_rates[53] / 1.00797000e+00));
  jac[18] += sp_rates[3] * mw_avg / 1.00797000e+00;
  jac[18] *= 1.70073700e+01 / rho;

  //partial of omega_OH wrt Y_H2;
  jac[32] = ((-mw_avg * fwd_rxn_rates[0] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[0] / 2.01594000e+00));
  jac[32] += -1.0 * ((-mw_avg * fwd_rxn_rates[1] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[1] / 2.01594000e+00));
  jac[32] += ((-mw_avg * fwd_rxn_rates[2] / 2.01594000e+00 + exp(2.89707478e+01 - (3.99956606e+03 / T)) * (rho / 2.01594000e+00) * conc[2]) + (-mw_avg * fwd_rxn_rates[2] / 2.01594000e+00));
  jac[32] += -1.0 * ((-mw_avg * fwd_rxn_rates[3] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[3] / 2.01594000e+00));
  jac[32] += ((-mw_avg * fwd_rxn_rates[4] / 2.01594000e+00 + exp(3.44100335e+01 - (9.64666348e+03 / T)) * (rho / 2.01594000e+00) * conc[2]) + (-mw_avg * fwd_rxn_rates[4] / 2.01594000e+00));
  jac[32] += -1.0 * ((-mw_avg * fwd_rxn_rates[5] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[5] / 2.01594000e+00));
  jac[32] += -1.0 * ((-mw_avg * fwd_rxn_rates[6] / 2.01594000e+00 + exp(1.91907890e+01 + 1.51 * logT - (1.72603316e+03 / T)) * (rho / 2.01594000e+00) * conc[3]) + (-mw_avg * fwd_rxn_rates[6] / 2.01594000e+00));
  jac[32] += ((-mw_avg * fwd_rxn_rates[7] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[7] / 2.01594000e+00));
  jac[32] += -2.0 * (2.0 * (-mw_avg * fwd_rxn_rates[8] / 2.01594000e+00));
  jac[32] += 2.0 * ((-mw_avg * fwd_rxn_rates[9] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[9] / 2.01594000e+00));
  jac[32] += ((-mw_avg * pres_mod[4] / 2.01594000e+00 + 2.5 * rho / 2.01594000e+00) * fwd_rxn_rates[22] + pres_mod[4] * ((-mw_avg * fwd_rxn_rates[22] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[22] / 2.01594000e+00)));
  jac[32] += -1.0 * ((-mw_avg * pres_mod[5] / 2.01594000e+00 + 2.5 * rho / 2.01594000e+00) * fwd_rxn_rates[23] + pres_mod[5] * ((-mw_avg * fwd_rxn_rates[23] / 2.01594000e+00)));
  jac[32] += ((-mw_avg * pres_mod[6] / 2.01594000e+00 + 3.0 * rho / 2.01594000e+00) * fwd_rxn_rates[24] + pres_mod[6] * ((-mw_avg * fwd_rxn_rates[24] / 2.01594000e+00)));
  jac[32] += -1.0 * ((-mw_avg * pres_mod[7] / 2.01594000e+00 + 3.0 * rho / 2.01594000e+00) * fwd_rxn_rates[25] + pres_mod[7] * ((-mw_avg * fwd_rxn_rates[25] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[25] / 2.01594000e+00)));
  jac[32] += (2.0 * (-mw_avg * fwd_rxn_rates[26] / 2.01594000e+00));
  jac[32] += -1.0 * ((-mw_avg * fwd_rxn_rates[27] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[27] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[27] / 2.01594000e+00));
  jac[32] += 2.0 * ((-mw_avg * fwd_rxn_rates[32] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[32] / 2.01594000e+00));
  jac[32] += -2.0 * (2.0 * (-mw_avg * fwd_rxn_rates[33] / 2.01594000e+00));
  jac[32] += ((-mw_avg * fwd_rxn_rates[34] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[34] / 2.01594000e+00));
  jac[32] += -1.0 * ((-mw_avg * fwd_rxn_rates[35] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[35] / 2.01594000e+00));
  jac[32] += -1.0 * ((-mw_avg * fwd_rxn_rates[36] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[36] / 2.01594000e+00));
  jac[32] += ((-mw_avg * fwd_rxn_rates[37] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[37] / 2.01594000e+00));
  Pr = (m + 6.5 * conc[4] + 0.6 * conc[12] + 0.5 * conc[8] + 0.2 * conc[5] - 0.35 * conc[10] + 6.7 * conc[7] + 2.7 * conc[1] + 1.8 * conc[11]) * (exp(2.78501566e+01 - 3.2 * logT));
  Fcent = 5.7000e-01 * exp(-T / 1.0000e-30) + 4.3000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[32] += 2.0 * (pres_mod[10] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 2.01594000e+00 + rho * 3.7 / ((m + 6.5 * conc[4] + 0.6 * conc[12] + 0.5 * conc[8] + 0.2 * conc[5] - 0.35 * conc[10] + 6.7 * conc[7] + 2.7 * conc[1] + 1.8 * conc[11]) * 2.01594000e+00)) * fwd_rxn_rates[42] + pres_mod[10] * ((-mw_avg * fwd_rxn_rates[42] / 2.01594000e+00)));
  Pr = (m + 6.5 * conc[4] + 0.6 * conc[12] + 0.5 * conc[8] + 0.2 * conc[5] - 0.35 * conc[10] + 6.7 * conc[7] + 2.7 * conc[1] + 1.8 * conc[11]) * (exp(2.78501521e+01 - 3.20001 * logT));
  Fcent = 5.7000e-01 * exp(-T / 1.0000e-30) + 4.3000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[32] += -2.0 * (pres_mod[11] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 2.01594000e+00 + rho * 3.7 / ((m + 6.5 * conc[4] + 0.6 * conc[12] + 0.5 * conc[8] + 0.2 * conc[5] - 0.35 * conc[10] + 6.7 * conc[7] + 2.7 * conc[1] + 1.8 * conc[11]) * 2.01594000e+00)) * fwd_rxn_rates[43] + pres_mod[11] * (2.0 * (-mw_avg * fwd_rxn_rates[43] / 2.01594000e+00)));
  jac[32] += ((-mw_avg * fwd_rxn_rates[44] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[44] / 2.01594000e+00));
  jac[32] += -1.0 * ((-mw_avg * fwd_rxn_rates[45] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[45] / 2.01594000e+00));
  jac[32] += ((-mw_avg * fwd_rxn_rates[48] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[48] / 2.01594000e+00));
  jac[32] += -1.0 * ((-mw_avg * fwd_rxn_rates[49] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[49] / 2.01594000e+00));
  jac[32] += -1.0 * ((-mw_avg * fwd_rxn_rates[50] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[50] / 2.01594000e+00));
  jac[32] += ((-mw_avg * fwd_rxn_rates[51] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[51] / 2.01594000e+00));
  jac[32] += -1.0 * ((-mw_avg * fwd_rxn_rates[52] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[52] / 2.01594000e+00));
  jac[32] += ((-mw_avg * fwd_rxn_rates[53] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[53] / 2.01594000e+00));
  jac[32] += sp_rates[3] * mw_avg / 2.01594000e+00;
  jac[32] *= 1.70073700e+01 / rho;

  //partial of omega_OH wrt Y_O;
  jac[46] = ((-mw_avg * fwd_rxn_rates[0] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[0] / 1.59994000e+01));
  jac[46] += -1.0 * ((-mw_avg * fwd_rxn_rates[1] / 1.59994000e+01 + exp(2.63075513e+01 + 0.40478 * logT - (-7.48736077e+02 / T)) * (rho / 1.59994000e+01) * conc[3]) + (-mw_avg * fwd_rxn_rates[1] / 1.59994000e+01));
  jac[46] += ((-mw_avg * fwd_rxn_rates[2] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[2] / 1.59994000e+01 + exp(2.89707478e+01 - (3.99956606e+03 / T)) * (rho / 1.59994000e+01) * conc[1]));
  jac[46] += -1.0 * ((-mw_avg * fwd_rxn_rates[3] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[3] / 1.59994000e+01));
  jac[46] += ((-mw_avg * fwd_rxn_rates[4] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[4] / 1.59994000e+01 + exp(3.44100335e+01 - (9.64666348e+03 / T)) * (rho / 1.59994000e+01) * conc[1]));
  jac[46] += -1.0 * ((-mw_avg * fwd_rxn_rates[5] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[5] / 1.59994000e+01));
  jac[46] += -1.0 * ((-mw_avg * fwd_rxn_rates[6] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[6] / 1.59994000e+01));
  jac[46] += ((-mw_avg * fwd_rxn_rates[7] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[7] / 1.59994000e+01));
  jac[46] += -2.0 * (2.0 * (-mw_avg * fwd_rxn_rates[8] / 1.59994000e+01));
  jac[46] += 2.0 * ((-mw_avg * fwd_rxn_rates[9] / 1.59994000e+01 + exp(1.50329128e+01 + 2.1464 * logT - (7.53063740e+03 / T)) * (rho / 1.59994000e+01) * conc[4]) + (-mw_avg * fwd_rxn_rates[9] / 1.59994000e+01));
  jac[46] += ((-mw_avg * pres_mod[4] / 1.59994000e+01 + rho / 1.59994000e+01) * fwd_rxn_rates[22] + pres_mod[4] * ((-mw_avg * fwd_rxn_rates[22] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[22] / 1.59994000e+01 + exp(4.29970685e+01 - 1.0 * logT) * (rho / 1.59994000e+01) * conc[0])));
  jac[46] += -1.0 * ((-mw_avg * pres_mod[5] / 1.59994000e+01 + rho / 1.59994000e+01) * fwd_rxn_rates[23] + pres_mod[5] * ((-mw_avg * fwd_rxn_rates[23] / 1.59994000e+01)));
  jac[46] += ((-mw_avg * pres_mod[6] / 1.59994000e+01 + rho / 1.59994000e+01) * fwd_rxn_rates[24] + pres_mod[6] * ((-mw_avg * fwd_rxn_rates[24] / 1.59994000e+01)));
  jac[46] += -1.0 * ((-mw_avg * pres_mod[7] / 1.59994000e+01 + rho / 1.59994000e+01) * fwd_rxn_rates[25] + pres_mod[7] * ((-mw_avg * fwd_rxn_rates[25] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[25] / 1.59994000e+01)));
  jac[46] += (2.0 * (-mw_avg * fwd_rxn_rates[26] / 1.59994000e+01));
  jac[46] += -1.0 * ((-mw_avg * fwd_rxn_rates[27] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[27] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[27] / 1.59994000e+01));
  jac[46] += 2.0 * ((-mw_avg * fwd_rxn_rates[32] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[32] / 1.59994000e+01));
  jac[46] += -2.0 * (2.0 * (-mw_avg * fwd_rxn_rates[33] / 1.59994000e+01));
  jac[46] += ((-mw_avg * fwd_rxn_rates[34] / 1.59994000e+01 + exp(2.40731699e+01 + 1.0 * logT - (-3.64293641e+02 / T)) * (rho / 1.59994000e+01) * conc[6]) + (-mw_avg * fwd_rxn_rates[34] / 1.59994000e+01));
  jac[46] += -1.0 * ((-mw_avg * fwd_rxn_rates[35] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[35] / 1.59994000e+01));
  jac[46] += -1.0 * ((-mw_avg * fwd_rxn_rates[36] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[36] / 1.59994000e+01));
  jac[46] += ((-mw_avg * fwd_rxn_rates[37] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[37] / 1.59994000e+01));
  Pr = (m + 6.5 * conc[4] + 0.6 * conc[12] + 0.5 * conc[8] + 0.2 * conc[5] - 0.35 * conc[10] + 6.7 * conc[7] + 2.7 * conc[1] + 1.8 * conc[11]) * (exp(2.78501566e+01 - 3.2 * logT));
  Fcent = 5.7000e-01 * exp(-T / 1.0000e-30) + 4.3000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[46] += 2.0 * (pres_mod[10] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 1.59994000e+01 + rho / ((m + 6.5 * conc[4] + 0.6 * conc[12] + 0.5 * conc[8] + 0.2 * conc[5] - 0.35 * conc[10] + 6.7 * conc[7] + 2.7 * conc[1] + 1.8 * conc[11]) * 1.59994000e+01)) * fwd_rxn_rates[42] + pres_mod[10] * ((-mw_avg * fwd_rxn_rates[42] / 1.59994000e+01)));
  Pr = (m + 6.5 * conc[4] + 0.6 * conc[12] + 0.5 * conc[8] + 0.2 * conc[5] - 0.35 * conc[10] + 6.7 * conc[7] + 2.7 * conc[1] + 1.8 * conc[11]) * (exp(2.78501521e+01 - 3.20001 * logT));
  Fcent = 5.7000e-01 * exp(-T / 1.0000e-30) + 4.3000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[46] += -2.0 * (pres_mod[11] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 1.59994000e+01 + rho / ((m + 6.5 * conc[4] + 0.6 * conc[12] + 0.5 * conc[8] + 0.2 * conc[5] - 0.35 * conc[10] + 6.7 * conc[7] + 2.7 * conc[1] + 1.8 * conc[11]) * 1.59994000e+01)) * fwd_rxn_rates[43] + pres_mod[11] * (2.0 * (-mw_avg * fwd_rxn_rates[43] / 1.59994000e+01)));
  jac[46] += ((-mw_avg * fwd_rxn_rates[44] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[44] / 1.59994000e+01));
  jac[46] += -1.0 * ((-mw_avg * fwd_rxn_rates[45] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[45] / 1.59994000e+01));
  jac[46] += ((-mw_avg * fwd_rxn_rates[48] / 1.59994000e+01 + exp(1.60720517e+01 + 2.0 * logT - (1.99777016e+03 / T)) * (rho / 1.59994000e+01) * conc[7]) + (-mw_avg * fwd_rxn_rates[48] / 1.59994000e+01));
  jac[46] += -1.0 * ((-mw_avg * fwd_rxn_rates[49] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[49] / 1.59994000e+01));
  jac[46] += -1.0 * ((-mw_avg * fwd_rxn_rates[50] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[50] / 1.59994000e+01));
  jac[46] += ((-mw_avg * fwd_rxn_rates[51] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[51] / 1.59994000e+01));
  jac[46] += -1.0 * ((-mw_avg * fwd_rxn_rates[52] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[52] / 1.59994000e+01));
  jac[46] += ((-mw_avg * fwd_rxn_rates[53] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[53] / 1.59994000e+01));
  jac[46] += sp_rates[3] * mw_avg / 1.59994000e+01;
  jac[46] *= 1.70073700e+01 / rho;

  //partial of omega_OH wrt Y_OH;
  jac[60] = ((-mw_avg * fwd_rxn_rates[0] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[0] / 1.70073700e+01));
  jac[60] += -1.0 * ((-mw_avg * fwd_rxn_rates[1] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[1] / 1.70073700e+01 + exp(2.63075513e+01 + 0.40478 * logT - (-7.48736077e+02 / T)) * (rho / 1.70073700e+01) * conc[2]));
  jac[60] += ((-mw_avg * fwd_rxn_rates[2] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[2] / 1.70073700e+01));
  jac[60] += -1.0 * ((-mw_avg * fwd_rxn_rates[3] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[3] / 1.70073700e+01 + exp(2.85866095e+01 - 0.053533 * logT - (3.32706731e+03 / T)) * (rho / 1.70073700e+01) * conc[0]));
  jac[60] += ((-mw_avg * fwd_rxn_rates[4] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[4] / 1.70073700e+01));
  jac[60] += -1.0 * ((-mw_avg * fwd_rxn_rates[5] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[5] / 1.70073700e+01 + exp(3.40258987e+01 - 0.053533 * logT - (8.97436602e+03 / T)) * (rho / 1.70073700e+01) * conc[0]));
  jac[60] += -1.0 * ((-mw_avg * fwd_rxn_rates[6] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[6] / 1.70073700e+01 + exp(1.91907890e+01 + 1.51 * logT - (1.72603316e+03 / T)) * (rho / 1.70073700e+01) * conc[1]));
  jac[60] += ((-mw_avg * fwd_rxn_rates[7] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[7] / 1.70073700e+01));
  jac[60] += -2.0 * (2.0 * (-mw_avg * fwd_rxn_rates[8] / 1.70073700e+01 + exp(1.04163112e+01 + 2.42 * logT - (-9.71208165e+02 / T)) * (rho / 1.70073700e+01) * conc[3]));
  jac[60] += 2.0 * ((-mw_avg * fwd_rxn_rates[9] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[9] / 1.70073700e+01));
  jac[60] += ((-mw_avg * pres_mod[4] / 1.70073700e+01 + rho / 1.70073700e+01) * fwd_rxn_rates[22] + pres_mod[4] * ((-mw_avg * fwd_rxn_rates[22] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[22] / 1.70073700e+01)));
  jac[60] += -1.0 * ((-mw_avg * pres_mod[5] / 1.70073700e+01 + rho / 1.70073700e+01) * fwd_rxn_rates[23] + pres_mod[5] * ((-mw_avg * fwd_rxn_rates[23] / 1.70073700e+01 + exp(4.38224602e+01 - 1.0301 * logT - (5.18313166e+04 / T)) * (rho / 1.70073700e+01))));
  jac[60] += ((-mw_avg * pres_mod[6] / 1.70073700e+01 + rho / 1.70073700e+01) * fwd_rxn_rates[24] + pres_mod[6] * ((-mw_avg * fwd_rxn_rates[24] / 1.70073700e+01)));
  jac[60] += -1.0 * ((-mw_avg * pres_mod[7] / 1.70073700e+01 + rho / 1.70073700e+01) * fwd_rxn_rates[25] + pres_mod[7] * ((-mw_avg * fwd_rxn_rates[25] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[25] / 1.70073700e+01 + exp(5.85301272e+01 - 3.0183 * logT - (4.50771425e+02 / T)) * (rho / 1.70073700e+01) * conc[0])));
  jac[60] += (2.0 * (-mw_avg * fwd_rxn_rates[26] / 1.70073700e+01));
  jac[60] += -1.0 * ((-mw_avg * fwd_rxn_rates[27] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[27] / 1.70073700e+01 + exp(5.44311720e+01 - 2.1363 * logT - (1.43809259e+02 / T)) * (rho / 1.70073700e+01) * conc[0] * conc[4]) + (-mw_avg * fwd_rxn_rates[27] / 1.70073700e+01));
  jac[60] += 2.0 * ((-mw_avg * fwd_rxn_rates[32] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[32] / 1.70073700e+01));
  jac[60] += -2.0 * (2.0 * (-mw_avg * fwd_rxn_rates[33] / 1.70073700e+01 + exp(2.25013658e+01 + 0.86409 * logT - (1.83206092e+04 / T)) * (rho / 1.70073700e+01) * conc[3]));
  jac[60] += ((-mw_avg * fwd_rxn_rates[34] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[34] / 1.70073700e+01));
  jac[60] += -1.0 * ((-mw_avg * fwd_rxn_rates[35] / 1.70073700e+01 + exp(2.06516517e+01 + 1.4593 * logT - (2.62487877e+04 / T)) * (rho / 1.70073700e+01) * conc[5]) + (-mw_avg * fwd_rxn_rates[35] / 1.70073700e+01));
  jac[60] += -1.0 * ((-mw_avg * fwd_rxn_rates[36] / 1.70073700e+01 + exp(3.09948627e+01 - (-2.50098683e+02 / T)) * (rho / 1.70073700e+01) * conc[6]) + (-mw_avg * fwd_rxn_rates[36] / 1.70073700e+01));
  jac[60] += ((-mw_avg * fwd_rxn_rates[37] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[37] / 1.70073700e+01));
  Pr = (m + 6.5 * conc[4] + 0.6 * conc[12] + 0.5 * conc[8] + 0.2 * conc[5] - 0.35 * conc[10] + 6.7 * conc[7] + 2.7 * conc[1] + 1.8 * conc[11]) * (exp(2.78501566e+01 - 3.2 * logT));
  Fcent = 5.7000e-01 * exp(-T / 1.0000e-30) + 4.3000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[60] += 2.0 * (pres_mod[10] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 1.70073700e+01 + rho / ((m + 6.5 * conc[4] + 0.6 * conc[12] + 0.5 * conc[8] + 0.2 * conc[5] - 0.35 * conc[10] + 6.7 * conc[7] + 2.7 * conc[1] + 1.8 * conc[11]) * 1.70073700e+01)) * fwd_rxn_rates[42] + pres_mod[10] * ((-mw_avg * fwd_rxn_rates[42] / 1.70073700e+01)));
  Pr = (m + 6.5 * conc[4] + 0.6 * conc[12] + 0.5 * conc[8] + 0.2 * conc[5] - 0.35 * conc[10] + 6.7 * conc[7] + 2.7 * conc[1] + 1.8 * conc[11]) * (exp(2.78501521e+01 - 3.20001 * logT));
  Fcent = 5.7000e-01 * exp(-T / 1.0000e-30) + 4.3000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[60] += -2.0 * (pres_mod[11] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 1.70073700e+01 + rho / ((m + 6.5 * conc[4] + 0.6 * conc[12] + 0.5 * conc[8] + 0.2 * conc[5] - 0.35 * conc[10] + 6.7 * conc[7] + 2.7 * conc[1] + 1.8 * conc[11]) * 1.70073700e+01)) * fwd_rxn_rates[43] + pres_mod[11] * (2.0 * (-mw_avg * fwd_rxn_rates[43] / 1.70073700e+01 + exp(1.09390890e+01 + 2.488 * logT - (-1.80654783e+03 / T)) * (rho / 1.70073700e+01) * conc[3])));
  jac[60] += ((-mw_avg * fwd_rxn_rates[44] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[44] / 1.70073700e+01));
  jac[60] += -1.0 * ((-mw_avg * fwd_rxn_rates[45] / 1.70073700e+01 + exp(1.88701627e+01 + 1.2843 * logT - (3.59925720e+04 / T)) * (rho / 1.70073700e+01) * conc[4]) + (-mw_avg * fwd_rxn_rates[45] / 1.70073700e+01));
  jac[60] += ((-mw_avg * fwd_rxn_rates[48] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[48] / 1.70073700e+01));
  jac[60] += -1.0 * ((-mw_avg * fwd_rxn_rates[49] / 1.70073700e+01 + exp(8.90174786e+00 + 2.6938 * logT - (9.31906943e+03 / T)) * (rho / 1.70073700e+01) * conc[6]) + (-mw_avg * fwd_rxn_rates[49] / 1.70073700e+01));
  jac[60] += -1.0 * ((-mw_avg * fwd_rxn_rates[50] / 1.70073700e+01 + exp(2.81849062e+01 - (1.60022900e+02 / T)) * (rho / 1.70073700e+01) * conc[7]) + (-mw_avg * fwd_rxn_rates[50] / 1.70073700e+01));
  jac[60] += ((-mw_avg * fwd_rxn_rates[51] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[51] / 1.70073700e+01));
  jac[60] += -1.0 * ((-mw_avg * fwd_rxn_rates[52] / 1.70073700e+01 + exp(3.19604378e+01 - (3.65838516e+03 / T)) * (rho / 1.70073700e+01) * conc[7]) + (-mw_avg * fwd_rxn_rates[52] / 1.70073700e+01));
  jac[60] += ((-mw_avg * fwd_rxn_rates[53] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[53] / 1.70073700e+01));
  jac[60] += sp_rates[3] * mw_avg / 1.70073700e+01;
  jac[60] *= 1.70073700e+01 / rho;

  //partial of omega_OH wrt Y_H2O;
  jac[74] = ((-mw_avg * fwd_rxn_rates[0] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[0] / 1.80153400e+01));
  jac[74] += -1.0 * ((-mw_avg * fwd_rxn_rates[1] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[1] / 1.80153400e+01));
  jac[74] += ((-mw_avg * fwd_rxn_rates[2] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[2] / 1.80153400e+01));
  jac[74] += -1.0 * ((-mw_avg * fwd_rxn_rates[3] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[3] / 1.80153400e+01));
  jac[74] += ((-mw_avg * fwd_rxn_rates[4] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[4] / 1.80153400e+01));
  jac[74] += -1.0 * ((-mw_avg * fwd_rxn_rates[5] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[5] / 1.80153400e+01));
  jac[74] += -1.0 * ((-mw_avg * fwd_rxn_rates[6] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[6] / 1.80153400e+01));
  jac[74] += ((-mw_avg * fwd_rxn_rates[7] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[7] / 1.80153400e+01 + exp(2.34232839e+01 + 1.1829 * logT - (9.55507805e+03 / T)) * (rho / 1.80153400e+01) * conc[0]));
  jac[74] += -2.0 * (2.0 * (-mw_avg * fwd_rxn_rates[8] / 1.80153400e+01));
  jac[74] += 2.0 * ((-mw_avg * fwd_rxn_rates[9] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[9] / 1.80153400e+01 + exp(1.50329128e+01 + 2.1464 * logT - (7.53063740e+03 / T)) * (rho / 1.80153400e+01) * conc[2]));
  jac[74] += ((-mw_avg * pres_mod[4] / 1.80153400e+01 + 12.0 * rho / 1.80153400e+01) * fwd_rxn_rates[22] + pres_mod[4] * ((-mw_avg * fwd_rxn_rates[22] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[22] / 1.80153400e+01)));
  jac[74] += -1.0 * ((-mw_avg * pres_mod[5] / 1.80153400e+01 + 12.0 * rho / 1.80153400e+01) * fwd_rxn_rates[23] + pres_mod[5] * ((-mw_avg * fwd_rxn_rates[23] / 1.80153400e+01)));
  jac[74] += ((-mw_avg * pres_mod[6] / 1.80153400e+01) * fwd_rxn_rates[24] + pres_mod[6] * ((-mw_avg * fwd_rxn_rates[24] / 1.80153400e+01 + exp(6.39721672e+01 - 3.322 * logT - (6.07835411e+04 / T)) * (rho / 1.80153400e+01))));
  jac[74] += -1.0 * ((-mw_avg * pres_mod[7] / 1.80153400e+01) * fwd_rxn_rates[25] + pres_mod[7] * ((-mw_avg * fwd_rxn_rates[25] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[25] / 1.80153400e+01)));
  jac[74] += (2.0 * (-mw_avg * fwd_rxn_rates[26] / 1.80153400e+01 + exp(5.98731945e+01 - 2.44 * logT - (6.04765789e+04 / T)) * (rho / 1.80153400e+01) * conc[4]));
  jac[74] += -1.0 * ((-mw_avg * fwd_rxn_rates[27] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[27] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[27] / 1.80153400e+01 + exp(5.44311720e+01 - 2.1363 * logT - (1.43809259e+02 / T)) * (rho / 1.80153400e+01) * conc[0] * conc[3]));
  jac[74] += 2.0 * ((-mw_avg * fwd_rxn_rates[32] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[32] / 1.80153400e+01));
  jac[74] += -2.0 * (2.0 * (-mw_avg * fwd_rxn_rates[33] / 1.80153400e+01));
  jac[74] += ((-mw_avg * fwd_rxn_rates[34] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[34] / 1.80153400e+01));
  jac[74] += -1.0 * ((-mw_avg * fwd_rxn_rates[35] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[35] / 1.80153400e+01));
  jac[74] += -1.0 * ((-mw_avg * fwd_rxn_rates[36] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[36] / 1.80153400e+01));
  jac[74] += ((-mw_avg * fwd_rxn_rates[37] / 1.80153400e+01 + exp(3.21899484e+01 + 0.18574 * logT - (3.48643603e+04 / T)) * (rho / 1.80153400e+01) * conc[5]) + (-mw_avg * fwd_rxn_rates[37] / 1.80153400e+01));
  Pr = (m + 6.5 * conc[4] + 0.6 * conc[12] + 0.5 * conc[8] + 0.2 * conc[5] - 0.35 * conc[10] + 6.7 * conc[7] + 2.7 * conc[1] + 1.8 * conc[11]) * (exp(2.78501566e+01 - 3.2 * logT));
  Fcent = 5.7000e-01 * exp(-T / 1.0000e-30) + 4.3000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[74] += 2.0 * (pres_mod[10] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 1.80153400e+01 + rho * 7.5 / ((m + 6.5 * conc[4] + 0.6 * conc[12] + 0.5 * conc[8] + 0.2 * conc[5] - 0.35 * conc[10] + 6.7 * conc[7] + 2.7 * conc[1] + 1.8 * conc[11]) * 1.80153400e+01)) * fwd_rxn_rates[42] + pres_mod[10] * ((-mw_avg * fwd_rxn_rates[42] / 1.80153400e+01)));
  Pr = (m + 6.5 * conc[4] + 0.6 * conc[12] + 0.5 * conc[8] + 0.2 * conc[5] - 0.35 * conc[10] + 6.7 * conc[7] + 2.7 * conc[1] + 1.8 * conc[11]) * (exp(2.78501521e+01 - 3.20001 * logT));
  Fcent = 5.7000e-01 * exp(-T / 1.0000e-30) + 4.3000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[74] += -2.0 * (pres_mod[11] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 1.80153400e+01 + rho * 7.5 / ((m + 6.5 * conc[4] + 0.6 * conc[12] + 0.5 * conc[8] + 0.2 * conc[5] - 0.35 * conc[10] + 6.7 * conc[7] + 2.7 * conc[1] + 1.8 * conc[11]) * 1.80153400e+01)) * fwd_rxn_rates[43] + pres_mod[11] * (2.0 * (-mw_avg * fwd_rxn_rates[43] / 1.80153400e+01)));
  jac[74] += ((-mw_avg * fwd_rxn_rates[44] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[44] / 1.80153400e+01));
  jac[74] += -1.0 * ((-mw_avg * fwd_rxn_rates[45] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[45] / 1.80153400e+01 + exp(1.88701627e+01 + 1.2843 * logT - (3.59925720e+04 / T)) * (rho / 1.80153400e+01) * conc[3]));
  jac[74] += ((-mw_avg * fwd_rxn_rates[48] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[48] / 1.80153400e+01));
  jac[74] += -1.0 * ((-mw_avg * fwd_rxn_rates[49] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[49] / 1.80153400e+01));
  jac[74] += -1.0 * ((-mw_avg * fwd_rxn_rates[50] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[50] / 1.80153400e+01));
  jac[74] += ((-mw_avg * fwd_rxn_rates[51] / 1.80153400e+01 + exp(2.56312037e+01 + 0.42021 * logT - (1.59826645e+04 / T)) * (rho / 1.80153400e+01) * conc[6]) + (-mw_avg * fwd_rxn_rates[51] / 1.80153400e+01));
  jac[74] += -1.0 * ((-mw_avg * fwd_rxn_rates[52] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[52] / 1.80153400e+01));
  jac[74] += ((-mw_avg * fwd_rxn_rates[53] / 1.80153400e+01 + exp(2.94067528e+01 + 0.42021 * logT - (1.94810268e+04 / T)) * (rho / 1.80153400e+01) * conc[6]) + (-mw_avg * fwd_rxn_rates[53] / 1.80153400e+01));
  jac[74] += sp_rates[3] * mw_avg / 1.80153400e+01;
  jac[74] *= 1.70073700e+01 / rho;

  //partial of omega_OH wrt Y_O2;
  jac[88] = ((-mw_avg * fwd_rxn_rates[0] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[0] / 3.19988000e+01 + exp(3.22754120e+01 - (7.69216995e+03 / T)) * (rho / 3.19988000e+01) * conc[0]));
  jac[88] += -1.0 * ((-mw_avg * fwd_rxn_rates[1] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[1] / 3.19988000e+01));
  jac[88] += ((-mw_avg * fwd_rxn_rates[2] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[2] / 3.19988000e+01));
  jac[88] += -1.0 * ((-mw_avg * fwd_rxn_rates[3] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[3] / 3.19988000e+01));
  jac[88] += ((-mw_avg * fwd_rxn_rates[4] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[4] / 3.19988000e+01));
  jac[88] += -1.0 * ((-mw_avg * fwd_rxn_rates[5] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[5] / 3.19988000e+01));
  jac[88] += -1.0 * ((-mw_avg * fwd_rxn_rates[6] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[6] / 3.19988000e+01));
  jac[88] += ((-mw_avg * fwd_rxn_rates[7] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[7] / 3.19988000e+01));
  jac[88] += -2.0 * (2.0 * (-mw_avg * fwd_rxn_rates[8] / 3.19988000e+01));
  jac[88] += 2.0 * ((-mw_avg * fwd_rxn_rates[9] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[9] / 3.19988000e+01));
  jac[88] += ((-mw_avg * pres_mod[4] / 3.19988000e+01 + rho / 3.19988000e+01) * fwd_rxn_rates[22] + pres_mod[4] * ((-mw_avg * fwd_rxn_rates[22] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[22] / 3.19988000e+01)));
  jac[88] += -1.0 * ((-mw_avg * pres_mod[5] / 3.19988000e+01 + rho / 3.19988000e+01) * fwd_rxn_rates[23] + pres_mod[5] * ((-mw_avg * fwd_rxn_rates[23] / 3.19988000e+01)));
  jac[88] += ((-mw_avg * pres_mod[6] / 3.19988000e+01 + 1.5 * rho / 3.19988000e+01) * fwd_rxn_rates[24] + pres_mod[6] * ((-mw_avg * fwd_rxn_rates[24] / 3.19988000e+01)));
  jac[88] += -1.0 * ((-mw_avg * pres_mod[7] / 3.19988000e+01 + 1.5 * rho / 3.19988000e+01) * fwd_rxn_rates[25] + pres_mod[7] * ((-mw_avg * fwd_rxn_rates[25] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[25] / 3.19988000e+01)));
  jac[88] += (2.0 * (-mw_avg * fwd_rxn_rates[26] / 3.19988000e+01));
  jac[88] += -1.0 * ((-mw_avg * fwd_rxn_rates[27] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[27] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[27] / 3.19988000e+01));
  jac[88] += 2.0 * ((-mw_avg * fwd_rxn_rates[32] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[32] / 3.19988000e+01));
  jac[88] += -2.0 * (2.0 * (-mw_avg * fwd_rxn_rates[33] / 3.19988000e+01));
  jac[88] += ((-mw_avg * fwd_rxn_rates[34] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[34] / 3.19988000e+01));
  jac[88] += -1.0 * ((-mw_avg * fwd_rxn_rates[35] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[35] / 3.19988000e+01 + exp(2.06516517e+01 + 1.4593 * logT - (2.62487877e+04 / T)) * (rho / 3.19988000e+01) * conc[3]));
  jac[88] += -1.0 * ((-mw_avg * fwd_rxn_rates[36] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[36] / 3.19988000e+01));
  jac[88] += ((-mw_avg * fwd_rxn_rates[37] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[37] / 3.19988000e+01 + exp(3.21899484e+01 + 0.18574 * logT - (3.48643603e+04 / T)) * (rho / 3.19988000e+01) * conc[4]));
  Pr = (m + 6.5 * conc[4] + 0.6 * conc[12] + 0.5 * conc[8] + 0.2 * conc[5] - 0.35 * conc[10] + 6.7 * conc[7] + 2.7 * conc[1] + 1.8 * conc[11]) * (exp(2.78501566e+01 - 3.2 * logT));
  Fcent = 5.7000e-01 * exp(-T / 1.0000e-30) + 4.3000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[88] += 2.0 * (pres_mod[10] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 3.19988000e+01 + rho * 1.2 / ((m + 6.5 * conc[4] + 0.6 * conc[12] + 0.5 * conc[8] + 0.2 * conc[5] - 0.35 * conc[10] + 6.7 * conc[7] + 2.7 * conc[1] + 1.8 * conc[11]) * 3.19988000e+01)) * fwd_rxn_rates[42] + pres_mod[10] * ((-mw_avg * fwd_rxn_rates[42] / 3.19988000e+01)));
  Pr = (m + 6.5 * conc[4] + 0.6 * conc[12] + 0.5 * conc[8] + 0.2 * conc[5] - 0.35 * conc[10] + 6.7 * conc[7] + 2.7 * conc[1] + 1.8 * conc[11]) * (exp(2.78501521e+01 - 3.20001 * logT));
  Fcent = 5.7000e-01 * exp(-T / 1.0000e-30) + 4.3000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[88] += -2.0 * (pres_mod[11] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 3.19988000e+01 + rho * 1.2 / ((m + 6.5 * conc[4] + 0.6 * conc[12] + 0.5 * conc[8] + 0.2 * conc[5] - 0.35 * conc[10] + 6.7 * conc[7] + 2.7 * conc[1] + 1.8 * conc[11]) * 3.19988000e+01)) * fwd_rxn_rates[43] + pres_mod[11] * (2.0 * (-mw_avg * fwd_rxn_rates[43] / 3.19988000e+01)));
  jac[88] += ((-mw_avg * fwd_rxn_rates[44] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[44] / 3.19988000e+01));
  jac[88] += -1.0 * ((-mw_avg * fwd_rxn_rates[45] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[45] / 3.19988000e+01));
  jac[88] += ((-mw_avg * fwd_rxn_rates[48] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[48] / 3.19988000e+01));
  jac[88] += -1.0 * ((-mw_avg * fwd_rxn_rates[49] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[49] / 3.19988000e+01));
  jac[88] += -1.0 * ((-mw_avg * fwd_rxn_rates[50] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[50] / 3.19988000e+01));
  jac[88] += ((-mw_avg * fwd_rxn_rates[51] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[51] / 3.19988000e+01));
  jac[88] += -1.0 * ((-mw_avg * fwd_rxn_rates[52] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[52] / 3.19988000e+01));
  jac[88] += ((-mw_avg * fwd_rxn_rates[53] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[53] / 3.19988000e+01));
  jac[88] += sp_rates[3] * mw_avg / 3.19988000e+01;
  jac[88] *= 1.70073700e+01 / rho;

  //partial of omega_OH wrt Y_HO2;
  jac[102] = ((-mw_avg * fwd_rxn_rates[0] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[0] / 3.30067700e+01));
  jac[102] += -1.0 * ((-mw_avg * fwd_rxn_rates[1] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[1] / 3.30067700e+01));
  jac[102] += ((-mw_avg * fwd_rxn_rates[2] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[2] / 3.30067700e+01));
  jac[102] += -1.0 * ((-mw_avg * fwd_rxn_rates[3] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[3] / 3.30067700e+01));
  jac[102] += ((-mw_avg * fwd_rxn_rates[4] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[4] / 3.30067700e+01));
  jac[102] += -1.0 * ((-mw_avg * fwd_rxn_rates[5] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[5] / 3.30067700e+01));
  jac[102] += -1.0 * ((-mw_avg * fwd_rxn_rates[6] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[6] / 3.30067700e+01));
  jac[102] += ((-mw_avg * fwd_rxn_rates[7] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[7] / 3.30067700e+01));
  jac[102] += -2.0 * (2.0 * (-mw_avg * fwd_rxn_rates[8] / 3.30067700e+01));
  jac[102] += 2.0 * ((-mw_avg * fwd_rxn_rates[9] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[9] / 3.30067700e+01));
  jac[102] += ((-mw_avg * pres_mod[4] / 3.30067700e+01 + rho / 3.30067700e+01) * fwd_rxn_rates[22] + pres_mod[4] * ((-mw_avg * fwd_rxn_rates[22] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[22] / 3.30067700e+01)));
  jac[102] += -1.0 * ((-mw_avg * pres_mod[5] / 3.30067700e+01 + rho / 3.30067700e+01) * fwd_rxn_rates[23] + pres_mod[5] * ((-mw_avg * fwd_rxn_rates[23] / 3.30067700e+01)));
  jac[102] += ((-mw_avg * pres_mod[6] / 3.30067700e+01 + rho / 3.30067700e+01) * fwd_rxn_rates[24] + pres_mod[6] * ((-mw_avg * fwd_rxn_rates[24] / 3.30067700e+01)));
  jac[102] += -1.0 * ((-mw_avg * pres_mod[7] / 3.30067700e+01 + rho / 3.30067700e+01) * fwd_rxn_rates[25] + pres_mod[7] * ((-mw_avg * fwd_rxn_rates[25] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[25] / 3.30067700e+01)));
  jac[102] += (2.0 * (-mw_avg * fwd_rxn_rates[26] / 3.30067700e+01));
  jac[102] += -1.0 * ((-mw_avg * fwd_rxn_rates[27] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[27] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[27] / 3.30067700e+01));
  jac[102] += 2.0 * ((-mw_avg * fwd_rxn_rates[32] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[32] / 3.30067700e+01 + exp(3.18907389e+01 - (1.48448916e+02 / T)) * (rho / 3.30067700e+01) * conc[0]));
  jac[102] += -2.0 * (2.0 * (-mw_avg * fwd_rxn_rates[33] / 3.30067700e+01));
  jac[102] += ((-mw_avg * fwd_rxn_rates[34] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[34] / 3.30067700e+01 + exp(2.40731699e+01 + 1.0 * logT - (-3.64293641e+02 / T)) * (rho / 3.30067700e+01) * conc[2]));
  jac[102] += -1.0 * ((-mw_avg * fwd_rxn_rates[35] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[35] / 3.30067700e+01));
  jac[102] += -1.0 * ((-mw_avg * fwd_rxn_rates[36] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[36] / 3.30067700e+01 + exp(3.09948627e+01 - (-2.50098683e+02 / T)) * (rho / 3.30067700e+01) * conc[3]));
  jac[102] += ((-mw_avg * fwd_rxn_rates[37] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[37] / 3.30067700e+01));
  Pr = (m + 6.5 * conc[4] + 0.6 * conc[12] + 0.5 * conc[8] + 0.2 * conc[5] - 0.35 * conc[10] + 6.7 * conc[7] + 2.7 * conc[1] + 1.8 * conc[11]) * (exp(2.78501566e+01 - 3.2 * logT));
  Fcent = 5.7000e-01 * exp(-T / 1.0000e-30) + 4.3000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[102] += 2.0 * (pres_mod[10] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 3.30067700e+01 + rho / ((m + 6.5 * conc[4] + 0.6 * conc[12] + 0.5 * conc[8] + 0.2 * conc[5] - 0.35 * conc[10] + 6.7 * conc[7] + 2.7 * conc[1] + 1.8 * conc[11]) * 3.30067700e+01)) * fwd_rxn_rates[42] + pres_mod[10] * ((-mw_avg * fwd_rxn_rates[42] / 3.30067700e+01)));
  Pr = (m + 6.5 * conc[4] + 0.6 * conc[12] + 0.5 * conc[8] + 0.2 * conc[5] - 0.35 * conc[10] + 6.7 * conc[7] + 2.7 * conc[1] + 1.8 * conc[11]) * (exp(2.78501521e+01 - 3.20001 * logT));
  Fcent = 5.7000e-01 * exp(-T / 1.0000e-30) + 4.3000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[102] += -2.0 * (pres_mod[11] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 3.30067700e+01 + rho / ((m + 6.5 * conc[4] + 0.6 * conc[12] + 0.5 * conc[8] + 0.2 * conc[5] - 0.35 * conc[10] + 6.7 * conc[7] + 2.7 * conc[1] + 1.8 * conc[11]) * 3.30067700e+01)) * fwd_rxn_rates[43] + pres_mod[11] * (2.0 * (-mw_avg * fwd_rxn_rates[43] / 3.30067700e+01)));
  jac[102] += ((-mw_avg * fwd_rxn_rates[44] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[44] / 3.30067700e+01));
  jac[102] += -1.0 * ((-mw_avg * fwd_rxn_rates[45] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[45] / 3.30067700e+01));
  jac[102] += ((-mw_avg * fwd_rxn_rates[48] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[48] / 3.30067700e+01));
  jac[102] += -1.0 * ((-mw_avg * fwd_rxn_rates[49] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[49] / 3.30067700e+01 + exp(8.90174786e+00 + 2.6938 * logT - (9.31906943e+03 / T)) * (rho / 3.30067700e+01) * conc[3]));
  jac[102] += -1.0 * ((-mw_avg * fwd_rxn_rates[50] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[50] / 3.30067700e+01));
  jac[102] += ((-mw_avg * fwd_rxn_rates[51] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[51] / 3.30067700e+01 + exp(2.56312037e+01 + 0.42021 * logT - (1.59826645e+04 / T)) * (rho / 3.30067700e+01) * conc[4]));
  jac[102] += -1.0 * ((-mw_avg * fwd_rxn_rates[52] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[52] / 3.30067700e+01));
  jac[102] += ((-mw_avg * fwd_rxn_rates[53] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[53] / 3.30067700e+01 + exp(2.94067528e+01 + 0.42021 * logT - (1.94810268e+04 / T)) * (rho / 3.30067700e+01) * conc[4]));
  jac[102] += sp_rates[3] * mw_avg / 3.30067700e+01;
  jac[102] *= 1.70073700e+01 / rho;

  //partial of omega_OH wrt Y_H2O2;
  jac[116] = ((-mw_avg * fwd_rxn_rates[0] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[0] / 3.40147400e+01));
  jac[116] += -1.0 * ((-mw_avg * fwd_rxn_rates[1] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[1] / 3.40147400e+01));
  jac[116] += ((-mw_avg * fwd_rxn_rates[2] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[2] / 3.40147400e+01));
  jac[116] += -1.0 * ((-mw_avg * fwd_rxn_rates[3] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[3] / 3.40147400e+01));
  jac[116] += ((-mw_avg * fwd_rxn_rates[4] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[4] / 3.40147400e+01));
  jac[116] += -1.0 * ((-mw_avg * fwd_rxn_rates[5] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[5] / 3.40147400e+01));
  jac[116] += -1.0 * ((-mw_avg * fwd_rxn_rates[6] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[6] / 3.40147400e+01));
  jac[116] += ((-mw_avg * fwd_rxn_rates[7] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[7] / 3.40147400e+01));
  jac[116] += -2.0 * (2.0 * (-mw_avg * fwd_rxn_rates[8] / 3.40147400e+01));
  jac[116] += 2.0 * ((-mw_avg * fwd_rxn_rates[9] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[9] / 3.40147400e+01));
  jac[116] += ((-mw_avg * pres_mod[4] / 3.40147400e+01 + rho / 3.40147400e+01) * fwd_rxn_rates[22] + pres_mod[4] * ((-mw_avg * fwd_rxn_rates[22] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[22] / 3.40147400e+01)));
  jac[116] += -1.0 * ((-mw_avg * pres_mod[5] / 3.40147400e+01 + rho / 3.40147400e+01) * fwd_rxn_rates[23] + pres_mod[5] * ((-mw_avg * fwd_rxn_rates[23] / 3.40147400e+01)));
  jac[116] += ((-mw_avg * pres_mod[6] / 3.40147400e+01 + rho / 3.40147400e+01) * fwd_rxn_rates[24] + pres_mod[6] * ((-mw_avg * fwd_rxn_rates[24] / 3.40147400e+01)));
  jac[116] += -1.0 * ((-mw_avg * pres_mod[7] / 3.40147400e+01 + rho / 3.40147400e+01) * fwd_rxn_rates[25] + pres_mod[7] * ((-mw_avg * fwd_rxn_rates[25] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[25] / 3.40147400e+01)));
  jac[116] += (2.0 * (-mw_avg * fwd_rxn_rates[26] / 3.40147400e+01));
  jac[116] += -1.0 * ((-mw_avg * fwd_rxn_rates[27] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[27] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[27] / 3.40147400e+01));
  jac[116] += 2.0 * ((-mw_avg * fwd_rxn_rates[32] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[32] / 3.40147400e+01));
  jac[116] += -2.0 * (2.0 * (-mw_avg * fwd_rxn_rates[33] / 3.40147400e+01));
  jac[116] += ((-mw_avg * fwd_rxn_rates[34] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[34] / 3.40147400e+01));
  jac[116] += -1.0 * ((-mw_avg * fwd_rxn_rates[35] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[35] / 3.40147400e+01));
  jac[116] += -1.0 * ((-mw_avg * fwd_rxn_rates[36] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[36] / 3.40147400e+01));
  jac[116] += ((-mw_avg * fwd_rxn_rates[37] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[37] / 3.40147400e+01));
  Pr = (m + 6.5 * conc[4] + 0.6 * conc[12] + 0.5 * conc[8] + 0.2 * conc[5] - 0.35 * conc[10] + 6.7 * conc[7] + 2.7 * conc[1] + 1.8 * conc[11]) * (exp(2.78501566e+01 - 3.2 * logT));
  Fcent = 5.7000e-01 * exp(-T / 1.0000e-30) + 4.3000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[116] += 2.0 * (pres_mod[10] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 3.40147400e+01 + rho * 7.7 / ((m + 6.5 * conc[4] + 0.6 * conc[12] + 0.5 * conc[8] + 0.2 * conc[5] - 0.35 * conc[10] + 6.7 * conc[7] + 2.7 * conc[1] + 1.8 * conc[11]) * 3.40147400e+01)) * fwd_rxn_rates[42] + pres_mod[10] * ((-mw_avg * fwd_rxn_rates[42] / 3.40147400e+01 + exp(2.83241683e+01 + 0.9 * logT - (2.45313092e+04 / T)) * (rho / 3.40147400e+01))));
  Pr = (m + 6.5 * conc[4] + 0.6 * conc[12] + 0.5 * conc[8] + 0.2 * conc[5] - 0.35 * conc[10] + 6.7 * conc[7] + 2.7 * conc[1] + 1.8 * conc[11]) * (exp(2.78501521e+01 - 3.20001 * logT));
  Fcent = 5.7000e-01 * exp(-T / 1.0000e-30) + 4.3000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[116] += -2.0 * (pres_mod[11] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 3.40147400e+01 + rho * 7.7 / ((m + 6.5 * conc[4] + 0.6 * conc[12] + 0.5 * conc[8] + 0.2 * conc[5] - 0.35 * conc[10] + 6.7 * conc[7] + 2.7 * conc[1] + 1.8 * conc[11]) * 3.40147400e+01)) * fwd_rxn_rates[43] + pres_mod[11] * (2.0 * (-mw_avg * fwd_rxn_rates[43] / 3.40147400e+01)));
  jac[116] += ((-mw_avg * fwd_rxn_rates[44] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[44] / 3.40147400e+01 + exp(3.08132330e+01 - (1.99777016e+03 / T)) * (rho / 3.40147400e+01) * conc[0]));
  jac[116] += -1.0 * ((-mw_avg * fwd_rxn_rates[45] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[45] / 3.40147400e+01));
  jac[116] += ((-mw_avg * fwd_rxn_rates[48] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[48] / 3.40147400e+01 + exp(1.60720517e+01 + 2.0 * logT - (1.99777016e+03 / T)) * (rho / 3.40147400e+01) * conc[2]));
  jac[116] += -1.0 * ((-mw_avg * fwd_rxn_rates[49] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[49] / 3.40147400e+01));
  jac[116] += -1.0 * ((-mw_avg * fwd_rxn_rates[50] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[50] / 3.40147400e+01 + exp(2.81849062e+01 - (1.60022900e+02 / T)) * (rho / 3.40147400e+01) * conc[3]));
  jac[116] += ((-mw_avg * fwd_rxn_rates[51] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[51] / 3.40147400e+01));
  jac[116] += -1.0 * ((-mw_avg * fwd_rxn_rates[52] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[52] / 3.40147400e+01 + exp(3.19604378e+01 - (3.65838516e+03 / T)) * (rho / 3.40147400e+01) * conc[3]));
  jac[116] += ((-mw_avg * fwd_rxn_rates[53] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[53] / 3.40147400e+01));
  jac[116] += sp_rates[3] * mw_avg / 3.40147400e+01;
  jac[116] *= 1.70073700e+01 / rho;

  //partial of omega_OH wrt Y_N2;
  jac[130] = ((-mw_avg * fwd_rxn_rates[0] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[0] / 2.80134000e+01));
  jac[130] += -1.0 * ((-mw_avg * fwd_rxn_rates[1] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[1] / 2.80134000e+01));
  jac[130] += ((-mw_avg * fwd_rxn_rates[2] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[2] / 2.80134000e+01));
  jac[130] += -1.0 * ((-mw_avg * fwd_rxn_rates[3] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[3] / 2.80134000e+01));
  jac[130] += ((-mw_avg * fwd_rxn_rates[4] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[4] / 2.80134000e+01));
  jac[130] += -1.0 * ((-mw_avg * fwd_rxn_rates[5] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[5] / 2.80134000e+01));
  jac[130] += -1.0 * ((-mw_avg * fwd_rxn_rates[6] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[6] / 2.80134000e+01));
  jac[130] += ((-mw_avg * fwd_rxn_rates[7] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[7] / 2.80134000e+01));
  jac[130] += -2.0 * (2.0 * (-mw_avg * fwd_rxn_rates[8] / 2.80134000e+01));
  jac[130] += 2.0 * ((-mw_avg * fwd_rxn_rates[9] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[9] / 2.80134000e+01));
  jac[130] += ((-mw_avg * pres_mod[4] / 2.80134000e+01 + rho / 2.80134000e+01) * fwd_rxn_rates[22] + pres_mod[4] * ((-mw_avg * fwd_rxn_rates[22] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[22] / 2.80134000e+01)));
  jac[130] += -1.0 * ((-mw_avg * pres_mod[5] / 2.80134000e+01 + rho / 2.80134000e+01) * fwd_rxn_rates[23] + pres_mod[5] * ((-mw_avg * fwd_rxn_rates[23] / 2.80134000e+01)));
  jac[130] += ((-mw_avg * pres_mod[6] / 2.80134000e+01 + 2.0 * rho / 2.80134000e+01) * fwd_rxn_rates[24] + pres_mod[6] * ((-mw_avg * fwd_rxn_rates[24] / 2.80134000e+01)));
  jac[130] += -1.0 * ((-mw_avg * pres_mod[7] / 2.80134000e+01 + 2.0 * rho / 2.80134000e+01) * fwd_rxn_rates[25] + pres_mod[7] * ((-mw_avg * fwd_rxn_rates[25] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[25] / 2.80134000e+01)));
  jac[130] += (2.0 * (-mw_avg * fwd_rxn_rates[26] / 2.80134000e+01));
  jac[130] += -1.0 * ((-mw_avg * fwd_rxn_rates[27] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[27] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[27] / 2.80134000e+01));
  jac[130] += 2.0 * ((-mw_avg * fwd_rxn_rates[32] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[32] / 2.80134000e+01));
  jac[130] += -2.0 * (2.0 * (-mw_avg * fwd_rxn_rates[33] / 2.80134000e+01));
  jac[130] += ((-mw_avg * fwd_rxn_rates[34] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[34] / 2.80134000e+01));
  jac[130] += -1.0 * ((-mw_avg * fwd_rxn_rates[35] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[35] / 2.80134000e+01));
  jac[130] += -1.0 * ((-mw_avg * fwd_rxn_rates[36] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[36] / 2.80134000e+01));
  jac[130] += ((-mw_avg * fwd_rxn_rates[37] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[37] / 2.80134000e+01));
  Pr = (m + 6.5 * conc[4] + 0.6 * conc[12] + 0.5 * conc[8] + 0.2 * conc[5] - 0.35 * conc[10] + 6.7 * conc[7] + 2.7 * conc[1] + 1.8 * conc[11]) * (exp(2.78501566e+01 - 3.2 * logT));
  Fcent = 5.7000e-01 * exp(-T / 1.0000e-30) + 4.3000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[130] += 2.0 * (pres_mod[10] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 2.80134000e+01 + rho * 1.5 / ((m + 6.5 * conc[4] + 0.6 * conc[12] + 0.5 * conc[8] + 0.2 * conc[5] - 0.35 * conc[10] + 6.7 * conc[7] + 2.7 * conc[1] + 1.8 * conc[11]) * 2.80134000e+01)) * fwd_rxn_rates[42] + pres_mod[10] * ((-mw_avg * fwd_rxn_rates[42] / 2.80134000e+01)));
  Pr = (m + 6.5 * conc[4] + 0.6 * conc[12] + 0.5 * conc[8] + 0.2 * conc[5] - 0.35 * conc[10] + 6.7 * conc[7] + 2.7 * conc[1] + 1.8 * conc[11]) * (exp(2.78501521e+01 - 3.20001 * logT));
  Fcent = 5.7000e-01 * exp(-T / 1.0000e-30) + 4.3000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[130] += -2.0 * (pres_mod[11] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 2.80134000e+01 + rho * 1.5 / ((m + 6.5 * conc[4] + 0.6 * conc[12] + 0.5 * conc[8] + 0.2 * conc[5] - 0.35 * conc[10] + 6.7 * conc[7] + 2.7 * conc[1] + 1.8 * conc[11]) * 2.80134000e+01)) * fwd_rxn_rates[43] + pres_mod[11] * (2.0 * (-mw_avg * fwd_rxn_rates[43] / 2.80134000e+01)));
  jac[130] += ((-mw_avg * fwd_rxn_rates[44] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[44] / 2.80134000e+01));
  jac[130] += -1.0 * ((-mw_avg * fwd_rxn_rates[45] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[45] / 2.80134000e+01));
  jac[130] += ((-mw_avg * fwd_rxn_rates[48] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[48] / 2.80134000e+01));
  jac[130] += -1.0 * ((-mw_avg * fwd_rxn_rates[49] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[49] / 2.80134000e+01));
  jac[130] += -1.0 * ((-mw_avg * fwd_rxn_rates[50] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[50] / 2.80134000e+01));
  jac[130] += ((-mw_avg * fwd_rxn_rates[51] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[51] / 2.80134000e+01));
  jac[130] += -1.0 * ((-mw_avg * fwd_rxn_rates[52] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[52] / 2.80134000e+01));
  jac[130] += ((-mw_avg * fwd_rxn_rates[53] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[53] / 2.80134000e+01));
  jac[130] += sp_rates[3] * mw_avg / 2.80134000e+01;
  jac[130] *= 1.70073700e+01 / rho;

  //partial of omega_OH wrt Y_AR;
  jac[144] = ((-mw_avg * fwd_rxn_rates[0] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[0] / 3.99480000e+01));
  jac[144] += -1.0 * ((-mw_avg * fwd_rxn_rates[1] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[1] / 3.99480000e+01));
  jac[144] += ((-mw_avg * fwd_rxn_rates[2] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[2] / 3.99480000e+01));
  jac[144] += -1.0 * ((-mw_avg * fwd_rxn_rates[3] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[3] / 3.99480000e+01));
  jac[144] += ((-mw_avg * fwd_rxn_rates[4] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[4] / 3.99480000e+01));
  jac[144] += -1.0 * ((-mw_avg * fwd_rxn_rates[5] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[5] / 3.99480000e+01));
  jac[144] += -1.0 * ((-mw_avg * fwd_rxn_rates[6] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[6] / 3.99480000e+01));
  jac[144] += ((-mw_avg * fwd_rxn_rates[7] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[7] / 3.99480000e+01));
  jac[144] += -2.0 * (2.0 * (-mw_avg * fwd_rxn_rates[8] / 3.99480000e+01));
  jac[144] += 2.0 * ((-mw_avg * fwd_rxn_rates[9] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[9] / 3.99480000e+01));
  jac[144] += ((-mw_avg * pres_mod[4] / 3.99480000e+01 + 0.75 * rho / 3.99480000e+01) * fwd_rxn_rates[22] + pres_mod[4] * ((-mw_avg * fwd_rxn_rates[22] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[22] / 3.99480000e+01)));
  jac[144] += -1.0 * ((-mw_avg * pres_mod[5] / 3.99480000e+01 + 0.75 * rho / 3.99480000e+01) * fwd_rxn_rates[23] + pres_mod[5] * ((-mw_avg * fwd_rxn_rates[23] / 3.99480000e+01)));
  jac[144] += ((-mw_avg * pres_mod[6] / 3.99480000e+01 + rho / 3.99480000e+01) * fwd_rxn_rates[24] + pres_mod[6] * ((-mw_avg * fwd_rxn_rates[24] / 3.99480000e+01)));
  jac[144] += -1.0 * ((-mw_avg * pres_mod[7] / 3.99480000e+01 + rho / 3.99480000e+01) * fwd_rxn_rates[25] + pres_mod[7] * ((-mw_avg * fwd_rxn_rates[25] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[25] / 3.99480000e+01)));
  jac[144] += (2.0 * (-mw_avg * fwd_rxn_rates[26] / 3.99480000e+01));
  jac[144] += -1.0 * ((-mw_avg * fwd_rxn_rates[27] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[27] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[27] / 3.99480000e+01));
  jac[144] += 2.0 * ((-mw_avg * fwd_rxn_rates[32] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[32] / 3.99480000e+01));
  jac[144] += -2.0 * (2.0 * (-mw_avg * fwd_rxn_rates[33] / 3.99480000e+01));
  jac[144] += ((-mw_avg * fwd_rxn_rates[34] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[34] / 3.99480000e+01));
  jac[144] += -1.0 * ((-mw_avg * fwd_rxn_rates[35] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[35] / 3.99480000e+01));
  jac[144] += -1.0 * ((-mw_avg * fwd_rxn_rates[36] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[36] / 3.99480000e+01));
  jac[144] += ((-mw_avg * fwd_rxn_rates[37] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[37] / 3.99480000e+01));
  Pr = (m + 6.5 * conc[4] + 0.6 * conc[12] + 0.5 * conc[8] + 0.2 * conc[5] - 0.35 * conc[10] + 6.7 * conc[7] + 2.7 * conc[1] + 1.8 * conc[11]) * (exp(2.78501566e+01 - 3.2 * logT));
  Fcent = 5.7000e-01 * exp(-T / 1.0000e-30) + 4.3000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[144] += 2.0 * (pres_mod[10] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 3.99480000e+01 + rho / ((m + 6.5 * conc[4] + 0.6 * conc[12] + 0.5 * conc[8] + 0.2 * conc[5] - 0.35 * conc[10] + 6.7 * conc[7] + 2.7 * conc[1] + 1.8 * conc[11]) * 3.99480000e+01)) * fwd_rxn_rates[42] + pres_mod[10] * ((-mw_avg * fwd_rxn_rates[42] / 3.99480000e+01)));
  Pr = (m + 6.5 * conc[4] + 0.6 * conc[12] + 0.5 * conc[8] + 0.2 * conc[5] - 0.35 * conc[10] + 6.7 * conc[7] + 2.7 * conc[1] + 1.8 * conc[11]) * (exp(2.78501521e+01 - 3.20001 * logT));
  Fcent = 5.7000e-01 * exp(-T / 1.0000e-30) + 4.3000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[144] += -2.0 * (pres_mod[11] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 3.99480000e+01 + rho / ((m + 6.5 * conc[4] + 0.6 * conc[12] + 0.5 * conc[8] + 0.2 * conc[5] - 0.35 * conc[10] + 6.7 * conc[7] + 2.7 * conc[1] + 1.8 * conc[11]) * 3.99480000e+01)) * fwd_rxn_rates[43] + pres_mod[11] * (2.0 * (-mw_avg * fwd_rxn_rates[43] / 3.99480000e+01)));
  jac[144] += ((-mw_avg * fwd_rxn_rates[44] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[44] / 3.99480000e+01));
  jac[144] += -1.0 * ((-mw_avg * fwd_rxn_rates[45] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[45] / 3.99480000e+01));
  jac[144] += ((-mw_avg * fwd_rxn_rates[48] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[48] / 3.99480000e+01));
  jac[144] += -1.0 * ((-mw_avg * fwd_rxn_rates[49] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[49] / 3.99480000e+01));
  jac[144] += -1.0 * ((-mw_avg * fwd_rxn_rates[50] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[50] / 3.99480000e+01));
  jac[144] += ((-mw_avg * fwd_rxn_rates[51] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[51] / 3.99480000e+01));
  jac[144] += -1.0 * ((-mw_avg * fwd_rxn_rates[52] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[52] / 3.99480000e+01));
  jac[144] += ((-mw_avg * fwd_rxn_rates[53] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[53] / 3.99480000e+01));
  jac[144] += sp_rates[3] * mw_avg / 3.99480000e+01;
  jac[144] *= 1.70073700e+01 / rho;

  //partial of omega_OH wrt Y_HE;
  jac[158] = ((-mw_avg * fwd_rxn_rates[0] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[0] / 4.00260000e+00));
  jac[158] += -1.0 * ((-mw_avg * fwd_rxn_rates[1] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[1] / 4.00260000e+00));
  jac[158] += ((-mw_avg * fwd_rxn_rates[2] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[2] / 4.00260000e+00));
  jac[158] += -1.0 * ((-mw_avg * fwd_rxn_rates[3] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[3] / 4.00260000e+00));
  jac[158] += ((-mw_avg * fwd_rxn_rates[4] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[4] / 4.00260000e+00));
  jac[158] += -1.0 * ((-mw_avg * fwd_rxn_rates[5] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[5] / 4.00260000e+00));
  jac[158] += -1.0 * ((-mw_avg * fwd_rxn_rates[6] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[6] / 4.00260000e+00));
  jac[158] += ((-mw_avg * fwd_rxn_rates[7] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[7] / 4.00260000e+00));
  jac[158] += -2.0 * (2.0 * (-mw_avg * fwd_rxn_rates[8] / 4.00260000e+00));
  jac[158] += 2.0 * ((-mw_avg * fwd_rxn_rates[9] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[9] / 4.00260000e+00));
  jac[158] += ((-mw_avg * pres_mod[4] / 4.00260000e+00 + 0.75 * rho / 4.00260000e+00) * fwd_rxn_rates[22] + pres_mod[4] * ((-mw_avg * fwd_rxn_rates[22] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[22] / 4.00260000e+00)));
  jac[158] += -1.0 * ((-mw_avg * pres_mod[5] / 4.00260000e+00 + 0.75 * rho / 4.00260000e+00) * fwd_rxn_rates[23] + pres_mod[5] * ((-mw_avg * fwd_rxn_rates[23] / 4.00260000e+00)));
  jac[158] += ((-mw_avg * pres_mod[6] / 4.00260000e+00 + 1.1 * rho / 4.00260000e+00) * fwd_rxn_rates[24] + pres_mod[6] * ((-mw_avg * fwd_rxn_rates[24] / 4.00260000e+00)));
  jac[158] += -1.0 * ((-mw_avg * pres_mod[7] / 4.00260000e+00 + 1.1 * rho / 4.00260000e+00) * fwd_rxn_rates[25] + pres_mod[7] * ((-mw_avg * fwd_rxn_rates[25] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[25] / 4.00260000e+00)));
  jac[158] += (2.0 * (-mw_avg * fwd_rxn_rates[26] / 4.00260000e+00));
  jac[158] += -1.0 * ((-mw_avg * fwd_rxn_rates[27] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[27] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[27] / 4.00260000e+00));
  jac[158] += 2.0 * ((-mw_avg * fwd_rxn_rates[32] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[32] / 4.00260000e+00));
  jac[158] += -2.0 * (2.0 * (-mw_avg * fwd_rxn_rates[33] / 4.00260000e+00));
  jac[158] += ((-mw_avg * fwd_rxn_rates[34] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[34] / 4.00260000e+00));
  jac[158] += -1.0 * ((-mw_avg * fwd_rxn_rates[35] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[35] / 4.00260000e+00));
  jac[158] += -1.0 * ((-mw_avg * fwd_rxn_rates[36] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[36] / 4.00260000e+00));
  jac[158] += ((-mw_avg * fwd_rxn_rates[37] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[37] / 4.00260000e+00));
  Pr = (m + 6.5 * conc[4] + 0.6 * conc[12] + 0.5 * conc[8] + 0.2 * conc[5] - 0.35 * conc[10] + 6.7 * conc[7] + 2.7 * conc[1] + 1.8 * conc[11]) * (exp(2.78501566e+01 - 3.2 * logT));
  Fcent = 5.7000e-01 * exp(-T / 1.0000e-30) + 4.3000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[158] += 2.0 * (pres_mod[10] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 4.00260000e+00 + rho * 0.65 / ((m + 6.5 * conc[4] + 0.6 * conc[12] + 0.5 * conc[8] + 0.2 * conc[5] - 0.35 * conc[10] + 6.7 * conc[7] + 2.7 * conc[1] + 1.8 * conc[11]) * 4.00260000e+00)) * fwd_rxn_rates[42] + pres_mod[10] * ((-mw_avg * fwd_rxn_rates[42] / 4.00260000e+00)));
  Pr = (m + 6.5 * conc[4] + 0.6 * conc[12] + 0.5 * conc[8] + 0.2 * conc[5] - 0.35 * conc[10] + 6.7 * conc[7] + 2.7 * conc[1] + 1.8 * conc[11]) * (exp(2.78501521e+01 - 3.20001 * logT));
  Fcent = 5.7000e-01 * exp(-T / 1.0000e-30) + 4.3000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[158] += -2.0 * (pres_mod[11] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 4.00260000e+00 + rho * 0.65 / ((m + 6.5 * conc[4] + 0.6 * conc[12] + 0.5 * conc[8] + 0.2 * conc[5] - 0.35 * conc[10] + 6.7 * conc[7] + 2.7 * conc[1] + 1.8 * conc[11]) * 4.00260000e+00)) * fwd_rxn_rates[43] + pres_mod[11] * (2.0 * (-mw_avg * fwd_rxn_rates[43] / 4.00260000e+00)));
  jac[158] += ((-mw_avg * fwd_rxn_rates[44] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[44] / 4.00260000e+00));
  jac[158] += -1.0 * ((-mw_avg * fwd_rxn_rates[45] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[45] / 4.00260000e+00));
  jac[158] += ((-mw_avg * fwd_rxn_rates[48] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[48] / 4.00260000e+00));
  jac[158] += -1.0 * ((-mw_avg * fwd_rxn_rates[49] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[49] / 4.00260000e+00));
  jac[158] += -1.0 * ((-mw_avg * fwd_rxn_rates[50] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[50] / 4.00260000e+00));
  jac[158] += ((-mw_avg * fwd_rxn_rates[51] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[51] / 4.00260000e+00));
  jac[158] += -1.0 * ((-mw_avg * fwd_rxn_rates[52] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[52] / 4.00260000e+00));
  jac[158] += ((-mw_avg * fwd_rxn_rates[53] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[53] / 4.00260000e+00));
  jac[158] += sp_rates[3] * mw_avg / 4.00260000e+00;
  jac[158] *= 1.70073700e+01 / rho;

  //partial of omega_OH wrt Y_CO;
  jac[172] = ((-mw_avg * fwd_rxn_rates[0] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[0] / 2.80105500e+01));
  jac[172] += -1.0 * ((-mw_avg * fwd_rxn_rates[1] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[1] / 2.80105500e+01));
  jac[172] += ((-mw_avg * fwd_rxn_rates[2] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[2] / 2.80105500e+01));
  jac[172] += -1.0 * ((-mw_avg * fwd_rxn_rates[3] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[3] / 2.80105500e+01));
  jac[172] += ((-mw_avg * fwd_rxn_rates[4] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[4] / 2.80105500e+01));
  jac[172] += -1.0 * ((-mw_avg * fwd_rxn_rates[5] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[5] / 2.80105500e+01));
  jac[172] += -1.0 * ((-mw_avg * fwd_rxn_rates[6] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[6] / 2.80105500e+01));
  jac[172] += ((-mw_avg * fwd_rxn_rates[7] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[7] / 2.80105500e+01));
  jac[172] += -2.0 * (2.0 * (-mw_avg * fwd_rxn_rates[8] / 2.80105500e+01));
  jac[172] += 2.0 * ((-mw_avg * fwd_rxn_rates[9] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[9] / 2.80105500e+01));
  jac[172] += ((-mw_avg * pres_mod[4] / 2.80105500e+01 + 1.9 * rho / 2.80105500e+01) * fwd_rxn_rates[22] + pres_mod[4] * ((-mw_avg * fwd_rxn_rates[22] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[22] / 2.80105500e+01)));
  jac[172] += -1.0 * ((-mw_avg * pres_mod[5] / 2.80105500e+01 + 1.9 * rho / 2.80105500e+01) * fwd_rxn_rates[23] + pres_mod[5] * ((-mw_avg * fwd_rxn_rates[23] / 2.80105500e+01)));
  jac[172] += ((-mw_avg * pres_mod[6] / 2.80105500e+01 + 1.9 * rho / 2.80105500e+01) * fwd_rxn_rates[24] + pres_mod[6] * ((-mw_avg * fwd_rxn_rates[24] / 2.80105500e+01)));
  jac[172] += -1.0 * ((-mw_avg * pres_mod[7] / 2.80105500e+01 + 1.9 * rho / 2.80105500e+01) * fwd_rxn_rates[25] + pres_mod[7] * ((-mw_avg * fwd_rxn_rates[25] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[25] / 2.80105500e+01)));
  jac[172] += (2.0 * (-mw_avg * fwd_rxn_rates[26] / 2.80105500e+01));
  jac[172] += -1.0 * ((-mw_avg * fwd_rxn_rates[27] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[27] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[27] / 2.80105500e+01));
  jac[172] += 2.0 * ((-mw_avg * fwd_rxn_rates[32] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[32] / 2.80105500e+01));
  jac[172] += -2.0 * (2.0 * (-mw_avg * fwd_rxn_rates[33] / 2.80105500e+01));
  jac[172] += ((-mw_avg * fwd_rxn_rates[34] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[34] / 2.80105500e+01));
  jac[172] += -1.0 * ((-mw_avg * fwd_rxn_rates[35] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[35] / 2.80105500e+01));
  jac[172] += -1.0 * ((-mw_avg * fwd_rxn_rates[36] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[36] / 2.80105500e+01));
  jac[172] += ((-mw_avg * fwd_rxn_rates[37] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[37] / 2.80105500e+01));
  Pr = (m + 6.5 * conc[4] + 0.6 * conc[12] + 0.5 * conc[8] + 0.2 * conc[5] - 0.35 * conc[10] + 6.7 * conc[7] + 2.7 * conc[1] + 1.8 * conc[11]) * (exp(2.78501566e+01 - 3.2 * logT));
  Fcent = 5.7000e-01 * exp(-T / 1.0000e-30) + 4.3000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[172] += 2.0 * (pres_mod[10] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 2.80105500e+01 + rho * 2.8 / ((m + 6.5 * conc[4] + 0.6 * conc[12] + 0.5 * conc[8] + 0.2 * conc[5] - 0.35 * conc[10] + 6.7 * conc[7] + 2.7 * conc[1] + 1.8 * conc[11]) * 2.80105500e+01)) * fwd_rxn_rates[42] + pres_mod[10] * ((-mw_avg * fwd_rxn_rates[42] / 2.80105500e+01)));
  Pr = (m + 6.5 * conc[4] + 0.6 * conc[12] + 0.5 * conc[8] + 0.2 * conc[5] - 0.35 * conc[10] + 6.7 * conc[7] + 2.7 * conc[1] + 1.8 * conc[11]) * (exp(2.78501521e+01 - 3.20001 * logT));
  Fcent = 5.7000e-01 * exp(-T / 1.0000e-30) + 4.3000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[172] += -2.0 * (pres_mod[11] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 2.80105500e+01 + rho * 2.8 / ((m + 6.5 * conc[4] + 0.6 * conc[12] + 0.5 * conc[8] + 0.2 * conc[5] - 0.35 * conc[10] + 6.7 * conc[7] + 2.7 * conc[1] + 1.8 * conc[11]) * 2.80105500e+01)) * fwd_rxn_rates[43] + pres_mod[11] * (2.0 * (-mw_avg * fwd_rxn_rates[43] / 2.80105500e+01)));
  jac[172] += ((-mw_avg * fwd_rxn_rates[44] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[44] / 2.80105500e+01));
  jac[172] += -1.0 * ((-mw_avg * fwd_rxn_rates[45] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[45] / 2.80105500e+01));
  jac[172] += ((-mw_avg * fwd_rxn_rates[48] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[48] / 2.80105500e+01));
  jac[172] += -1.0 * ((-mw_avg * fwd_rxn_rates[49] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[49] / 2.80105500e+01));
  jac[172] += -1.0 * ((-mw_avg * fwd_rxn_rates[50] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[50] / 2.80105500e+01));
  jac[172] += ((-mw_avg * fwd_rxn_rates[51] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[51] / 2.80105500e+01));
  jac[172] += -1.0 * ((-mw_avg * fwd_rxn_rates[52] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[52] / 2.80105500e+01));
  jac[172] += ((-mw_avg * fwd_rxn_rates[53] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[53] / 2.80105500e+01));
  jac[172] += sp_rates[3] * mw_avg / 2.80105500e+01;
  jac[172] *= 1.70073700e+01 / rho;

  //partial of omega_OH wrt Y_CO2;
  jac[186] = ((-mw_avg * fwd_rxn_rates[0] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[0] / 4.40099500e+01));
  jac[186] += -1.0 * ((-mw_avg * fwd_rxn_rates[1] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[1] / 4.40099500e+01));
  jac[186] += ((-mw_avg * fwd_rxn_rates[2] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[2] / 4.40099500e+01));
  jac[186] += -1.0 * ((-mw_avg * fwd_rxn_rates[3] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[3] / 4.40099500e+01));
  jac[186] += ((-mw_avg * fwd_rxn_rates[4] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[4] / 4.40099500e+01));
  jac[186] += -1.0 * ((-mw_avg * fwd_rxn_rates[5] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[5] / 4.40099500e+01));
  jac[186] += -1.0 * ((-mw_avg * fwd_rxn_rates[6] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[6] / 4.40099500e+01));
  jac[186] += ((-mw_avg * fwd_rxn_rates[7] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[7] / 4.40099500e+01));
  jac[186] += -2.0 * (2.0 * (-mw_avg * fwd_rxn_rates[8] / 4.40099500e+01));
  jac[186] += 2.0 * ((-mw_avg * fwd_rxn_rates[9] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[9] / 4.40099500e+01));
  jac[186] += ((-mw_avg * pres_mod[4] / 4.40099500e+01 + 3.8 * rho / 4.40099500e+01) * fwd_rxn_rates[22] + pres_mod[4] * ((-mw_avg * fwd_rxn_rates[22] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[22] / 4.40099500e+01)));
  jac[186] += -1.0 * ((-mw_avg * pres_mod[5] / 4.40099500e+01 + 3.8 * rho / 4.40099500e+01) * fwd_rxn_rates[23] + pres_mod[5] * ((-mw_avg * fwd_rxn_rates[23] / 4.40099500e+01)));
  jac[186] += ((-mw_avg * pres_mod[6] / 4.40099500e+01 + 3.8 * rho / 4.40099500e+01) * fwd_rxn_rates[24] + pres_mod[6] * ((-mw_avg * fwd_rxn_rates[24] / 4.40099500e+01)));
  jac[186] += -1.0 * ((-mw_avg * pres_mod[7] / 4.40099500e+01 + 3.8 * rho / 4.40099500e+01) * fwd_rxn_rates[25] + pres_mod[7] * ((-mw_avg * fwd_rxn_rates[25] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[25] / 4.40099500e+01)));
  jac[186] += (2.0 * (-mw_avg * fwd_rxn_rates[26] / 4.40099500e+01));
  jac[186] += -1.0 * ((-mw_avg * fwd_rxn_rates[27] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[27] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[27] / 4.40099500e+01));
  jac[186] += 2.0 * ((-mw_avg * fwd_rxn_rates[32] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[32] / 4.40099500e+01));
  jac[186] += -2.0 * (2.0 * (-mw_avg * fwd_rxn_rates[33] / 4.40099500e+01));
  jac[186] += ((-mw_avg * fwd_rxn_rates[34] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[34] / 4.40099500e+01));
  jac[186] += -1.0 * ((-mw_avg * fwd_rxn_rates[35] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[35] / 4.40099500e+01));
  jac[186] += -1.0 * ((-mw_avg * fwd_rxn_rates[36] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[36] / 4.40099500e+01));
  jac[186] += ((-mw_avg * fwd_rxn_rates[37] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[37] / 4.40099500e+01));
  Pr = (m + 6.5 * conc[4] + 0.6 * conc[12] + 0.5 * conc[8] + 0.2 * conc[5] - 0.35 * conc[10] + 6.7 * conc[7] + 2.7 * conc[1] + 1.8 * conc[11]) * (exp(2.78501566e+01 - 3.2 * logT));
  Fcent = 5.7000e-01 * exp(-T / 1.0000e-30) + 4.3000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[186] += 2.0 * (pres_mod[10] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 4.40099500e+01 + rho * 1.6 / ((m + 6.5 * conc[4] + 0.6 * conc[12] + 0.5 * conc[8] + 0.2 * conc[5] - 0.35 * conc[10] + 6.7 * conc[7] + 2.7 * conc[1] + 1.8 * conc[11]) * 4.40099500e+01)) * fwd_rxn_rates[42] + pres_mod[10] * ((-mw_avg * fwd_rxn_rates[42] / 4.40099500e+01)));
  Pr = (m + 6.5 * conc[4] + 0.6 * conc[12] + 0.5 * conc[8] + 0.2 * conc[5] - 0.35 * conc[10] + 6.7 * conc[7] + 2.7 * conc[1] + 1.8 * conc[11]) * (exp(2.78501521e+01 - 3.20001 * logT));
  Fcent = 5.7000e-01 * exp(-T / 1.0000e-30) + 4.3000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[186] += -2.0 * (pres_mod[11] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 4.40099500e+01 + rho * 1.6 / ((m + 6.5 * conc[4] + 0.6 * conc[12] + 0.5 * conc[8] + 0.2 * conc[5] - 0.35 * conc[10] + 6.7 * conc[7] + 2.7 * conc[1] + 1.8 * conc[11]) * 4.40099500e+01)) * fwd_rxn_rates[43] + pres_mod[11] * (2.0 * (-mw_avg * fwd_rxn_rates[43] / 4.40099500e+01)));
  jac[186] += ((-mw_avg * fwd_rxn_rates[44] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[44] / 4.40099500e+01));
  jac[186] += -1.0 * ((-mw_avg * fwd_rxn_rates[45] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[45] / 4.40099500e+01));
  jac[186] += ((-mw_avg * fwd_rxn_rates[48] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[48] / 4.40099500e+01));
  jac[186] += -1.0 * ((-mw_avg * fwd_rxn_rates[49] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[49] / 4.40099500e+01));
  jac[186] += -1.0 * ((-mw_avg * fwd_rxn_rates[50] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[50] / 4.40099500e+01));
  jac[186] += ((-mw_avg * fwd_rxn_rates[51] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[51] / 4.40099500e+01));
  jac[186] += -1.0 * ((-mw_avg * fwd_rxn_rates[52] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[52] / 4.40099500e+01));
  jac[186] += ((-mw_avg * fwd_rxn_rates[53] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[53] / 4.40099500e+01));
  jac[186] += sp_rates[3] * mw_avg / 4.40099500e+01;
  jac[186] *= 1.70073700e+01 / rho;

  //partial of omega_H2O wrt T;
  jac[5] = ((1.0 / T) * (fwd_rxn_rates[6] * (1.51000000e+00 + (1.72603316e+03 / T) + 1.0 - 2.0)));
  jac[5] += -1.0 * ((1.0 / T) * (fwd_rxn_rates[7] * (1.18290000e+00 + (9.55507805e+03 / T) + 1.0 - 2.0)));
  jac[5] += ((1.0 / T) * (fwd_rxn_rates[8] * (2.42000000e+00 + (-9.71208165e+02 / T) + 1.0 - 2.0)));
  jac[5] += -1.0 * ((1.0 / T) * (fwd_rxn_rates[9] * (2.14640000e+00 + (7.53063740e+03 / T) + 1.0 - 2.0)));
  jac[5] += -1.0 * ((-pres_mod[6] * fwd_rxn_rates[24] / T) + (pres_mod[6] / T) * (fwd_rxn_rates[24] * (-3.32200000e+00 + (6.07835411e+04 / T) + 1.0 - 1.0)));
  jac[5] += ((-pres_mod[7] * fwd_rxn_rates[25] / T) + (pres_mod[7] / T) * (fwd_rxn_rates[25] * (-3.01830000e+00 + (4.50771425e+02 / T) + 1.0 - 2.0)));
  jac[5] += -1.0 * ((1.0 / T) * (fwd_rxn_rates[26] * (-2.44000000e+00 + (6.04765789e+04 / T) + 1.0 - 2.0)));
  jac[5] += ((1.0 / T) * (fwd_rxn_rates[27] * (-2.13630000e+00 + (1.43809259e+02 / T) + 1.0 - 3.0)));
  jac[5] += ((1.0 / T) * (fwd_rxn_rates[36] * ((-2.50098683e+02 / T) + 1.0 - 2.0)));
  jac[5] += -1.0 * ((1.0 / T) * (fwd_rxn_rates[37] * (1.85740000e-01 + (3.48643603e+04 / T) + 1.0 - 2.0)));
  jac[5] += ((1.0 / T) * (fwd_rxn_rates[44] * ((1.99777016e+03 / T) + 1.0 - 2.0)));
  jac[5] += -1.0 * ((1.0 / T) * (fwd_rxn_rates[45] * (1.28430000e+00 + (3.59925720e+04 / T) + 1.0 - 2.0)));
  jac[5] += ((1.0 / T) * (fwd_rxn_rates[50] * ((1.60022900e+02 / T) + 1.0 - 2.0)));
  jac[5] += -1.0 * ((1.0 / T) * (fwd_rxn_rates[51] * (4.20210000e-01 + (1.59826645e+04 / T) + 1.0 - 2.0)));
  jac[5] += ((1.0 / T) * (fwd_rxn_rates[52] * ((3.65838516e+03 / T) + 1.0 - 2.0)));
  jac[5] += -1.0 * ((1.0 / T) * (fwd_rxn_rates[53] * (4.20210000e-01 + (1.94810268e+04 / T) + 1.0 - 2.0)));
  jac[5] *= 1.80153400e+01 / rho;

  //partial of omega_H2O wrt Y_H;
  jac[19] = ((-mw_avg * fwd_rxn_rates[6] / 1.00797000e+00) + (-mw_avg * fwd_rxn_rates[6] / 1.00797000e+00));
  jac[19] += -1.0 * ((-mw_avg * fwd_rxn_rates[7] / 1.00797000e+00 + exp(2.34232839e+01 + 1.1829 * logT - (9.55507805e+03 / T)) * (rho / 1.00797000e+00) * conc[4]) + (-mw_avg * fwd_rxn_rates[7] / 1.00797000e+00));
  jac[19] += (2.0 * (-mw_avg * fwd_rxn_rates[8] / 1.00797000e+00));
  jac[19] += -1.0 * ((-mw_avg * fwd_rxn_rates[9] / 1.00797000e+00) + (-mw_avg * fwd_rxn_rates[9] / 1.00797000e+00));
  jac[19] += -1.0 * ((-mw_avg * pres_mod[6] / 1.00797000e+00 + rho / 1.00797000e+00) * fwd_rxn_rates[24] + pres_mod[6] * ((-mw_avg * fwd_rxn_rates[24] / 1.00797000e+00)));
  jac[19] += ((-mw_avg * pres_mod[7] / 1.00797000e+00 + rho / 1.00797000e+00) * fwd_rxn_rates[25] + pres_mod[7] * ((-mw_avg * fwd_rxn_rates[25] / 1.00797000e+00 + exp(5.85301272e+01 - 3.0183 * logT - (4.50771425e+02 / T)) * (rho / 1.00797000e+00) * conc[3]) + (-mw_avg * fwd_rxn_rates[25] / 1.00797000e+00)));
  jac[19] += -1.0 * (2.0 * (-mw_avg * fwd_rxn_rates[26] / 1.00797000e+00));
  jac[19] += ((-mw_avg * fwd_rxn_rates[27] / 1.00797000e+00 + exp(5.44311720e+01 - 2.1363 * logT - (1.43809259e+02 / T)) * (rho / 1.00797000e+00) * conc[3] * conc[4]) + (-mw_avg * fwd_rxn_rates[27] / 1.00797000e+00) + (-mw_avg * fwd_rxn_rates[27] / 1.00797000e+00));
  jac[19] += ((-mw_avg * fwd_rxn_rates[36] / 1.00797000e+00) + (-mw_avg * fwd_rxn_rates[36] / 1.00797000e+00));
  jac[19] += -1.0 * ((-mw_avg * fwd_rxn_rates[37] / 1.00797000e+00) + (-mw_avg * fwd_rxn_rates[37] / 1.00797000e+00));
  jac[19] += ((-mw_avg * fwd_rxn_rates[44] / 1.00797000e+00 + exp(3.08132330e+01 - (1.99777016e+03 / T)) * (rho / 1.00797000e+00) * conc[7]) + (-mw_avg * fwd_rxn_rates[44] / 1.00797000e+00));
  jac[19] += -1.0 * ((-mw_avg * fwd_rxn_rates[45] / 1.00797000e+00) + (-mw_avg * fwd_rxn_rates[45] / 1.00797000e+00));
  jac[19] += ((-mw_avg * fwd_rxn_rates[50] / 1.00797000e+00) + (-mw_avg * fwd_rxn_rates[50] / 1.00797000e+00));
  jac[19] += -1.0 * ((-mw_avg * fwd_rxn_rates[51] / 1.00797000e+00) + (-mw_avg * fwd_rxn_rates[51] / 1.00797000e+00));
  jac[19] += ((-mw_avg * fwd_rxn_rates[52] / 1.00797000e+00) + (-mw_avg * fwd_rxn_rates[52] / 1.00797000e+00));
  jac[19] += -1.0 * ((-mw_avg * fwd_rxn_rates[53] / 1.00797000e+00) + (-mw_avg * fwd_rxn_rates[53] / 1.00797000e+00));
  jac[19] += sp_rates[4] * mw_avg / 1.00797000e+00;
  jac[19] *= 1.80153400e+01 / rho;

  //partial of omega_H2O wrt Y_H2;
  jac[33] = ((-mw_avg * fwd_rxn_rates[6] / 2.01594000e+00 + exp(1.91907890e+01 + 1.51 * logT - (1.72603316e+03 / T)) * (rho / 2.01594000e+00) * conc[3]) + (-mw_avg * fwd_rxn_rates[6] / 2.01594000e+00));
  jac[33] += -1.0 * ((-mw_avg * fwd_rxn_rates[7] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[7] / 2.01594000e+00));
  jac[33] += (2.0 * (-mw_avg * fwd_rxn_rates[8] / 2.01594000e+00));
  jac[33] += -1.0 * ((-mw_avg * fwd_rxn_rates[9] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[9] / 2.01594000e+00));
  jac[33] += -1.0 * ((-mw_avg * pres_mod[6] / 2.01594000e+00 + 3.0 * rho / 2.01594000e+00) * fwd_rxn_rates[24] + pres_mod[6] * ((-mw_avg * fwd_rxn_rates[24] / 2.01594000e+00)));
  jac[33] += ((-mw_avg * pres_mod[7] / 2.01594000e+00 + 3.0 * rho / 2.01594000e+00) * fwd_rxn_rates[25] + pres_mod[7] * ((-mw_avg * fwd_rxn_rates[25] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[25] / 2.01594000e+00)));
  jac[33] += -1.0 * (2.0 * (-mw_avg * fwd_rxn_rates[26] / 2.01594000e+00));
  jac[33] += ((-mw_avg * fwd_rxn_rates[27] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[27] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[27] / 2.01594000e+00));
  jac[33] += ((-mw_avg * fwd_rxn_rates[36] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[36] / 2.01594000e+00));
  jac[33] += -1.0 * ((-mw_avg * fwd_rxn_rates[37] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[37] / 2.01594000e+00));
  jac[33] += ((-mw_avg * fwd_rxn_rates[44] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[44] / 2.01594000e+00));
  jac[33] += -1.0 * ((-mw_avg * fwd_rxn_rates[45] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[45] / 2.01594000e+00));
  jac[33] += ((-mw_avg * fwd_rxn_rates[50] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[50] / 2.01594000e+00));
  jac[33] += -1.0 * ((-mw_avg * fwd_rxn_rates[51] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[51] / 2.01594000e+00));
  jac[33] += ((-mw_avg * fwd_rxn_rates[52] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[52] / 2.01594000e+00));
  jac[33] += -1.0 * ((-mw_avg * fwd_rxn_rates[53] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[53] / 2.01594000e+00));
  jac[33] += sp_rates[4] * mw_avg / 2.01594000e+00;
  jac[33] *= 1.80153400e+01 / rho;

  //partial of omega_H2O wrt Y_O;
  jac[47] = ((-mw_avg * fwd_rxn_rates[6] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[6] / 1.59994000e+01));
  jac[47] += -1.0 * ((-mw_avg * fwd_rxn_rates[7] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[7] / 1.59994000e+01));
  jac[47] += (2.0 * (-mw_avg * fwd_rxn_rates[8] / 1.59994000e+01));
  jac[47] += -1.0 * ((-mw_avg * fwd_rxn_rates[9] / 1.59994000e+01 + exp(1.50329128e+01 + 2.1464 * logT - (7.53063740e+03 / T)) * (rho / 1.59994000e+01) * conc[4]) + (-mw_avg * fwd_rxn_rates[9] / 1.59994000e+01));
  jac[47] += -1.0 * ((-mw_avg * pres_mod[6] / 1.59994000e+01 + rho / 1.59994000e+01) * fwd_rxn_rates[24] + pres_mod[6] * ((-mw_avg * fwd_rxn_rates[24] / 1.59994000e+01)));
  jac[47] += ((-mw_avg * pres_mod[7] / 1.59994000e+01 + rho / 1.59994000e+01) * fwd_rxn_rates[25] + pres_mod[7] * ((-mw_avg * fwd_rxn_rates[25] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[25] / 1.59994000e+01)));
  jac[47] += -1.0 * (2.0 * (-mw_avg * fwd_rxn_rates[26] / 1.59994000e+01));
  jac[47] += ((-mw_avg * fwd_rxn_rates[27] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[27] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[27] / 1.59994000e+01));
  jac[47] += ((-mw_avg * fwd_rxn_rates[36] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[36] / 1.59994000e+01));
  jac[47] += -1.0 * ((-mw_avg * fwd_rxn_rates[37] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[37] / 1.59994000e+01));
  jac[47] += ((-mw_avg * fwd_rxn_rates[44] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[44] / 1.59994000e+01));
  jac[47] += -1.0 * ((-mw_avg * fwd_rxn_rates[45] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[45] / 1.59994000e+01));
  jac[47] += ((-mw_avg * fwd_rxn_rates[50] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[50] / 1.59994000e+01));
  jac[47] += -1.0 * ((-mw_avg * fwd_rxn_rates[51] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[51] / 1.59994000e+01));
  jac[47] += ((-mw_avg * fwd_rxn_rates[52] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[52] / 1.59994000e+01));
  jac[47] += -1.0 * ((-mw_avg * fwd_rxn_rates[53] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[53] / 1.59994000e+01));
  jac[47] += sp_rates[4] * mw_avg / 1.59994000e+01;
  jac[47] *= 1.80153400e+01 / rho;

  //partial of omega_H2O wrt Y_OH;
  jac[61] = ((-mw_avg * fwd_rxn_rates[6] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[6] / 1.70073700e+01 + exp(1.91907890e+01 + 1.51 * logT - (1.72603316e+03 / T)) * (rho / 1.70073700e+01) * conc[1]));
  jac[61] += -1.0 * ((-mw_avg * fwd_rxn_rates[7] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[7] / 1.70073700e+01));
  jac[61] += (2.0 * (-mw_avg * fwd_rxn_rates[8] / 1.70073700e+01 + exp(1.04163112e+01 + 2.42 * logT - (-9.71208165e+02 / T)) * (rho / 1.70073700e+01) * conc[3]));
  jac[61] += -1.0 * ((-mw_avg * fwd_rxn_rates[9] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[9] / 1.70073700e+01));
  jac[61] += -1.0 * ((-mw_avg * pres_mod[6] / 1.70073700e+01 + rho / 1.70073700e+01) * fwd_rxn_rates[24] + pres_mod[6] * ((-mw_avg * fwd_rxn_rates[24] / 1.70073700e+01)));
  jac[61] += ((-mw_avg * pres_mod[7] / 1.70073700e+01 + rho / 1.70073700e+01) * fwd_rxn_rates[25] + pres_mod[7] * ((-mw_avg * fwd_rxn_rates[25] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[25] / 1.70073700e+01 + exp(5.85301272e+01 - 3.0183 * logT - (4.50771425e+02 / T)) * (rho / 1.70073700e+01) * conc[0])));
  jac[61] += -1.0 * (2.0 * (-mw_avg * fwd_rxn_rates[26] / 1.70073700e+01));
  jac[61] += ((-mw_avg * fwd_rxn_rates[27] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[27] / 1.70073700e+01 + exp(5.44311720e+01 - 2.1363 * logT - (1.43809259e+02 / T)) * (rho / 1.70073700e+01) * conc[0] * conc[4]) + (-mw_avg * fwd_rxn_rates[27] / 1.70073700e+01));
  jac[61] += ((-mw_avg * fwd_rxn_rates[36] / 1.70073700e+01 + exp(3.09948627e+01 - (-2.50098683e+02 / T)) * (rho / 1.70073700e+01) * conc[6]) + (-mw_avg * fwd_rxn_rates[36] / 1.70073700e+01));
  jac[61] += -1.0 * ((-mw_avg * fwd_rxn_rates[37] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[37] / 1.70073700e+01));
  jac[61] += ((-mw_avg * fwd_rxn_rates[44] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[44] / 1.70073700e+01));
  jac[61] += -1.0 * ((-mw_avg * fwd_rxn_rates[45] / 1.70073700e+01 + exp(1.88701627e+01 + 1.2843 * logT - (3.59925720e+04 / T)) * (rho / 1.70073700e+01) * conc[4]) + (-mw_avg * fwd_rxn_rates[45] / 1.70073700e+01));
  jac[61] += ((-mw_avg * fwd_rxn_rates[50] / 1.70073700e+01 + exp(2.81849062e+01 - (1.60022900e+02 / T)) * (rho / 1.70073700e+01) * conc[7]) + (-mw_avg * fwd_rxn_rates[50] / 1.70073700e+01));
  jac[61] += -1.0 * ((-mw_avg * fwd_rxn_rates[51] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[51] / 1.70073700e+01));
  jac[61] += ((-mw_avg * fwd_rxn_rates[52] / 1.70073700e+01 + exp(3.19604378e+01 - (3.65838516e+03 / T)) * (rho / 1.70073700e+01) * conc[7]) + (-mw_avg * fwd_rxn_rates[52] / 1.70073700e+01));
  jac[61] += -1.0 * ((-mw_avg * fwd_rxn_rates[53] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[53] / 1.70073700e+01));
  jac[61] += sp_rates[4] * mw_avg / 1.70073700e+01;
  jac[61] *= 1.80153400e+01 / rho;

  //partial of omega_H2O wrt Y_H2O;
  jac[75] = ((-mw_avg * fwd_rxn_rates[6] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[6] / 1.80153400e+01));
  jac[75] += -1.0 * ((-mw_avg * fwd_rxn_rates[7] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[7] / 1.80153400e+01 + exp(2.34232839e+01 + 1.1829 * logT - (9.55507805e+03 / T)) * (rho / 1.80153400e+01) * conc[0]));
  jac[75] += (2.0 * (-mw_avg * fwd_rxn_rates[8] / 1.80153400e+01));
  jac[75] += -1.0 * ((-mw_avg * fwd_rxn_rates[9] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[9] / 1.80153400e+01 + exp(1.50329128e+01 + 2.1464 * logT - (7.53063740e+03 / T)) * (rho / 1.80153400e+01) * conc[2]));
  jac[75] += -1.0 * ((-mw_avg * pres_mod[6] / 1.80153400e+01) * fwd_rxn_rates[24] + pres_mod[6] * ((-mw_avg * fwd_rxn_rates[24] / 1.80153400e+01 + exp(6.39721672e+01 - 3.322 * logT - (6.07835411e+04 / T)) * (rho / 1.80153400e+01))));
  jac[75] += ((-mw_avg * pres_mod[7] / 1.80153400e+01) * fwd_rxn_rates[25] + pres_mod[7] * ((-mw_avg * fwd_rxn_rates[25] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[25] / 1.80153400e+01)));
  jac[75] += -1.0 * (2.0 * (-mw_avg * fwd_rxn_rates[26] / 1.80153400e+01 + exp(5.98731945e+01 - 2.44 * logT - (6.04765789e+04 / T)) * (rho / 1.80153400e+01) * conc[4]));
  jac[75] += ((-mw_avg * fwd_rxn_rates[27] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[27] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[27] / 1.80153400e+01 + exp(5.44311720e+01 - 2.1363 * logT - (1.43809259e+02 / T)) * (rho / 1.80153400e+01) * conc[0] * conc[3]));
  jac[75] += ((-mw_avg * fwd_rxn_rates[36] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[36] / 1.80153400e+01));
  jac[75] += -1.0 * ((-mw_avg * fwd_rxn_rates[37] / 1.80153400e+01 + exp(3.21899484e+01 + 0.18574 * logT - (3.48643603e+04 / T)) * (rho / 1.80153400e+01) * conc[5]) + (-mw_avg * fwd_rxn_rates[37] / 1.80153400e+01));
  jac[75] += ((-mw_avg * fwd_rxn_rates[44] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[44] / 1.80153400e+01));
  jac[75] += -1.0 * ((-mw_avg * fwd_rxn_rates[45] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[45] / 1.80153400e+01 + exp(1.88701627e+01 + 1.2843 * logT - (3.59925720e+04 / T)) * (rho / 1.80153400e+01) * conc[3]));
  jac[75] += ((-mw_avg * fwd_rxn_rates[50] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[50] / 1.80153400e+01));
  jac[75] += -1.0 * ((-mw_avg * fwd_rxn_rates[51] / 1.80153400e+01 + exp(2.56312037e+01 + 0.42021 * logT - (1.59826645e+04 / T)) * (rho / 1.80153400e+01) * conc[6]) + (-mw_avg * fwd_rxn_rates[51] / 1.80153400e+01));
  jac[75] += ((-mw_avg * fwd_rxn_rates[52] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[52] / 1.80153400e+01));
  jac[75] += -1.0 * ((-mw_avg * fwd_rxn_rates[53] / 1.80153400e+01 + exp(2.94067528e+01 + 0.42021 * logT - (1.94810268e+04 / T)) * (rho / 1.80153400e+01) * conc[6]) + (-mw_avg * fwd_rxn_rates[53] / 1.80153400e+01));
  jac[75] += sp_rates[4] * mw_avg / 1.80153400e+01;
  jac[75] *= 1.80153400e+01 / rho;

  //partial of omega_H2O wrt Y_O2;
  jac[89] = ((-mw_avg * fwd_rxn_rates[6] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[6] / 3.19988000e+01));
  jac[89] += -1.0 * ((-mw_avg * fwd_rxn_rates[7] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[7] / 3.19988000e+01));
  jac[89] += (2.0 * (-mw_avg * fwd_rxn_rates[8] / 3.19988000e+01));
  jac[89] += -1.0 * ((-mw_avg * fwd_rxn_rates[9] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[9] / 3.19988000e+01));
  jac[89] += -1.0 * ((-mw_avg * pres_mod[6] / 3.19988000e+01 + 1.5 * rho / 3.19988000e+01) * fwd_rxn_rates[24] + pres_mod[6] * ((-mw_avg * fwd_rxn_rates[24] / 3.19988000e+01)));
  jac[89] += ((-mw_avg * pres_mod[7] / 3.19988000e+01 + 1.5 * rho / 3.19988000e+01) * fwd_rxn_rates[25] + pres_mod[7] * ((-mw_avg * fwd_rxn_rates[25] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[25] / 3.19988000e+01)));
  jac[89] += -1.0 * (2.0 * (-mw_avg * fwd_rxn_rates[26] / 3.19988000e+01));
  jac[89] += ((-mw_avg * fwd_rxn_rates[27] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[27] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[27] / 3.19988000e+01));
  jac[89] += ((-mw_avg * fwd_rxn_rates[36] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[36] / 3.19988000e+01));
  jac[89] += -1.0 * ((-mw_avg * fwd_rxn_rates[37] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[37] / 3.19988000e+01 + exp(3.21899484e+01 + 0.18574 * logT - (3.48643603e+04 / T)) * (rho / 3.19988000e+01) * conc[4]));
  jac[89] += ((-mw_avg * fwd_rxn_rates[44] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[44] / 3.19988000e+01));
  jac[89] += -1.0 * ((-mw_avg * fwd_rxn_rates[45] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[45] / 3.19988000e+01));
  jac[89] += ((-mw_avg * fwd_rxn_rates[50] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[50] / 3.19988000e+01));
  jac[89] += -1.0 * ((-mw_avg * fwd_rxn_rates[51] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[51] / 3.19988000e+01));
  jac[89] += ((-mw_avg * fwd_rxn_rates[52] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[52] / 3.19988000e+01));
  jac[89] += -1.0 * ((-mw_avg * fwd_rxn_rates[53] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[53] / 3.19988000e+01));
  jac[89] += sp_rates[4] * mw_avg / 3.19988000e+01;
  jac[89] *= 1.80153400e+01 / rho;

  //partial of omega_H2O wrt Y_HO2;
  jac[103] = ((-mw_avg * fwd_rxn_rates[6] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[6] / 3.30067700e+01));
  jac[103] += -1.0 * ((-mw_avg * fwd_rxn_rates[7] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[7] / 3.30067700e+01));
  jac[103] += (2.0 * (-mw_avg * fwd_rxn_rates[8] / 3.30067700e+01));
  jac[103] += -1.0 * ((-mw_avg * fwd_rxn_rates[9] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[9] / 3.30067700e+01));
  jac[103] += -1.0 * ((-mw_avg * pres_mod[6] / 3.30067700e+01 + rho / 3.30067700e+01) * fwd_rxn_rates[24] + pres_mod[6] * ((-mw_avg * fwd_rxn_rates[24] / 3.30067700e+01)));
  jac[103] += ((-mw_avg * pres_mod[7] / 3.30067700e+01 + rho / 3.30067700e+01) * fwd_rxn_rates[25] + pres_mod[7] * ((-mw_avg * fwd_rxn_rates[25] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[25] / 3.30067700e+01)));
  jac[103] += -1.0 * (2.0 * (-mw_avg * fwd_rxn_rates[26] / 3.30067700e+01));
  jac[103] += ((-mw_avg * fwd_rxn_rates[27] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[27] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[27] / 3.30067700e+01));
  jac[103] += ((-mw_avg * fwd_rxn_rates[36] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[36] / 3.30067700e+01 + exp(3.09948627e+01 - (-2.50098683e+02 / T)) * (rho / 3.30067700e+01) * conc[3]));
  jac[103] += -1.0 * ((-mw_avg * fwd_rxn_rates[37] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[37] / 3.30067700e+01));
  jac[103] += ((-mw_avg * fwd_rxn_rates[44] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[44] / 3.30067700e+01));
  jac[103] += -1.0 * ((-mw_avg * fwd_rxn_rates[45] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[45] / 3.30067700e+01));
  jac[103] += ((-mw_avg * fwd_rxn_rates[50] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[50] / 3.30067700e+01));
  jac[103] += -1.0 * ((-mw_avg * fwd_rxn_rates[51] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[51] / 3.30067700e+01 + exp(2.56312037e+01 + 0.42021 * logT - (1.59826645e+04 / T)) * (rho / 3.30067700e+01) * conc[4]));
  jac[103] += ((-mw_avg * fwd_rxn_rates[52] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[52] / 3.30067700e+01));
  jac[103] += -1.0 * ((-mw_avg * fwd_rxn_rates[53] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[53] / 3.30067700e+01 + exp(2.94067528e+01 + 0.42021 * logT - (1.94810268e+04 / T)) * (rho / 3.30067700e+01) * conc[4]));
  jac[103] += sp_rates[4] * mw_avg / 3.30067700e+01;
  jac[103] *= 1.80153400e+01 / rho;

  //partial of omega_H2O wrt Y_H2O2;
  jac[117] = ((-mw_avg * fwd_rxn_rates[6] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[6] / 3.40147400e+01));
  jac[117] += -1.0 * ((-mw_avg * fwd_rxn_rates[7] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[7] / 3.40147400e+01));
  jac[117] += (2.0 * (-mw_avg * fwd_rxn_rates[8] / 3.40147400e+01));
  jac[117] += -1.0 * ((-mw_avg * fwd_rxn_rates[9] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[9] / 3.40147400e+01));
  jac[117] += -1.0 * ((-mw_avg * pres_mod[6] / 3.40147400e+01 + rho / 3.40147400e+01) * fwd_rxn_rates[24] + pres_mod[6] * ((-mw_avg * fwd_rxn_rates[24] / 3.40147400e+01)));
  jac[117] += ((-mw_avg * pres_mod[7] / 3.40147400e+01 + rho / 3.40147400e+01) * fwd_rxn_rates[25] + pres_mod[7] * ((-mw_avg * fwd_rxn_rates[25] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[25] / 3.40147400e+01)));
  jac[117] += -1.0 * (2.0 * (-mw_avg * fwd_rxn_rates[26] / 3.40147400e+01));
  jac[117] += ((-mw_avg * fwd_rxn_rates[27] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[27] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[27] / 3.40147400e+01));
  jac[117] += ((-mw_avg * fwd_rxn_rates[36] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[36] / 3.40147400e+01));
  jac[117] += -1.0 * ((-mw_avg * fwd_rxn_rates[37] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[37] / 3.40147400e+01));
  jac[117] += ((-mw_avg * fwd_rxn_rates[44] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[44] / 3.40147400e+01 + exp(3.08132330e+01 - (1.99777016e+03 / T)) * (rho / 3.40147400e+01) * conc[0]));
  jac[117] += -1.0 * ((-mw_avg * fwd_rxn_rates[45] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[45] / 3.40147400e+01));
  jac[117] += ((-mw_avg * fwd_rxn_rates[50] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[50] / 3.40147400e+01 + exp(2.81849062e+01 - (1.60022900e+02 / T)) * (rho / 3.40147400e+01) * conc[3]));
  jac[117] += -1.0 * ((-mw_avg * fwd_rxn_rates[51] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[51] / 3.40147400e+01));
  jac[117] += ((-mw_avg * fwd_rxn_rates[52] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[52] / 3.40147400e+01 + exp(3.19604378e+01 - (3.65838516e+03 / T)) * (rho / 3.40147400e+01) * conc[3]));
  jac[117] += -1.0 * ((-mw_avg * fwd_rxn_rates[53] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[53] / 3.40147400e+01));
  jac[117] += sp_rates[4] * mw_avg / 3.40147400e+01;
  jac[117] *= 1.80153400e+01 / rho;

  //partial of omega_H2O wrt Y_N2;
  jac[131] = ((-mw_avg * fwd_rxn_rates[6] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[6] / 2.80134000e+01));
  jac[131] += -1.0 * ((-mw_avg * fwd_rxn_rates[7] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[7] / 2.80134000e+01));
  jac[131] += (2.0 * (-mw_avg * fwd_rxn_rates[8] / 2.80134000e+01));
  jac[131] += -1.0 * ((-mw_avg * fwd_rxn_rates[9] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[9] / 2.80134000e+01));
  jac[131] += -1.0 * ((-mw_avg * pres_mod[6] / 2.80134000e+01 + 2.0 * rho / 2.80134000e+01) * fwd_rxn_rates[24] + pres_mod[6] * ((-mw_avg * fwd_rxn_rates[24] / 2.80134000e+01)));
  jac[131] += ((-mw_avg * pres_mod[7] / 2.80134000e+01 + 2.0 * rho / 2.80134000e+01) * fwd_rxn_rates[25] + pres_mod[7] * ((-mw_avg * fwd_rxn_rates[25] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[25] / 2.80134000e+01)));
  jac[131] += -1.0 * (2.0 * (-mw_avg * fwd_rxn_rates[26] / 2.80134000e+01));
  jac[131] += ((-mw_avg * fwd_rxn_rates[27] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[27] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[27] / 2.80134000e+01));
  jac[131] += ((-mw_avg * fwd_rxn_rates[36] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[36] / 2.80134000e+01));
  jac[131] += -1.0 * ((-mw_avg * fwd_rxn_rates[37] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[37] / 2.80134000e+01));
  jac[131] += ((-mw_avg * fwd_rxn_rates[44] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[44] / 2.80134000e+01));
  jac[131] += -1.0 * ((-mw_avg * fwd_rxn_rates[45] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[45] / 2.80134000e+01));
  jac[131] += ((-mw_avg * fwd_rxn_rates[50] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[50] / 2.80134000e+01));
  jac[131] += -1.0 * ((-mw_avg * fwd_rxn_rates[51] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[51] / 2.80134000e+01));
  jac[131] += ((-mw_avg * fwd_rxn_rates[52] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[52] / 2.80134000e+01));
  jac[131] += -1.0 * ((-mw_avg * fwd_rxn_rates[53] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[53] / 2.80134000e+01));
  jac[131] += sp_rates[4] * mw_avg / 2.80134000e+01;
  jac[131] *= 1.80153400e+01 / rho;

  //partial of omega_H2O wrt Y_AR;
  jac[145] = ((-mw_avg * fwd_rxn_rates[6] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[6] / 3.99480000e+01));
  jac[145] += -1.0 * ((-mw_avg * fwd_rxn_rates[7] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[7] / 3.99480000e+01));
  jac[145] += (2.0 * (-mw_avg * fwd_rxn_rates[8] / 3.99480000e+01));
  jac[145] += -1.0 * ((-mw_avg * fwd_rxn_rates[9] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[9] / 3.99480000e+01));
  jac[145] += -1.0 * ((-mw_avg * pres_mod[6] / 3.99480000e+01 + rho / 3.99480000e+01) * fwd_rxn_rates[24] + pres_mod[6] * ((-mw_avg * fwd_rxn_rates[24] / 3.99480000e+01)));
  jac[145] += ((-mw_avg * pres_mod[7] / 3.99480000e+01 + rho / 3.99480000e+01) * fwd_rxn_rates[25] + pres_mod[7] * ((-mw_avg * fwd_rxn_rates[25] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[25] / 3.99480000e+01)));
  jac[145] += -1.0 * (2.0 * (-mw_avg * fwd_rxn_rates[26] / 3.99480000e+01));
  jac[145] += ((-mw_avg * fwd_rxn_rates[27] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[27] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[27] / 3.99480000e+01));
  jac[145] += ((-mw_avg * fwd_rxn_rates[36] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[36] / 3.99480000e+01));
  jac[145] += -1.0 * ((-mw_avg * fwd_rxn_rates[37] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[37] / 3.99480000e+01));
  jac[145] += ((-mw_avg * fwd_rxn_rates[44] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[44] / 3.99480000e+01));
  jac[145] += -1.0 * ((-mw_avg * fwd_rxn_rates[45] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[45] / 3.99480000e+01));
  jac[145] += ((-mw_avg * fwd_rxn_rates[50] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[50] / 3.99480000e+01));
  jac[145] += -1.0 * ((-mw_avg * fwd_rxn_rates[51] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[51] / 3.99480000e+01));
  jac[145] += ((-mw_avg * fwd_rxn_rates[52] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[52] / 3.99480000e+01));
  jac[145] += -1.0 * ((-mw_avg * fwd_rxn_rates[53] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[53] / 3.99480000e+01));
  jac[145] += sp_rates[4] * mw_avg / 3.99480000e+01;
  jac[145] *= 1.80153400e+01 / rho;

  //partial of omega_H2O wrt Y_HE;
  jac[159] = ((-mw_avg * fwd_rxn_rates[6] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[6] / 4.00260000e+00));
  jac[159] += -1.0 * ((-mw_avg * fwd_rxn_rates[7] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[7] / 4.00260000e+00));
  jac[159] += (2.0 * (-mw_avg * fwd_rxn_rates[8] / 4.00260000e+00));
  jac[159] += -1.0 * ((-mw_avg * fwd_rxn_rates[9] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[9] / 4.00260000e+00));
  jac[159] += -1.0 * ((-mw_avg * pres_mod[6] / 4.00260000e+00 + 1.1 * rho / 4.00260000e+00) * fwd_rxn_rates[24] + pres_mod[6] * ((-mw_avg * fwd_rxn_rates[24] / 4.00260000e+00)));
  jac[159] += ((-mw_avg * pres_mod[7] / 4.00260000e+00 + 1.1 * rho / 4.00260000e+00) * fwd_rxn_rates[25] + pres_mod[7] * ((-mw_avg * fwd_rxn_rates[25] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[25] / 4.00260000e+00)));
  jac[159] += -1.0 * (2.0 * (-mw_avg * fwd_rxn_rates[26] / 4.00260000e+00));
  jac[159] += ((-mw_avg * fwd_rxn_rates[27] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[27] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[27] / 4.00260000e+00));
  jac[159] += ((-mw_avg * fwd_rxn_rates[36] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[36] / 4.00260000e+00));
  jac[159] += -1.0 * ((-mw_avg * fwd_rxn_rates[37] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[37] / 4.00260000e+00));
  jac[159] += ((-mw_avg * fwd_rxn_rates[44] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[44] / 4.00260000e+00));
  jac[159] += -1.0 * ((-mw_avg * fwd_rxn_rates[45] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[45] / 4.00260000e+00));
  jac[159] += ((-mw_avg * fwd_rxn_rates[50] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[50] / 4.00260000e+00));
  jac[159] += -1.0 * ((-mw_avg * fwd_rxn_rates[51] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[51] / 4.00260000e+00));
  jac[159] += ((-mw_avg * fwd_rxn_rates[52] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[52] / 4.00260000e+00));
  jac[159] += -1.0 * ((-mw_avg * fwd_rxn_rates[53] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[53] / 4.00260000e+00));
  jac[159] += sp_rates[4] * mw_avg / 4.00260000e+00;
  jac[159] *= 1.80153400e+01 / rho;

  //partial of omega_H2O wrt Y_CO;
  jac[173] = ((-mw_avg * fwd_rxn_rates[6] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[6] / 2.80105500e+01));
  jac[173] += -1.0 * ((-mw_avg * fwd_rxn_rates[7] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[7] / 2.80105500e+01));
  jac[173] += (2.0 * (-mw_avg * fwd_rxn_rates[8] / 2.80105500e+01));
  jac[173] += -1.0 * ((-mw_avg * fwd_rxn_rates[9] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[9] / 2.80105500e+01));
  jac[173] += -1.0 * ((-mw_avg * pres_mod[6] / 2.80105500e+01 + 1.9 * rho / 2.80105500e+01) * fwd_rxn_rates[24] + pres_mod[6] * ((-mw_avg * fwd_rxn_rates[24] / 2.80105500e+01)));
  jac[173] += ((-mw_avg * pres_mod[7] / 2.80105500e+01 + 1.9 * rho / 2.80105500e+01) * fwd_rxn_rates[25] + pres_mod[7] * ((-mw_avg * fwd_rxn_rates[25] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[25] / 2.80105500e+01)));
  jac[173] += -1.0 * (2.0 * (-mw_avg * fwd_rxn_rates[26] / 2.80105500e+01));
  jac[173] += ((-mw_avg * fwd_rxn_rates[27] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[27] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[27] / 2.80105500e+01));
  jac[173] += ((-mw_avg * fwd_rxn_rates[36] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[36] / 2.80105500e+01));
  jac[173] += -1.0 * ((-mw_avg * fwd_rxn_rates[37] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[37] / 2.80105500e+01));
  jac[173] += ((-mw_avg * fwd_rxn_rates[44] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[44] / 2.80105500e+01));
  jac[173] += -1.0 * ((-mw_avg * fwd_rxn_rates[45] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[45] / 2.80105500e+01));
  jac[173] += ((-mw_avg * fwd_rxn_rates[50] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[50] / 2.80105500e+01));
  jac[173] += -1.0 * ((-mw_avg * fwd_rxn_rates[51] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[51] / 2.80105500e+01));
  jac[173] += ((-mw_avg * fwd_rxn_rates[52] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[52] / 2.80105500e+01));
  jac[173] += -1.0 * ((-mw_avg * fwd_rxn_rates[53] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[53] / 2.80105500e+01));
  jac[173] += sp_rates[4] * mw_avg / 2.80105500e+01;
  jac[173] *= 1.80153400e+01 / rho;

  //partial of omega_H2O wrt Y_CO2;
  jac[187] = ((-mw_avg * fwd_rxn_rates[6] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[6] / 4.40099500e+01));
  jac[187] += -1.0 * ((-mw_avg * fwd_rxn_rates[7] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[7] / 4.40099500e+01));
  jac[187] += (2.0 * (-mw_avg * fwd_rxn_rates[8] / 4.40099500e+01));
  jac[187] += -1.0 * ((-mw_avg * fwd_rxn_rates[9] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[9] / 4.40099500e+01));
  jac[187] += -1.0 * ((-mw_avg * pres_mod[6] / 4.40099500e+01 + 3.8 * rho / 4.40099500e+01) * fwd_rxn_rates[24] + pres_mod[6] * ((-mw_avg * fwd_rxn_rates[24] / 4.40099500e+01)));
  jac[187] += ((-mw_avg * pres_mod[7] / 4.40099500e+01 + 3.8 * rho / 4.40099500e+01) * fwd_rxn_rates[25] + pres_mod[7] * ((-mw_avg * fwd_rxn_rates[25] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[25] / 4.40099500e+01)));
  jac[187] += -1.0 * (2.0 * (-mw_avg * fwd_rxn_rates[26] / 4.40099500e+01));
  jac[187] += ((-mw_avg * fwd_rxn_rates[27] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[27] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[27] / 4.40099500e+01));
  jac[187] += ((-mw_avg * fwd_rxn_rates[36] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[36] / 4.40099500e+01));
  jac[187] += -1.0 * ((-mw_avg * fwd_rxn_rates[37] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[37] / 4.40099500e+01));
  jac[187] += ((-mw_avg * fwd_rxn_rates[44] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[44] / 4.40099500e+01));
  jac[187] += -1.0 * ((-mw_avg * fwd_rxn_rates[45] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[45] / 4.40099500e+01));
  jac[187] += ((-mw_avg * fwd_rxn_rates[50] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[50] / 4.40099500e+01));
  jac[187] += -1.0 * ((-mw_avg * fwd_rxn_rates[51] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[51] / 4.40099500e+01));
  jac[187] += ((-mw_avg * fwd_rxn_rates[52] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[52] / 4.40099500e+01));
  jac[187] += -1.0 * ((-mw_avg * fwd_rxn_rates[53] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[53] / 4.40099500e+01));
  jac[187] += sp_rates[4] * mw_avg / 4.40099500e+01;
  jac[187] *= 1.80153400e+01 / rho;

  //partial of omega_O2 wrt T;
  jac[6] = -1.0 * ((1.0 / T) * (fwd_rxn_rates[0] * ((7.69216995e+03 / T) + 1.0 - 2.0)));
  jac[6] += ((1.0 / T) * (fwd_rxn_rates[1] * (4.04780000e-01 + (-7.48736077e+02 / T) + 1.0 - 2.0)));
  jac[6] += ((-pres_mod[2] * fwd_rxn_rates[16] / T) + (pres_mod[2] / T) * (fwd_rxn_rates[16] * (-5.00000000e-01 + 1.0 - 2.0)));
  jac[6] += -1.0 * ((-pres_mod[3] * fwd_rxn_rates[17] / T) + (pres_mod[3] / T) * (fwd_rxn_rates[17] * (-9.34910000e-01 + (6.02702601e+04 / T) + 1.0 - 1.0)));
  jac[6] += ((1.0 / T) * (fwd_rxn_rates[18] * ((-8.99751398e+02 / T) + 1.0 - 3.0)));
  jac[6] += -1.0 * ((1.0 / T) * (fwd_rxn_rates[19] * (-4.34910000e-01 + (5.93745344e+04 / T) + 1.0 - 2.0)));
  jac[6] += ((1.0 / T) * (fwd_rxn_rates[20] * ((-8.99751398e+02 / T) + 1.0 - 3.0)));
  jac[6] += -1.0 * ((1.0 / T) * (fwd_rxn_rates[21] * (-4.34910000e-01 + (5.93745344e+04 / T) + 1.0 - 2.0)));
  Pr = (m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * (exp(1.87346128e+01 - 2.16 * logT - (2.64088106e+02 / T)));
  Fcent = 5.00000000e-01 * exp(-T / 1.00000000e-30) + 5.00000000e-01 * exp(T / 1.00000000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  lnF_AB = 2.0 * log(Fcent) * A / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)));
  jac[6] += -1.0 * (pres_mod[8]* (((-2.1600e+00 + (2.64088106e+02 / T) - 1.0) / (T * (1.0 + Pr))) + (((1.0 / (Fcent * (1.0 + A * A / (B * B)))) - lnF_AB * (-2.90977303e-01 * B + 5.10817170e-01 * A) / Fcent) * (-5.00000000e+29 * exp(-T / 1.00000000e-30) - 5.00000000e-31 * exp(-T / 1.00000000e+30))) - lnF_AB * (4.34294482e-01 * B + 6.08012275e-02 * A) * (-2.16000000e+00 + (2.64088106e+02 / T) - 1.0) / T) * fwd_rxn_rates[28] + (pres_mod[8] / T) * (fwd_rxn_rates[28] * (4.40000000e-01 + 1.0 - 2.0)));
  Pr = (m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * (exp(1.87345962e+01 - 2.159967 * logT - (2.64188750e+02 / T)));
  Fcent = 5.00000000e-01 * exp(-T / 1.00000000e-30) + 5.00000000e-01 * exp(T / 1.00000000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  lnF_AB = 2.0 * log(Fcent) * A / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)));
  jac[6] += (pres_mod[9]* (((-2.1600e+00 + (2.64188750e+02 / T) - 1.0) / (T * (1.0 + Pr))) + (((1.0 / (Fcent * (1.0 + A * A / (B * B)))) - lnF_AB * (-2.90977303e-01 * B + 5.10817170e-01 * A) / Fcent) * (-5.00000000e+29 * exp(-T / 1.00000000e-30) - 5.00000000e-31 * exp(-T / 1.00000000e+30))) - lnF_AB * (4.34294482e-01 * B + 6.08012275e-02 * A) * (-2.15996700e+00 + (2.64188750e+02 / T) - 1.0) / T) * fwd_rxn_rates[29] + (pres_mod[9] / T) * (fwd_rxn_rates[29] * (-4.94330000e-02 + (2.52182000e+04 / T) + 1.0 - 1.0)));
  jac[6] += ((1.0 / T) * (fwd_rxn_rates[30] * (2.09000000e+00 + (-7.30167382e+02 / T) + 1.0 - 2.0)));
  jac[6] += -1.0 * ((1.0 / T) * (fwd_rxn_rates[31] * (2.60280000e+00 + (2.65552467e+04 / T) + 1.0 - 2.0)));
  jac[6] += ((1.0 / T) * (fwd_rxn_rates[34] * (1.00000000e+00 + (-3.64293641e+02 / T) + 1.0 - 2.0)));
  jac[6] += -1.0 * ((1.0 / T) * (fwd_rxn_rates[35] * (1.45930000e+00 + (2.62487877e+04 / T) + 1.0 - 2.0)));
  jac[6] += ((1.0 / T) * (fwd_rxn_rates[36] * ((-2.50098683e+02 / T) + 1.0 - 2.0)));
  jac[6] += -1.0 * ((1.0 / T) * (fwd_rxn_rates[37] * (1.85740000e-01 + (3.48643603e+04 / T) + 1.0 - 2.0)));
  jac[6] += ((1.0 / T) * (fwd_rxn_rates[38] * ((6.02954209e+03 / T) + 1.0 - 2.0)));
  jac[6] += -1.0 * ((1.0 / T) * (fwd_rxn_rates[39] * (-2.34470000e-01 + (2.53213594e+04 / T) + 1.0 - 2.0)));
  jac[6] += ((1.0 / T) * (fwd_rxn_rates[40] * ((-8.19890914e+02 / T) + 1.0 - 2.0)));
  jac[6] += -1.0 * ((1.0 / T) * (fwd_rxn_rates[41] * (-2.34470000e-01 + (1.84720774e+04 / T) + 1.0 - 2.0)));
  jac[6] *= 3.19988000e+01 / rho;

  //partial of omega_O2 wrt Y_H;
  jac[20] = -1.0 * ((-mw_avg * fwd_rxn_rates[0] / 1.00797000e+00 + exp(3.22754120e+01 - (7.69216995e+03 / T)) * (rho / 1.00797000e+00) * conc[5]) + (-mw_avg * fwd_rxn_rates[0] / 1.00797000e+00));
  jac[20] += ((-mw_avg * fwd_rxn_rates[1] / 1.00797000e+00) + (-mw_avg * fwd_rxn_rates[1] / 1.00797000e+00));
  jac[20] += ((-mw_avg * pres_mod[2] / 1.00797000e+00 + rho / 1.00797000e+00) * fwd_rxn_rates[16] + pres_mod[2] * (2.0 * (-mw_avg * fwd_rxn_rates[16] / 1.00797000e+00)));
  jac[20] += -1.0 * ((-mw_avg * pres_mod[3] / 1.00797000e+00 + rho / 1.00797000e+00) * fwd_rxn_rates[17] + pres_mod[3] * ((-mw_avg * fwd_rxn_rates[17] / 1.00797000e+00)));
  jac[20] += (2.0 * (-mw_avg * fwd_rxn_rates[18] / 1.00797000e+00) + (-mw_avg * fwd_rxn_rates[18] / 1.00797000e+00));
  jac[20] += -1.0 * ((-mw_avg * fwd_rxn_rates[19] / 1.00797000e+00) + (-mw_avg * fwd_rxn_rates[19] / 1.00797000e+00));
  jac[20] += (2.0 * (-mw_avg * fwd_rxn_rates[20] / 1.00797000e+00) + (-mw_avg * fwd_rxn_rates[20] / 1.00797000e+00));
  jac[20] += -1.0 * ((-mw_avg * fwd_rxn_rates[21] / 1.00797000e+00) + (-mw_avg * fwd_rxn_rates[21] / 1.00797000e+00));
  Pr = (m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * (exp(1.87346128e+01 - 2.16 * logT - (2.64088106e+02 / T)));
  Fcent = 5.0000e-01 * exp(-T / 1.0000e-30) + 5.0000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[20] += -1.0 * (pres_mod[8] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 1.00797000e+00 + rho / ((m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * 1.00797000e+00)) * fwd_rxn_rates[28] + pres_mod[8] * ((-mw_avg * fwd_rxn_rates[28] / 1.00797000e+00 + exp(2.91680604e+01 + 0.44 * logT) * (rho / 1.00797000e+00) * conc[5]) + (-mw_avg * fwd_rxn_rates[28] / 1.00797000e+00)));
  Pr = (m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * (exp(1.87345962e+01 - 2.159967 * logT - (2.64188750e+02 / T)));
  Fcent = 5.0000e-01 * exp(-T / 1.0000e-30) + 5.0000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[20] += (pres_mod[9] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 1.00797000e+00 + rho / ((m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * 1.00797000e+00)) * fwd_rxn_rates[29] + pres_mod[9] * ((-mw_avg * fwd_rxn_rates[29] / 1.00797000e+00)));
  jac[20] += ((-mw_avg * fwd_rxn_rates[30] / 1.00797000e+00 + exp(1.48271115e+01 + 2.09 * logT - (-7.30167382e+02 / T)) * (rho / 1.00797000e+00) * conc[6]) + (-mw_avg * fwd_rxn_rates[30] / 1.00797000e+00));
  jac[20] += -1.0 * ((-mw_avg * fwd_rxn_rates[31] / 1.00797000e+00) + (-mw_avg * fwd_rxn_rates[31] / 1.00797000e+00));
  jac[20] += ((-mw_avg * fwd_rxn_rates[34] / 1.00797000e+00) + (-mw_avg * fwd_rxn_rates[34] / 1.00797000e+00));
  jac[20] += -1.0 * ((-mw_avg * fwd_rxn_rates[35] / 1.00797000e+00) + (-mw_avg * fwd_rxn_rates[35] / 1.00797000e+00));
  jac[20] += ((-mw_avg * fwd_rxn_rates[36] / 1.00797000e+00) + (-mw_avg * fwd_rxn_rates[36] / 1.00797000e+00));
  jac[20] += -1.0 * ((-mw_avg * fwd_rxn_rates[37] / 1.00797000e+00) + (-mw_avg * fwd_rxn_rates[37] / 1.00797000e+00));
  jac[20] += (2.0 * (-mw_avg * fwd_rxn_rates[38] / 1.00797000e+00));
  jac[20] += -1.0 * ((-mw_avg * fwd_rxn_rates[39] / 1.00797000e+00) + (-mw_avg * fwd_rxn_rates[39] / 1.00797000e+00));
  jac[20] += (2.0 * (-mw_avg * fwd_rxn_rates[40] / 1.00797000e+00));
  jac[20] += -1.0 * ((-mw_avg * fwd_rxn_rates[41] / 1.00797000e+00) + (-mw_avg * fwd_rxn_rates[41] / 1.00797000e+00));
  jac[20] += sp_rates[5] * mw_avg / 1.00797000e+00;
  jac[20] *= 3.19988000e+01 / rho;

  //partial of omega_O2 wrt Y_H2;
  jac[34] = -1.0 * ((-mw_avg * fwd_rxn_rates[0] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[0] / 2.01594000e+00));
  jac[34] += ((-mw_avg * fwd_rxn_rates[1] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[1] / 2.01594000e+00));
  jac[34] += ((-mw_avg * pres_mod[2] / 2.01594000e+00 + 2.5 * rho / 2.01594000e+00) * fwd_rxn_rates[16] + pres_mod[2] * (2.0 * (-mw_avg * fwd_rxn_rates[16] / 2.01594000e+00)));
  jac[34] += -1.0 * ((-mw_avg * pres_mod[3] / 2.01594000e+00 + 2.5 * rho / 2.01594000e+00) * fwd_rxn_rates[17] + pres_mod[3] * ((-mw_avg * fwd_rxn_rates[17] / 2.01594000e+00)));
  jac[34] += (2.0 * (-mw_avg * fwd_rxn_rates[18] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[18] / 2.01594000e+00));
  jac[34] += -1.0 * ((-mw_avg * fwd_rxn_rates[19] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[19] / 2.01594000e+00));
  jac[34] += (2.0 * (-mw_avg * fwd_rxn_rates[20] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[20] / 2.01594000e+00));
  jac[34] += -1.0 * ((-mw_avg * fwd_rxn_rates[21] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[21] / 2.01594000e+00));
  Pr = (m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * (exp(1.87346128e+01 - 2.16 * logT - (2.64088106e+02 / T)));
  Fcent = 5.0000e-01 * exp(-T / 1.0000e-30) + 5.0000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[34] += -1.0 * (pres_mod[8] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 2.01594000e+00 + rho * 2.0 / ((m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * 2.01594000e+00)) * fwd_rxn_rates[28] + pres_mod[8] * ((-mw_avg * fwd_rxn_rates[28] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[28] / 2.01594000e+00)));
  Pr = (m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * (exp(1.87345962e+01 - 2.159967 * logT - (2.64188750e+02 / T)));
  Fcent = 5.0000e-01 * exp(-T / 1.0000e-30) + 5.0000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[34] += (pres_mod[9] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 2.01594000e+00 + rho * 2.0 / ((m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * 2.01594000e+00)) * fwd_rxn_rates[29] + pres_mod[9] * ((-mw_avg * fwd_rxn_rates[29] / 2.01594000e+00)));
  jac[34] += ((-mw_avg * fwd_rxn_rates[30] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[30] / 2.01594000e+00));
  jac[34] += -1.0 * ((-mw_avg * fwd_rxn_rates[31] / 2.01594000e+00 + exp(1.17897235e+01 + 2.6028 * logT - (2.65552467e+04 / T)) * (rho / 2.01594000e+00) * conc[5]) + (-mw_avg * fwd_rxn_rates[31] / 2.01594000e+00));
  jac[34] += ((-mw_avg * fwd_rxn_rates[34] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[34] / 2.01594000e+00));
  jac[34] += -1.0 * ((-mw_avg * fwd_rxn_rates[35] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[35] / 2.01594000e+00));
  jac[34] += ((-mw_avg * fwd_rxn_rates[36] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[36] / 2.01594000e+00));
  jac[34] += -1.0 * ((-mw_avg * fwd_rxn_rates[37] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[37] / 2.01594000e+00));
  jac[34] += (2.0 * (-mw_avg * fwd_rxn_rates[38] / 2.01594000e+00));
  jac[34] += -1.0 * ((-mw_avg * fwd_rxn_rates[39] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[39] / 2.01594000e+00));
  jac[34] += (2.0 * (-mw_avg * fwd_rxn_rates[40] / 2.01594000e+00));
  jac[34] += -1.0 * ((-mw_avg * fwd_rxn_rates[41] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[41] / 2.01594000e+00));
  jac[34] += sp_rates[5] * mw_avg / 2.01594000e+00;
  jac[34] *= 3.19988000e+01 / rho;

  //partial of omega_O2 wrt Y_O;
  jac[48] = -1.0 * ((-mw_avg * fwd_rxn_rates[0] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[0] / 1.59994000e+01));
  jac[48] += ((-mw_avg * fwd_rxn_rates[1] / 1.59994000e+01 + exp(2.63075513e+01 + 0.40478 * logT - (-7.48736077e+02 / T)) * (rho / 1.59994000e+01) * conc[3]) + (-mw_avg * fwd_rxn_rates[1] / 1.59994000e+01));
  jac[48] += ((-mw_avg * pres_mod[2] / 1.59994000e+01 + rho / 1.59994000e+01) * fwd_rxn_rates[16] + pres_mod[2] * (2.0 * (-mw_avg * fwd_rxn_rates[16] / 1.59994000e+01 + exp(3.63576645e+01 - 0.5 * logT) * (rho / 1.59994000e+01) * conc[2])));
  jac[48] += -1.0 * ((-mw_avg * pres_mod[3] / 1.59994000e+01 + rho / 1.59994000e+01) * fwd_rxn_rates[17] + pres_mod[3] * ((-mw_avg * fwd_rxn_rates[17] / 1.59994000e+01)));
  jac[48] += (2.0 * (-mw_avg * fwd_rxn_rates[18] / 1.59994000e+01 + exp(3.05680644e+01 - (-8.99751398e+02 / T)) * (rho / 1.59994000e+01) * conc[2] * conc[9]) + (-mw_avg * fwd_rxn_rates[18] / 1.59994000e+01));
  jac[48] += -1.0 * ((-mw_avg * fwd_rxn_rates[19] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[19] / 1.59994000e+01));
  jac[48] += (2.0 * (-mw_avg * fwd_rxn_rates[20] / 1.59994000e+01 + exp(3.05680644e+01 - (-8.99751398e+02 / T)) * (rho / 1.59994000e+01) * conc[2] * conc[10]) + (-mw_avg * fwd_rxn_rates[20] / 1.59994000e+01));
  jac[48] += -1.0 * ((-mw_avg * fwd_rxn_rates[21] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[21] / 1.59994000e+01));
  Pr = (m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * (exp(1.87346128e+01 - 2.16 * logT - (2.64088106e+02 / T)));
  Fcent = 5.0000e-01 * exp(-T / 1.0000e-30) + 5.0000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[48] += -1.0 * (pres_mod[8] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 1.59994000e+01 + rho / ((m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * 1.59994000e+01)) * fwd_rxn_rates[28] + pres_mod[8] * ((-mw_avg * fwd_rxn_rates[28] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[28] / 1.59994000e+01)));
  Pr = (m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * (exp(1.87345962e+01 - 2.159967 * logT - (2.64188750e+02 / T)));
  Fcent = 5.0000e-01 * exp(-T / 1.0000e-30) + 5.0000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[48] += (pres_mod[9] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 1.59994000e+01 + rho / ((m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * 1.59994000e+01)) * fwd_rxn_rates[29] + pres_mod[9] * ((-mw_avg * fwd_rxn_rates[29] / 1.59994000e+01)));
  jac[48] += ((-mw_avg * fwd_rxn_rates[30] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[30] / 1.59994000e+01));
  jac[48] += -1.0 * ((-mw_avg * fwd_rxn_rates[31] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[31] / 1.59994000e+01));
  jac[48] += ((-mw_avg * fwd_rxn_rates[34] / 1.59994000e+01 + exp(2.40731699e+01 + 1.0 * logT - (-3.64293641e+02 / T)) * (rho / 1.59994000e+01) * conc[6]) + (-mw_avg * fwd_rxn_rates[34] / 1.59994000e+01));
  jac[48] += -1.0 * ((-mw_avg * fwd_rxn_rates[35] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[35] / 1.59994000e+01));
  jac[48] += ((-mw_avg * fwd_rxn_rates[36] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[36] / 1.59994000e+01));
  jac[48] += -1.0 * ((-mw_avg * fwd_rxn_rates[37] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[37] / 1.59994000e+01));
  jac[48] += (2.0 * (-mw_avg * fwd_rxn_rates[38] / 1.59994000e+01));
  jac[48] += -1.0 * ((-mw_avg * fwd_rxn_rates[39] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[39] / 1.59994000e+01));
  jac[48] += (2.0 * (-mw_avg * fwd_rxn_rates[40] / 1.59994000e+01));
  jac[48] += -1.0 * ((-mw_avg * fwd_rxn_rates[41] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[41] / 1.59994000e+01));
  jac[48] += sp_rates[5] * mw_avg / 1.59994000e+01;
  jac[48] *= 3.19988000e+01 / rho;

  //partial of omega_O2 wrt Y_OH;
  jac[62] = -1.0 * ((-mw_avg * fwd_rxn_rates[0] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[0] / 1.70073700e+01));
  jac[62] += ((-mw_avg * fwd_rxn_rates[1] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[1] / 1.70073700e+01 + exp(2.63075513e+01 + 0.40478 * logT - (-7.48736077e+02 / T)) * (rho / 1.70073700e+01) * conc[2]));
  jac[62] += ((-mw_avg * pres_mod[2] / 1.70073700e+01 + rho / 1.70073700e+01) * fwd_rxn_rates[16] + pres_mod[2] * (2.0 * (-mw_avg * fwd_rxn_rates[16] / 1.70073700e+01)));
  jac[62] += -1.0 * ((-mw_avg * pres_mod[3] / 1.70073700e+01 + rho / 1.70073700e+01) * fwd_rxn_rates[17] + pres_mod[3] * ((-mw_avg * fwd_rxn_rates[17] / 1.70073700e+01)));
  jac[62] += (2.0 * (-mw_avg * fwd_rxn_rates[18] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[18] / 1.70073700e+01));
  jac[62] += -1.0 * ((-mw_avg * fwd_rxn_rates[19] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[19] / 1.70073700e+01));
  jac[62] += (2.0 * (-mw_avg * fwd_rxn_rates[20] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[20] / 1.70073700e+01));
  jac[62] += -1.0 * ((-mw_avg * fwd_rxn_rates[21] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[21] / 1.70073700e+01));
  Pr = (m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * (exp(1.87346128e+01 - 2.16 * logT - (2.64088106e+02 / T)));
  Fcent = 5.0000e-01 * exp(-T / 1.0000e-30) + 5.0000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[62] += -1.0 * (pres_mod[8] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 1.70073700e+01 + rho / ((m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * 1.70073700e+01)) * fwd_rxn_rates[28] + pres_mod[8] * ((-mw_avg * fwd_rxn_rates[28] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[28] / 1.70073700e+01)));
  Pr = (m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * (exp(1.87345962e+01 - 2.159967 * logT - (2.64188750e+02 / T)));
  Fcent = 5.0000e-01 * exp(-T / 1.0000e-30) + 5.0000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[62] += (pres_mod[9] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 1.70073700e+01 + rho / ((m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * 1.70073700e+01)) * fwd_rxn_rates[29] + pres_mod[9] * ((-mw_avg * fwd_rxn_rates[29] / 1.70073700e+01)));
  jac[62] += ((-mw_avg * fwd_rxn_rates[30] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[30] / 1.70073700e+01));
  jac[62] += -1.0 * ((-mw_avg * fwd_rxn_rates[31] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[31] / 1.70073700e+01));
  jac[62] += ((-mw_avg * fwd_rxn_rates[34] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[34] / 1.70073700e+01));
  jac[62] += -1.0 * ((-mw_avg * fwd_rxn_rates[35] / 1.70073700e+01 + exp(2.06516517e+01 + 1.4593 * logT - (2.62487877e+04 / T)) * (rho / 1.70073700e+01) * conc[5]) + (-mw_avg * fwd_rxn_rates[35] / 1.70073700e+01));
  jac[62] += ((-mw_avg * fwd_rxn_rates[36] / 1.70073700e+01 + exp(3.09948627e+01 - (-2.50098683e+02 / T)) * (rho / 1.70073700e+01) * conc[6]) + (-mw_avg * fwd_rxn_rates[36] / 1.70073700e+01));
  jac[62] += -1.0 * ((-mw_avg * fwd_rxn_rates[37] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[37] / 1.70073700e+01));
  jac[62] += (2.0 * (-mw_avg * fwd_rxn_rates[38] / 1.70073700e+01));
  jac[62] += -1.0 * ((-mw_avg * fwd_rxn_rates[39] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[39] / 1.70073700e+01));
  jac[62] += (2.0 * (-mw_avg * fwd_rxn_rates[40] / 1.70073700e+01));
  jac[62] += -1.0 * ((-mw_avg * fwd_rxn_rates[41] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[41] / 1.70073700e+01));
  jac[62] += sp_rates[5] * mw_avg / 1.70073700e+01;
  jac[62] *= 3.19988000e+01 / rho;

  //partial of omega_O2 wrt Y_H2O;
  jac[76] = -1.0 * ((-mw_avg * fwd_rxn_rates[0] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[0] / 1.80153400e+01));
  jac[76] += ((-mw_avg * fwd_rxn_rates[1] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[1] / 1.80153400e+01));
  jac[76] += ((-mw_avg * pres_mod[2] / 1.80153400e+01 + 12.0 * rho / 1.80153400e+01) * fwd_rxn_rates[16] + pres_mod[2] * (2.0 * (-mw_avg * fwd_rxn_rates[16] / 1.80153400e+01)));
  jac[76] += -1.0 * ((-mw_avg * pres_mod[3] / 1.80153400e+01 + 12.0 * rho / 1.80153400e+01) * fwd_rxn_rates[17] + pres_mod[3] * ((-mw_avg * fwd_rxn_rates[17] / 1.80153400e+01)));
  jac[76] += (2.0 * (-mw_avg * fwd_rxn_rates[18] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[18] / 1.80153400e+01));
  jac[76] += -1.0 * ((-mw_avg * fwd_rxn_rates[19] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[19] / 1.80153400e+01));
  jac[76] += (2.0 * (-mw_avg * fwd_rxn_rates[20] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[20] / 1.80153400e+01));
  jac[76] += -1.0 * ((-mw_avg * fwd_rxn_rates[21] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[21] / 1.80153400e+01));
  Pr = (m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * (exp(1.87346128e+01 - 2.16 * logT - (2.64088106e+02 / T)));
  Fcent = 5.0000e-01 * exp(-T / 1.0000e-30) + 5.0000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[76] += -1.0 * (pres_mod[8] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 1.80153400e+01 + rho * 14.0 / ((m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * 1.80153400e+01)) * fwd_rxn_rates[28] + pres_mod[8] * ((-mw_avg * fwd_rxn_rates[28] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[28] / 1.80153400e+01)));
  Pr = (m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * (exp(1.87345962e+01 - 2.159967 * logT - (2.64188750e+02 / T)));
  Fcent = 5.0000e-01 * exp(-T / 1.0000e-30) + 5.0000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[76] += (pres_mod[9] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 1.80153400e+01 + rho * 14.0 / ((m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * 1.80153400e+01)) * fwd_rxn_rates[29] + pres_mod[9] * ((-mw_avg * fwd_rxn_rates[29] / 1.80153400e+01)));
  jac[76] += ((-mw_avg * fwd_rxn_rates[30] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[30] / 1.80153400e+01));
  jac[76] += -1.0 * ((-mw_avg * fwd_rxn_rates[31] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[31] / 1.80153400e+01));
  jac[76] += ((-mw_avg * fwd_rxn_rates[34] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[34] / 1.80153400e+01));
  jac[76] += -1.0 * ((-mw_avg * fwd_rxn_rates[35] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[35] / 1.80153400e+01));
  jac[76] += ((-mw_avg * fwd_rxn_rates[36] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[36] / 1.80153400e+01));
  jac[76] += -1.0 * ((-mw_avg * fwd_rxn_rates[37] / 1.80153400e+01 + exp(3.21899484e+01 + 0.18574 * logT - (3.48643603e+04 / T)) * (rho / 1.80153400e+01) * conc[5]) + (-mw_avg * fwd_rxn_rates[37] / 1.80153400e+01));
  jac[76] += (2.0 * (-mw_avg * fwd_rxn_rates[38] / 1.80153400e+01));
  jac[76] += -1.0 * ((-mw_avg * fwd_rxn_rates[39] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[39] / 1.80153400e+01));
  jac[76] += (2.0 * (-mw_avg * fwd_rxn_rates[40] / 1.80153400e+01));
  jac[76] += -1.0 * ((-mw_avg * fwd_rxn_rates[41] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[41] / 1.80153400e+01));
  jac[76] += sp_rates[5] * mw_avg / 1.80153400e+01;
  jac[76] *= 3.19988000e+01 / rho;

  //partial of omega_O2 wrt Y_O2;
  jac[90] = -1.0 * ((-mw_avg * fwd_rxn_rates[0] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[0] / 3.19988000e+01 + exp(3.22754120e+01 - (7.69216995e+03 / T)) * (rho / 3.19988000e+01) * conc[0]));
  jac[90] += ((-mw_avg * fwd_rxn_rates[1] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[1] / 3.19988000e+01));
  jac[90] += ((-mw_avg * pres_mod[2] / 3.19988000e+01 + rho / 3.19988000e+01) * fwd_rxn_rates[16] + pres_mod[2] * (2.0 * (-mw_avg * fwd_rxn_rates[16] / 3.19988000e+01)));
  jac[90] += -1.0 * ((-mw_avg * pres_mod[3] / 3.19988000e+01 + rho / 3.19988000e+01) * fwd_rxn_rates[17] + pres_mod[3] * ((-mw_avg * fwd_rxn_rates[17] / 3.19988000e+01 + exp(4.31509343e+01 - 0.93491 * logT - (6.02702601e+04 / T)) * (rho / 3.19988000e+01))));
  jac[90] += (2.0 * (-mw_avg * fwd_rxn_rates[18] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[18] / 3.19988000e+01));
  jac[90] += -1.0 * ((-mw_avg * fwd_rxn_rates[19] / 3.19988000e+01 + exp(3.73613450e+01 - 0.43491 * logT - (5.93745344e+04 / T)) * (rho / 3.19988000e+01) * conc[9]) + (-mw_avg * fwd_rxn_rates[19] / 3.19988000e+01));
  jac[90] += (2.0 * (-mw_avg * fwd_rxn_rates[20] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[20] / 3.19988000e+01));
  jac[90] += -1.0 * ((-mw_avg * fwd_rxn_rates[21] / 3.19988000e+01 + exp(3.73613450e+01 - 0.43491 * logT - (5.93745344e+04 / T)) * (rho / 3.19988000e+01) * conc[10]) + (-mw_avg * fwd_rxn_rates[21] / 3.19988000e+01));
  Pr = (m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * (exp(1.87346128e+01 - 2.16 * logT - (2.64088106e+02 / T)));
  Fcent = 5.0000e-01 * exp(-T / 1.0000e-30) + 5.0000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[90] += -1.0 * (pres_mod[8] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 3.19988000e+01 + rho * 0.78 / ((m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * 3.19988000e+01)) * fwd_rxn_rates[28] + pres_mod[8] * ((-mw_avg * fwd_rxn_rates[28] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[28] / 3.19988000e+01 + exp(2.91680604e+01 + 0.44 * logT) * (rho / 3.19988000e+01) * conc[0])));
  Pr = (m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * (exp(1.87345962e+01 - 2.159967 * logT - (2.64188750e+02 / T)));
  Fcent = 5.0000e-01 * exp(-T / 1.0000e-30) + 5.0000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[90] += (pres_mod[9] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 3.19988000e+01 + rho * 0.78 / ((m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * 3.19988000e+01)) * fwd_rxn_rates[29] + pres_mod[9] * ((-mw_avg * fwd_rxn_rates[29] / 3.19988000e+01)));
  jac[90] += ((-mw_avg * fwd_rxn_rates[30] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[30] / 3.19988000e+01));
  jac[90] += -1.0 * ((-mw_avg * fwd_rxn_rates[31] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[31] / 3.19988000e+01 + exp(1.17897235e+01 + 2.6028 * logT - (2.65552467e+04 / T)) * (rho / 3.19988000e+01) * conc[1]));
  jac[90] += ((-mw_avg * fwd_rxn_rates[34] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[34] / 3.19988000e+01));
  jac[90] += -1.0 * ((-mw_avg * fwd_rxn_rates[35] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[35] / 3.19988000e+01 + exp(2.06516517e+01 + 1.4593 * logT - (2.62487877e+04 / T)) * (rho / 3.19988000e+01) * conc[3]));
  jac[90] += ((-mw_avg * fwd_rxn_rates[36] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[36] / 3.19988000e+01));
  jac[90] += -1.0 * ((-mw_avg * fwd_rxn_rates[37] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[37] / 3.19988000e+01 + exp(3.21899484e+01 + 0.18574 * logT - (3.48643603e+04 / T)) * (rho / 3.19988000e+01) * conc[4]));
  jac[90] += (2.0 * (-mw_avg * fwd_rxn_rates[38] / 3.19988000e+01));
  jac[90] += -1.0 * ((-mw_avg * fwd_rxn_rates[39] / 3.19988000e+01 + exp(3.74200513e+01 - 0.23447 * logT - (2.53213594e+04 / T)) * (rho / 3.19988000e+01) * conc[7]) + (-mw_avg * fwd_rxn_rates[39] / 3.19988000e+01));
  jac[90] += (2.0 * (-mw_avg * fwd_rxn_rates[40] / 3.19988000e+01));
  jac[90] += -1.0 * ((-mw_avg * fwd_rxn_rates[41] / 3.19988000e+01 + exp(2.93395801e+01 - 0.23447 * logT - (1.84720774e+04 / T)) * (rho / 3.19988000e+01) * conc[7]) + (-mw_avg * fwd_rxn_rates[41] / 3.19988000e+01));
  jac[90] += sp_rates[5] * mw_avg / 3.19988000e+01;
  jac[90] *= 3.19988000e+01 / rho;

  //partial of omega_O2 wrt Y_HO2;
  jac[104] = -1.0 * ((-mw_avg * fwd_rxn_rates[0] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[0] / 3.30067700e+01));
  jac[104] += ((-mw_avg * fwd_rxn_rates[1] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[1] / 3.30067700e+01));
  jac[104] += ((-mw_avg * pres_mod[2] / 3.30067700e+01 + rho / 3.30067700e+01) * fwd_rxn_rates[16] + pres_mod[2] * (2.0 * (-mw_avg * fwd_rxn_rates[16] / 3.30067700e+01)));
  jac[104] += -1.0 * ((-mw_avg * pres_mod[3] / 3.30067700e+01 + rho / 3.30067700e+01) * fwd_rxn_rates[17] + pres_mod[3] * ((-mw_avg * fwd_rxn_rates[17] / 3.30067700e+01)));
  jac[104] += (2.0 * (-mw_avg * fwd_rxn_rates[18] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[18] / 3.30067700e+01));
  jac[104] += -1.0 * ((-mw_avg * fwd_rxn_rates[19] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[19] / 3.30067700e+01));
  jac[104] += (2.0 * (-mw_avg * fwd_rxn_rates[20] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[20] / 3.30067700e+01));
  jac[104] += -1.0 * ((-mw_avg * fwd_rxn_rates[21] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[21] / 3.30067700e+01));
  Pr = (m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * (exp(1.87346128e+01 - 2.16 * logT - (2.64088106e+02 / T)));
  Fcent = 5.0000e-01 * exp(-T / 1.0000e-30) + 5.0000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[104] += -1.0 * (pres_mod[8] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 3.30067700e+01 + rho / ((m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * 3.30067700e+01)) * fwd_rxn_rates[28] + pres_mod[8] * ((-mw_avg * fwd_rxn_rates[28] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[28] / 3.30067700e+01)));
  Pr = (m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * (exp(1.87345962e+01 - 2.159967 * logT - (2.64188750e+02 / T)));
  Fcent = 5.0000e-01 * exp(-T / 1.0000e-30) + 5.0000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[104] += (pres_mod[9] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 3.30067700e+01 + rho / ((m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * 3.30067700e+01)) * fwd_rxn_rates[29] + pres_mod[9] * ((-mw_avg * fwd_rxn_rates[29] / 3.30067700e+01 + exp(3.34150001e+01 - 0.049433 * logT - (2.52182000e+04 / T)) * (rho / 3.30067700e+01))));
  jac[104] += ((-mw_avg * fwd_rxn_rates[30] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[30] / 3.30067700e+01 + exp(1.48271115e+01 + 2.09 * logT - (-7.30167382e+02 / T)) * (rho / 3.30067700e+01) * conc[0]));
  jac[104] += -1.0 * ((-mw_avg * fwd_rxn_rates[31] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[31] / 3.30067700e+01));
  jac[104] += ((-mw_avg * fwd_rxn_rates[34] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[34] / 3.30067700e+01 + exp(2.40731699e+01 + 1.0 * logT - (-3.64293641e+02 / T)) * (rho / 3.30067700e+01) * conc[2]));
  jac[104] += -1.0 * ((-mw_avg * fwd_rxn_rates[35] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[35] / 3.30067700e+01));
  jac[104] += ((-mw_avg * fwd_rxn_rates[36] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[36] / 3.30067700e+01 + exp(3.09948627e+01 - (-2.50098683e+02 / T)) * (rho / 3.30067700e+01) * conc[3]));
  jac[104] += -1.0 * ((-mw_avg * fwd_rxn_rates[37] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[37] / 3.30067700e+01));
  jac[104] += (2.0 * (-mw_avg * fwd_rxn_rates[38] / 3.30067700e+01 + exp(3.36712758e+01 - (6.02954209e+03 / T)) * (rho / 3.30067700e+01) * conc[6]));
  jac[104] += -1.0 * ((-mw_avg * fwd_rxn_rates[39] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[39] / 3.30067700e+01));
  jac[104] += (2.0 * (-mw_avg * fwd_rxn_rates[40] / 3.30067700e+01 + exp(2.55908003e+01 - (-8.19890914e+02 / T)) * (rho / 3.30067700e+01) * conc[6]));
  jac[104] += -1.0 * ((-mw_avg * fwd_rxn_rates[41] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[41] / 3.30067700e+01));
  jac[104] += sp_rates[5] * mw_avg / 3.30067700e+01;
  jac[104] *= 3.19988000e+01 / rho;

  //partial of omega_O2 wrt Y_H2O2;
  jac[118] = -1.0 * ((-mw_avg * fwd_rxn_rates[0] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[0] / 3.40147400e+01));
  jac[118] += ((-mw_avg * fwd_rxn_rates[1] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[1] / 3.40147400e+01));
  jac[118] += ((-mw_avg * pres_mod[2] / 3.40147400e+01 + rho / 3.40147400e+01) * fwd_rxn_rates[16] + pres_mod[2] * (2.0 * (-mw_avg * fwd_rxn_rates[16] / 3.40147400e+01)));
  jac[118] += -1.0 * ((-mw_avg * pres_mod[3] / 3.40147400e+01 + rho / 3.40147400e+01) * fwd_rxn_rates[17] + pres_mod[3] * ((-mw_avg * fwd_rxn_rates[17] / 3.40147400e+01)));
  jac[118] += (2.0 * (-mw_avg * fwd_rxn_rates[18] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[18] / 3.40147400e+01));
  jac[118] += -1.0 * ((-mw_avg * fwd_rxn_rates[19] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[19] / 3.40147400e+01));
  jac[118] += (2.0 * (-mw_avg * fwd_rxn_rates[20] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[20] / 3.40147400e+01));
  jac[118] += -1.0 * ((-mw_avg * fwd_rxn_rates[21] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[21] / 3.40147400e+01));
  Pr = (m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * (exp(1.87346128e+01 - 2.16 * logT - (2.64088106e+02 / T)));
  Fcent = 5.0000e-01 * exp(-T / 1.0000e-30) + 5.0000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[118] += -1.0 * (pres_mod[8] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 3.40147400e+01 + rho / ((m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * 3.40147400e+01)) * fwd_rxn_rates[28] + pres_mod[8] * ((-mw_avg * fwd_rxn_rates[28] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[28] / 3.40147400e+01)));
  Pr = (m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * (exp(1.87345962e+01 - 2.159967 * logT - (2.64188750e+02 / T)));
  Fcent = 5.0000e-01 * exp(-T / 1.0000e-30) + 5.0000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[118] += (pres_mod[9] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 3.40147400e+01 + rho / ((m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * 3.40147400e+01)) * fwd_rxn_rates[29] + pres_mod[9] * ((-mw_avg * fwd_rxn_rates[29] / 3.40147400e+01)));
  jac[118] += ((-mw_avg * fwd_rxn_rates[30] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[30] / 3.40147400e+01));
  jac[118] += -1.0 * ((-mw_avg * fwd_rxn_rates[31] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[31] / 3.40147400e+01));
  jac[118] += ((-mw_avg * fwd_rxn_rates[34] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[34] / 3.40147400e+01));
  jac[118] += -1.0 * ((-mw_avg * fwd_rxn_rates[35] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[35] / 3.40147400e+01));
  jac[118] += ((-mw_avg * fwd_rxn_rates[36] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[36] / 3.40147400e+01));
  jac[118] += -1.0 * ((-mw_avg * fwd_rxn_rates[37] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[37] / 3.40147400e+01));
  jac[118] += (2.0 * (-mw_avg * fwd_rxn_rates[38] / 3.40147400e+01));
  jac[118] += -1.0 * ((-mw_avg * fwd_rxn_rates[39] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[39] / 3.40147400e+01 + exp(3.74200513e+01 - 0.23447 * logT - (2.53213594e+04 / T)) * (rho / 3.40147400e+01) * conc[5]));
  jac[118] += (2.0 * (-mw_avg * fwd_rxn_rates[40] / 3.40147400e+01));
  jac[118] += -1.0 * ((-mw_avg * fwd_rxn_rates[41] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[41] / 3.40147400e+01 + exp(2.93395801e+01 - 0.23447 * logT - (1.84720774e+04 / T)) * (rho / 3.40147400e+01) * conc[5]));
  jac[118] += sp_rates[5] * mw_avg / 3.40147400e+01;
  jac[118] *= 3.19988000e+01 / rho;

  //partial of omega_O2 wrt Y_N2;
  jac[132] = -1.0 * ((-mw_avg * fwd_rxn_rates[0] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[0] / 2.80134000e+01));
  jac[132] += ((-mw_avg * fwd_rxn_rates[1] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[1] / 2.80134000e+01));
  jac[132] += ((-mw_avg * pres_mod[2] / 2.80134000e+01 + rho / 2.80134000e+01) * fwd_rxn_rates[16] + pres_mod[2] * (2.0 * (-mw_avg * fwd_rxn_rates[16] / 2.80134000e+01)));
  jac[132] += -1.0 * ((-mw_avg * pres_mod[3] / 2.80134000e+01 + rho / 2.80134000e+01) * fwd_rxn_rates[17] + pres_mod[3] * ((-mw_avg * fwd_rxn_rates[17] / 2.80134000e+01)));
  jac[132] += (2.0 * (-mw_avg * fwd_rxn_rates[18] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[18] / 2.80134000e+01));
  jac[132] += -1.0 * ((-mw_avg * fwd_rxn_rates[19] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[19] / 2.80134000e+01));
  jac[132] += (2.0 * (-mw_avg * fwd_rxn_rates[20] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[20] / 2.80134000e+01));
  jac[132] += -1.0 * ((-mw_avg * fwd_rxn_rates[21] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[21] / 2.80134000e+01));
  Pr = (m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * (exp(1.87346128e+01 - 2.16 * logT - (2.64088106e+02 / T)));
  Fcent = 5.0000e-01 * exp(-T / 1.0000e-30) + 5.0000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[132] += -1.0 * (pres_mod[8] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 2.80134000e+01 + rho / ((m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * 2.80134000e+01)) * fwd_rxn_rates[28] + pres_mod[8] * ((-mw_avg * fwd_rxn_rates[28] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[28] / 2.80134000e+01)));
  Pr = (m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * (exp(1.87345962e+01 - 2.159967 * logT - (2.64188750e+02 / T)));
  Fcent = 5.0000e-01 * exp(-T / 1.0000e-30) + 5.0000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[132] += (pres_mod[9] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 2.80134000e+01 + rho / ((m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * 2.80134000e+01)) * fwd_rxn_rates[29] + pres_mod[9] * ((-mw_avg * fwd_rxn_rates[29] / 2.80134000e+01)));
  jac[132] += ((-mw_avg * fwd_rxn_rates[30] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[30] / 2.80134000e+01));
  jac[132] += -1.0 * ((-mw_avg * fwd_rxn_rates[31] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[31] / 2.80134000e+01));
  jac[132] += ((-mw_avg * fwd_rxn_rates[34] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[34] / 2.80134000e+01));
  jac[132] += -1.0 * ((-mw_avg * fwd_rxn_rates[35] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[35] / 2.80134000e+01));
  jac[132] += ((-mw_avg * fwd_rxn_rates[36] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[36] / 2.80134000e+01));
  jac[132] += -1.0 * ((-mw_avg * fwd_rxn_rates[37] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[37] / 2.80134000e+01));
  jac[132] += (2.0 * (-mw_avg * fwd_rxn_rates[38] / 2.80134000e+01));
  jac[132] += -1.0 * ((-mw_avg * fwd_rxn_rates[39] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[39] / 2.80134000e+01));
  jac[132] += (2.0 * (-mw_avg * fwd_rxn_rates[40] / 2.80134000e+01));
  jac[132] += -1.0 * ((-mw_avg * fwd_rxn_rates[41] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[41] / 2.80134000e+01));
  jac[132] += sp_rates[5] * mw_avg / 2.80134000e+01;
  jac[132] *= 3.19988000e+01 / rho;

  //partial of omega_O2 wrt Y_AR;
  jac[146] = -1.0 * ((-mw_avg * fwd_rxn_rates[0] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[0] / 3.99480000e+01));
  jac[146] += ((-mw_avg * fwd_rxn_rates[1] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[1] / 3.99480000e+01));
  jac[146] += ((-mw_avg * pres_mod[2] / 3.99480000e+01) * fwd_rxn_rates[16] + pres_mod[2] * (2.0 * (-mw_avg * fwd_rxn_rates[16] / 3.99480000e+01)));
  jac[146] += -1.0 * ((-mw_avg * pres_mod[3] / 3.99480000e+01) * fwd_rxn_rates[17] + pres_mod[3] * ((-mw_avg * fwd_rxn_rates[17] / 3.99480000e+01)));
  jac[146] += (2.0 * (-mw_avg * fwd_rxn_rates[18] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[18] / 3.99480000e+01 + exp(3.05680644e+01 - (-8.99751398e+02 / T)) * (rho / 3.99480000e+01) * conc[2] * conc[2]));
  jac[146] += -1.0 * ((-mw_avg * fwd_rxn_rates[19] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[19] / 3.99480000e+01 + exp(3.73613450e+01 - 0.43491 * logT - (5.93745344e+04 / T)) * (rho / 3.99480000e+01) * conc[5]));
  jac[146] += (2.0 * (-mw_avg * fwd_rxn_rates[20] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[20] / 3.99480000e+01));
  jac[146] += -1.0 * ((-mw_avg * fwd_rxn_rates[21] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[21] / 3.99480000e+01));
  Pr = (m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * (exp(1.87346128e+01 - 2.16 * logT - (2.64088106e+02 / T)));
  Fcent = 5.0000e-01 * exp(-T / 1.0000e-30) + 5.0000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[146] += -1.0 * (pres_mod[8] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 3.99480000e+01 + rho * 0.67 / ((m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * 3.99480000e+01)) * fwd_rxn_rates[28] + pres_mod[8] * ((-mw_avg * fwd_rxn_rates[28] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[28] / 3.99480000e+01)));
  Pr = (m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * (exp(1.87345962e+01 - 2.159967 * logT - (2.64188750e+02 / T)));
  Fcent = 5.0000e-01 * exp(-T / 1.0000e-30) + 5.0000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[146] += (pres_mod[9] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 3.99480000e+01 + rho * 0.67 / ((m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * 3.99480000e+01)) * fwd_rxn_rates[29] + pres_mod[9] * ((-mw_avg * fwd_rxn_rates[29] / 3.99480000e+01)));
  jac[146] += ((-mw_avg * fwd_rxn_rates[30] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[30] / 3.99480000e+01));
  jac[146] += -1.0 * ((-mw_avg * fwd_rxn_rates[31] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[31] / 3.99480000e+01));
  jac[146] += ((-mw_avg * fwd_rxn_rates[34] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[34] / 3.99480000e+01));
  jac[146] += -1.0 * ((-mw_avg * fwd_rxn_rates[35] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[35] / 3.99480000e+01));
  jac[146] += ((-mw_avg * fwd_rxn_rates[36] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[36] / 3.99480000e+01));
  jac[146] += -1.0 * ((-mw_avg * fwd_rxn_rates[37] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[37] / 3.99480000e+01));
  jac[146] += (2.0 * (-mw_avg * fwd_rxn_rates[38] / 3.99480000e+01));
  jac[146] += -1.0 * ((-mw_avg * fwd_rxn_rates[39] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[39] / 3.99480000e+01));
  jac[146] += (2.0 * (-mw_avg * fwd_rxn_rates[40] / 3.99480000e+01));
  jac[146] += -1.0 * ((-mw_avg * fwd_rxn_rates[41] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[41] / 3.99480000e+01));
  jac[146] += sp_rates[5] * mw_avg / 3.99480000e+01;
  jac[146] *= 3.19988000e+01 / rho;

  //partial of omega_O2 wrt Y_HE;
  jac[160] = -1.0 * ((-mw_avg * fwd_rxn_rates[0] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[0] / 4.00260000e+00));
  jac[160] += ((-mw_avg * fwd_rxn_rates[1] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[1] / 4.00260000e+00));
  jac[160] += ((-mw_avg * pres_mod[2] / 4.00260000e+00) * fwd_rxn_rates[16] + pres_mod[2] * (2.0 * (-mw_avg * fwd_rxn_rates[16] / 4.00260000e+00)));
  jac[160] += -1.0 * ((-mw_avg * pres_mod[3] / 4.00260000e+00) * fwd_rxn_rates[17] + pres_mod[3] * ((-mw_avg * fwd_rxn_rates[17] / 4.00260000e+00)));
  jac[160] += (2.0 * (-mw_avg * fwd_rxn_rates[18] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[18] / 4.00260000e+00));
  jac[160] += -1.0 * ((-mw_avg * fwd_rxn_rates[19] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[19] / 4.00260000e+00));
  jac[160] += (2.0 * (-mw_avg * fwd_rxn_rates[20] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[20] / 4.00260000e+00 + exp(3.05680644e+01 - (-8.99751398e+02 / T)) * (rho / 4.00260000e+00) * conc[2] * conc[2]));
  jac[160] += -1.0 * ((-mw_avg * fwd_rxn_rates[21] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[21] / 4.00260000e+00 + exp(3.73613450e+01 - 0.43491 * logT - (5.93745344e+04 / T)) * (rho / 4.00260000e+00) * conc[5]));
  Pr = (m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * (exp(1.87346128e+01 - 2.16 * logT - (2.64088106e+02 / T)));
  Fcent = 5.0000e-01 * exp(-T / 1.0000e-30) + 5.0000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[160] += -1.0 * (pres_mod[8] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 4.00260000e+00 + rho * 0.8 / ((m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * 4.00260000e+00)) * fwd_rxn_rates[28] + pres_mod[8] * ((-mw_avg * fwd_rxn_rates[28] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[28] / 4.00260000e+00)));
  Pr = (m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * (exp(1.87345962e+01 - 2.159967 * logT - (2.64188750e+02 / T)));
  Fcent = 5.0000e-01 * exp(-T / 1.0000e-30) + 5.0000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[160] += (pres_mod[9] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 4.00260000e+00 + rho * 0.8 / ((m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * 4.00260000e+00)) * fwd_rxn_rates[29] + pres_mod[9] * ((-mw_avg * fwd_rxn_rates[29] / 4.00260000e+00)));
  jac[160] += ((-mw_avg * fwd_rxn_rates[30] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[30] / 4.00260000e+00));
  jac[160] += -1.0 * ((-mw_avg * fwd_rxn_rates[31] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[31] / 4.00260000e+00));
  jac[160] += ((-mw_avg * fwd_rxn_rates[34] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[34] / 4.00260000e+00));
  jac[160] += -1.0 * ((-mw_avg * fwd_rxn_rates[35] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[35] / 4.00260000e+00));
  jac[160] += ((-mw_avg * fwd_rxn_rates[36] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[36] / 4.00260000e+00));
  jac[160] += -1.0 * ((-mw_avg * fwd_rxn_rates[37] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[37] / 4.00260000e+00));
  jac[160] += (2.0 * (-mw_avg * fwd_rxn_rates[38] / 4.00260000e+00));
  jac[160] += -1.0 * ((-mw_avg * fwd_rxn_rates[39] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[39] / 4.00260000e+00));
  jac[160] += (2.0 * (-mw_avg * fwd_rxn_rates[40] / 4.00260000e+00));
  jac[160] += -1.0 * ((-mw_avg * fwd_rxn_rates[41] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[41] / 4.00260000e+00));
  jac[160] += sp_rates[5] * mw_avg / 4.00260000e+00;
  jac[160] *= 3.19988000e+01 / rho;

  //partial of omega_O2 wrt Y_CO;
  jac[174] = -1.0 * ((-mw_avg * fwd_rxn_rates[0] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[0] / 2.80105500e+01));
  jac[174] += ((-mw_avg * fwd_rxn_rates[1] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[1] / 2.80105500e+01));
  jac[174] += ((-mw_avg * pres_mod[2] / 2.80105500e+01 + 1.9 * rho / 2.80105500e+01) * fwd_rxn_rates[16] + pres_mod[2] * (2.0 * (-mw_avg * fwd_rxn_rates[16] / 2.80105500e+01)));
  jac[174] += -1.0 * ((-mw_avg * pres_mod[3] / 2.80105500e+01 + 1.9 * rho / 2.80105500e+01) * fwd_rxn_rates[17] + pres_mod[3] * ((-mw_avg * fwd_rxn_rates[17] / 2.80105500e+01)));
  jac[174] += (2.0 * (-mw_avg * fwd_rxn_rates[18] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[18] / 2.80105500e+01));
  jac[174] += -1.0 * ((-mw_avg * fwd_rxn_rates[19] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[19] / 2.80105500e+01));
  jac[174] += (2.0 * (-mw_avg * fwd_rxn_rates[20] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[20] / 2.80105500e+01));
  jac[174] += -1.0 * ((-mw_avg * fwd_rxn_rates[21] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[21] / 2.80105500e+01));
  Pr = (m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * (exp(1.87346128e+01 - 2.16 * logT - (2.64088106e+02 / T)));
  Fcent = 5.0000e-01 * exp(-T / 1.0000e-30) + 5.0000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[174] += -1.0 * (pres_mod[8] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 2.80105500e+01 + rho * 1.9 / ((m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * 2.80105500e+01)) * fwd_rxn_rates[28] + pres_mod[8] * ((-mw_avg * fwd_rxn_rates[28] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[28] / 2.80105500e+01)));
  Pr = (m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * (exp(1.87345962e+01 - 2.159967 * logT - (2.64188750e+02 / T)));
  Fcent = 5.0000e-01 * exp(-T / 1.0000e-30) + 5.0000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[174] += (pres_mod[9] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 2.80105500e+01 + rho * 1.9 / ((m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * 2.80105500e+01)) * fwd_rxn_rates[29] + pres_mod[9] * ((-mw_avg * fwd_rxn_rates[29] / 2.80105500e+01)));
  jac[174] += ((-mw_avg * fwd_rxn_rates[30] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[30] / 2.80105500e+01));
  jac[174] += -1.0 * ((-mw_avg * fwd_rxn_rates[31] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[31] / 2.80105500e+01));
  jac[174] += ((-mw_avg * fwd_rxn_rates[34] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[34] / 2.80105500e+01));
  jac[174] += -1.0 * ((-mw_avg * fwd_rxn_rates[35] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[35] / 2.80105500e+01));
  jac[174] += ((-mw_avg * fwd_rxn_rates[36] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[36] / 2.80105500e+01));
  jac[174] += -1.0 * ((-mw_avg * fwd_rxn_rates[37] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[37] / 2.80105500e+01));
  jac[174] += (2.0 * (-mw_avg * fwd_rxn_rates[38] / 2.80105500e+01));
  jac[174] += -1.0 * ((-mw_avg * fwd_rxn_rates[39] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[39] / 2.80105500e+01));
  jac[174] += (2.0 * (-mw_avg * fwd_rxn_rates[40] / 2.80105500e+01));
  jac[174] += -1.0 * ((-mw_avg * fwd_rxn_rates[41] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[41] / 2.80105500e+01));
  jac[174] += sp_rates[5] * mw_avg / 2.80105500e+01;
  jac[174] *= 3.19988000e+01 / rho;

  //partial of omega_O2 wrt Y_CO2;
  jac[188] = -1.0 * ((-mw_avg * fwd_rxn_rates[0] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[0] / 4.40099500e+01));
  jac[188] += ((-mw_avg * fwd_rxn_rates[1] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[1] / 4.40099500e+01));
  jac[188] += ((-mw_avg * pres_mod[2] / 4.40099500e+01 + 3.8 * rho / 4.40099500e+01) * fwd_rxn_rates[16] + pres_mod[2] * (2.0 * (-mw_avg * fwd_rxn_rates[16] / 4.40099500e+01)));
  jac[188] += -1.0 * ((-mw_avg * pres_mod[3] / 4.40099500e+01 + 3.8 * rho / 4.40099500e+01) * fwd_rxn_rates[17] + pres_mod[3] * ((-mw_avg * fwd_rxn_rates[17] / 4.40099500e+01)));
  jac[188] += (2.0 * (-mw_avg * fwd_rxn_rates[18] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[18] / 4.40099500e+01));
  jac[188] += -1.0 * ((-mw_avg * fwd_rxn_rates[19] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[19] / 4.40099500e+01));
  jac[188] += (2.0 * (-mw_avg * fwd_rxn_rates[20] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[20] / 4.40099500e+01));
  jac[188] += -1.0 * ((-mw_avg * fwd_rxn_rates[21] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[21] / 4.40099500e+01));
  Pr = (m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * (exp(1.87346128e+01 - 2.16 * logT - (2.64088106e+02 / T)));
  Fcent = 5.0000e-01 * exp(-T / 1.0000e-30) + 5.0000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[188] += -1.0 * (pres_mod[8] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 4.40099500e+01 + rho * 3.8 / ((m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * 4.40099500e+01)) * fwd_rxn_rates[28] + pres_mod[8] * ((-mw_avg * fwd_rxn_rates[28] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[28] / 4.40099500e+01)));
  Pr = (m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * (exp(1.87345962e+01 - 2.159967 * logT - (2.64188750e+02 / T)));
  Fcent = 5.0000e-01 * exp(-T / 1.0000e-30) + 5.0000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[188] += (pres_mod[9] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 4.40099500e+01 + rho * 3.8 / ((m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * 4.40099500e+01)) * fwd_rxn_rates[29] + pres_mod[9] * ((-mw_avg * fwd_rxn_rates[29] / 4.40099500e+01)));
  jac[188] += ((-mw_avg * fwd_rxn_rates[30] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[30] / 4.40099500e+01));
  jac[188] += -1.0 * ((-mw_avg * fwd_rxn_rates[31] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[31] / 4.40099500e+01));
  jac[188] += ((-mw_avg * fwd_rxn_rates[34] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[34] / 4.40099500e+01));
  jac[188] += -1.0 * ((-mw_avg * fwd_rxn_rates[35] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[35] / 4.40099500e+01));
  jac[188] += ((-mw_avg * fwd_rxn_rates[36] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[36] / 4.40099500e+01));
  jac[188] += -1.0 * ((-mw_avg * fwd_rxn_rates[37] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[37] / 4.40099500e+01));
  jac[188] += (2.0 * (-mw_avg * fwd_rxn_rates[38] / 4.40099500e+01));
  jac[188] += -1.0 * ((-mw_avg * fwd_rxn_rates[39] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[39] / 4.40099500e+01));
  jac[188] += (2.0 * (-mw_avg * fwd_rxn_rates[40] / 4.40099500e+01));
  jac[188] += -1.0 * ((-mw_avg * fwd_rxn_rates[41] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[41] / 4.40099500e+01));
  jac[188] += sp_rates[5] * mw_avg / 4.40099500e+01;
  jac[188] *= 3.19988000e+01 / rho;

  //partial of omega_HO2 wrt T;
  Pr = (m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * (exp(1.87346128e+01 - 2.16 * logT - (2.64088106e+02 / T)));
  Fcent = 5.00000000e-01 * exp(-T / 1.00000000e-30) + 5.00000000e-01 * exp(T / 1.00000000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  lnF_AB = 2.0 * log(Fcent) * A / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)));
  jac[7] = (pres_mod[8]* (((-2.1600e+00 + (2.64088106e+02 / T) - 1.0) / (T * (1.0 + Pr))) + (((1.0 / (Fcent * (1.0 + A * A / (B * B)))) - lnF_AB * (-2.90977303e-01 * B + 5.10817170e-01 * A) / Fcent) * (-5.00000000e+29 * exp(-T / 1.00000000e-30) - 5.00000000e-31 * exp(-T / 1.00000000e+30))) - lnF_AB * (4.34294482e-01 * B + 6.08012275e-02 * A) * (-2.16000000e+00 + (2.64088106e+02 / T) - 1.0) / T) * fwd_rxn_rates[28] + (pres_mod[8] / T) * (fwd_rxn_rates[28] * (4.40000000e-01 + 1.0 - 2.0)));
  Pr = (m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * (exp(1.87345962e+01 - 2.159967 * logT - (2.64188750e+02 / T)));
  Fcent = 5.00000000e-01 * exp(-T / 1.00000000e-30) + 5.00000000e-01 * exp(T / 1.00000000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  lnF_AB = 2.0 * log(Fcent) * A / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)));
  jac[7] += -1.0 * (pres_mod[9]* (((-2.1600e+00 + (2.64188750e+02 / T) - 1.0) / (T * (1.0 + Pr))) + (((1.0 / (Fcent * (1.0 + A * A / (B * B)))) - lnF_AB * (-2.90977303e-01 * B + 5.10817170e-01 * A) / Fcent) * (-5.00000000e+29 * exp(-T / 1.00000000e-30) - 5.00000000e-31 * exp(-T / 1.00000000e+30))) - lnF_AB * (4.34294482e-01 * B + 6.08012275e-02 * A) * (-2.15996700e+00 + (2.64188750e+02 / T) - 1.0) / T) * fwd_rxn_rates[29] + (pres_mod[9] / T) * (fwd_rxn_rates[29] * (-4.94330000e-02 + (2.52182000e+04 / T) + 1.0 - 1.0)));
  jac[7] += -1.0 * ((1.0 / T) * (fwd_rxn_rates[30] * (2.09000000e+00 + (-7.30167382e+02 / T) + 1.0 - 2.0)));
  jac[7] += ((1.0 / T) * (fwd_rxn_rates[31] * (2.60280000e+00 + (2.65552467e+04 / T) + 1.0 - 2.0)));
  jac[7] += -1.0 * ((1.0 / T) * (fwd_rxn_rates[32] * ((1.48448916e+02 / T) + 1.0 - 2.0)));
  jac[7] += ((1.0 / T) * (fwd_rxn_rates[33] * (8.64090000e-01 + (1.83206092e+04 / T) + 1.0 - 2.0)));
  jac[7] += -1.0 * ((1.0 / T) * (fwd_rxn_rates[34] * (1.00000000e+00 + (-3.64293641e+02 / T) + 1.0 - 2.0)));
  jac[7] += ((1.0 / T) * (fwd_rxn_rates[35] * (1.45930000e+00 + (2.62487877e+04 / T) + 1.0 - 2.0)));
  jac[7] += -1.0 * ((1.0 / T) * (fwd_rxn_rates[36] * ((-2.50098683e+02 / T) + 1.0 - 2.0)));
  jac[7] += ((1.0 / T) * (fwd_rxn_rates[37] * (1.85740000e-01 + (3.48643603e+04 / T) + 1.0 - 2.0)));
  jac[7] += -2.0 * ((1.0 / T) * (fwd_rxn_rates[38] * ((6.02954209e+03 / T) + 1.0 - 2.0)));
  jac[7] += 2.0 * ((1.0 / T) * (fwd_rxn_rates[39] * (-2.34470000e-01 + (2.53213594e+04 / T) + 1.0 - 2.0)));
  jac[7] += -2.0 * ((1.0 / T) * (fwd_rxn_rates[40] * ((-8.19890914e+02 / T) + 1.0 - 2.0)));
  jac[7] += 2.0 * ((1.0 / T) * (fwd_rxn_rates[41] * (-2.34470000e-01 + (1.84720774e+04 / T) + 1.0 - 2.0)));
  jac[7] += ((1.0 / T) * (fwd_rxn_rates[46] * ((4.00057249e+03 / T) + 1.0 - 2.0)));
  jac[7] += -1.0 * ((1.0 / T) * (fwd_rxn_rates[47] * (7.47310000e-01 + (1.19941692e+04 / T) + 1.0 - 2.0)));
  jac[7] += ((1.0 / T) * (fwd_rxn_rates[48] * (2.00000000e+00 + (1.99777016e+03 / T) + 1.0 - 2.0)));
  jac[7] += -1.0 * ((1.0 / T) * (fwd_rxn_rates[49] * (2.69380000e+00 + (9.31906943e+03 / T) + 1.0 - 2.0)));
  jac[7] += ((1.0 / T) * (fwd_rxn_rates[50] * ((1.60022900e+02 / T) + 1.0 - 2.0)));
  jac[7] += -1.0 * ((1.0 / T) * (fwd_rxn_rates[51] * (4.20210000e-01 + (1.59826645e+04 / T) + 1.0 - 2.0)));
  jac[7] += ((1.0 / T) * (fwd_rxn_rates[52] * ((3.65838516e+03 / T) + 1.0 - 2.0)));
  jac[7] += -1.0 * ((1.0 / T) * (fwd_rxn_rates[53] * (4.20210000e-01 + (1.94810268e+04 / T) + 1.0 - 2.0)));
  jac[7] *= 3.30067700e+01 / rho;

  //partial of omega_HO2 wrt Y_H;
  Pr = (m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * (exp(1.87346128e+01 - 2.16 * logT - (2.64088106e+02 / T)));
  Fcent = 5.0000e-01 * exp(-T / 1.0000e-30) + 5.0000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[21] = (pres_mod[8] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 1.00797000e+00 + rho / ((m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * 1.00797000e+00)) * fwd_rxn_rates[28] + pres_mod[8] * ((-mw_avg * fwd_rxn_rates[28] / 1.00797000e+00 + exp(2.91680604e+01 + 0.44 * logT) * (rho / 1.00797000e+00) * conc[5]) + (-mw_avg * fwd_rxn_rates[28] / 1.00797000e+00)));
  Pr = (m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * (exp(1.87345962e+01 - 2.159967 * logT - (2.64188750e+02 / T)));
  Fcent = 5.0000e-01 * exp(-T / 1.0000e-30) + 5.0000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[21] += -1.0 * (pres_mod[9] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 1.00797000e+00 + rho / ((m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * 1.00797000e+00)) * fwd_rxn_rates[29] + pres_mod[9] * ((-mw_avg * fwd_rxn_rates[29] / 1.00797000e+00)));
  jac[21] += -1.0 * ((-mw_avg * fwd_rxn_rates[30] / 1.00797000e+00 + exp(1.48271115e+01 + 2.09 * logT - (-7.30167382e+02 / T)) * (rho / 1.00797000e+00) * conc[6]) + (-mw_avg * fwd_rxn_rates[30] / 1.00797000e+00));
  jac[21] += ((-mw_avg * fwd_rxn_rates[31] / 1.00797000e+00) + (-mw_avg * fwd_rxn_rates[31] / 1.00797000e+00));
  jac[21] += -1.0 * ((-mw_avg * fwd_rxn_rates[32] / 1.00797000e+00 + exp(3.18907389e+01 - (1.48448916e+02 / T)) * (rho / 1.00797000e+00) * conc[6]) + (-mw_avg * fwd_rxn_rates[32] / 1.00797000e+00));
  jac[21] += (2.0 * (-mw_avg * fwd_rxn_rates[33] / 1.00797000e+00));
  jac[21] += -1.0 * ((-mw_avg * fwd_rxn_rates[34] / 1.00797000e+00) + (-mw_avg * fwd_rxn_rates[34] / 1.00797000e+00));
  jac[21] += ((-mw_avg * fwd_rxn_rates[35] / 1.00797000e+00) + (-mw_avg * fwd_rxn_rates[35] / 1.00797000e+00));
  jac[21] += -1.0 * ((-mw_avg * fwd_rxn_rates[36] / 1.00797000e+00) + (-mw_avg * fwd_rxn_rates[36] / 1.00797000e+00));
  jac[21] += ((-mw_avg * fwd_rxn_rates[37] / 1.00797000e+00) + (-mw_avg * fwd_rxn_rates[37] / 1.00797000e+00));
  jac[21] += -2.0 * (2.0 * (-mw_avg * fwd_rxn_rates[38] / 1.00797000e+00));
  jac[21] += 2.0 * ((-mw_avg * fwd_rxn_rates[39] / 1.00797000e+00) + (-mw_avg * fwd_rxn_rates[39] / 1.00797000e+00));
  jac[21] += -2.0 * (2.0 * (-mw_avg * fwd_rxn_rates[40] / 1.00797000e+00));
  jac[21] += 2.0 * ((-mw_avg * fwd_rxn_rates[41] / 1.00797000e+00) + (-mw_avg * fwd_rxn_rates[41] / 1.00797000e+00));
  jac[21] += ((-mw_avg * fwd_rxn_rates[46] / 1.00797000e+00 + exp(3.15063801e+01 - (4.00057249e+03 / T)) * (rho / 1.00797000e+00) * conc[7]) + (-mw_avg * fwd_rxn_rates[46] / 1.00797000e+00));
  jac[21] += -1.0 * ((-mw_avg * fwd_rxn_rates[47] / 1.00797000e+00) + (-mw_avg * fwd_rxn_rates[47] / 1.00797000e+00));
  jac[21] += ((-mw_avg * fwd_rxn_rates[48] / 1.00797000e+00) + (-mw_avg * fwd_rxn_rates[48] / 1.00797000e+00));
  jac[21] += -1.0 * ((-mw_avg * fwd_rxn_rates[49] / 1.00797000e+00) + (-mw_avg * fwd_rxn_rates[49] / 1.00797000e+00));
  jac[21] += ((-mw_avg * fwd_rxn_rates[50] / 1.00797000e+00) + (-mw_avg * fwd_rxn_rates[50] / 1.00797000e+00));
  jac[21] += -1.0 * ((-mw_avg * fwd_rxn_rates[51] / 1.00797000e+00) + (-mw_avg * fwd_rxn_rates[51] / 1.00797000e+00));
  jac[21] += ((-mw_avg * fwd_rxn_rates[52] / 1.00797000e+00) + (-mw_avg * fwd_rxn_rates[52] / 1.00797000e+00));
  jac[21] += -1.0 * ((-mw_avg * fwd_rxn_rates[53] / 1.00797000e+00) + (-mw_avg * fwd_rxn_rates[53] / 1.00797000e+00));
  jac[21] += sp_rates[6] * mw_avg / 1.00797000e+00;
  jac[21] *= 3.30067700e+01 / rho;

  //partial of omega_HO2 wrt Y_H2;
  Pr = (m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * (exp(1.87346128e+01 - 2.16 * logT - (2.64088106e+02 / T)));
  Fcent = 5.0000e-01 * exp(-T / 1.0000e-30) + 5.0000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[35] = (pres_mod[8] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 2.01594000e+00 + rho * 2.0 / ((m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * 2.01594000e+00)) * fwd_rxn_rates[28] + pres_mod[8] * ((-mw_avg * fwd_rxn_rates[28] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[28] / 2.01594000e+00)));
  Pr = (m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * (exp(1.87345962e+01 - 2.159967 * logT - (2.64188750e+02 / T)));
  Fcent = 5.0000e-01 * exp(-T / 1.0000e-30) + 5.0000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[35] += -1.0 * (pres_mod[9] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 2.01594000e+00 + rho * 2.0 / ((m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * 2.01594000e+00)) * fwd_rxn_rates[29] + pres_mod[9] * ((-mw_avg * fwd_rxn_rates[29] / 2.01594000e+00)));
  jac[35] += -1.0 * ((-mw_avg * fwd_rxn_rates[30] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[30] / 2.01594000e+00));
  jac[35] += ((-mw_avg * fwd_rxn_rates[31] / 2.01594000e+00 + exp(1.17897235e+01 + 2.6028 * logT - (2.65552467e+04 / T)) * (rho / 2.01594000e+00) * conc[5]) + (-mw_avg * fwd_rxn_rates[31] / 2.01594000e+00));
  jac[35] += -1.0 * ((-mw_avg * fwd_rxn_rates[32] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[32] / 2.01594000e+00));
  jac[35] += (2.0 * (-mw_avg * fwd_rxn_rates[33] / 2.01594000e+00));
  jac[35] += -1.0 * ((-mw_avg * fwd_rxn_rates[34] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[34] / 2.01594000e+00));
  jac[35] += ((-mw_avg * fwd_rxn_rates[35] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[35] / 2.01594000e+00));
  jac[35] += -1.0 * ((-mw_avg * fwd_rxn_rates[36] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[36] / 2.01594000e+00));
  jac[35] += ((-mw_avg * fwd_rxn_rates[37] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[37] / 2.01594000e+00));
  jac[35] += -2.0 * (2.0 * (-mw_avg * fwd_rxn_rates[38] / 2.01594000e+00));
  jac[35] += 2.0 * ((-mw_avg * fwd_rxn_rates[39] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[39] / 2.01594000e+00));
  jac[35] += -2.0 * (2.0 * (-mw_avg * fwd_rxn_rates[40] / 2.01594000e+00));
  jac[35] += 2.0 * ((-mw_avg * fwd_rxn_rates[41] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[41] / 2.01594000e+00));
  jac[35] += ((-mw_avg * fwd_rxn_rates[46] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[46] / 2.01594000e+00));
  jac[35] += -1.0 * ((-mw_avg * fwd_rxn_rates[47] / 2.01594000e+00 + exp(2.47202181e+01 + 0.74731 * logT - (1.19941692e+04 / T)) * (rho / 2.01594000e+00) * conc[6]) + (-mw_avg * fwd_rxn_rates[47] / 2.01594000e+00));
  jac[35] += ((-mw_avg * fwd_rxn_rates[48] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[48] / 2.01594000e+00));
  jac[35] += -1.0 * ((-mw_avg * fwd_rxn_rates[49] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[49] / 2.01594000e+00));
  jac[35] += ((-mw_avg * fwd_rxn_rates[50] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[50] / 2.01594000e+00));
  jac[35] += -1.0 * ((-mw_avg * fwd_rxn_rates[51] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[51] / 2.01594000e+00));
  jac[35] += ((-mw_avg * fwd_rxn_rates[52] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[52] / 2.01594000e+00));
  jac[35] += -1.0 * ((-mw_avg * fwd_rxn_rates[53] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[53] / 2.01594000e+00));
  jac[35] += sp_rates[6] * mw_avg / 2.01594000e+00;
  jac[35] *= 3.30067700e+01 / rho;

  //partial of omega_HO2 wrt Y_O;
  Pr = (m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * (exp(1.87346128e+01 - 2.16 * logT - (2.64088106e+02 / T)));
  Fcent = 5.0000e-01 * exp(-T / 1.0000e-30) + 5.0000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[49] = (pres_mod[8] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 1.59994000e+01 + rho / ((m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * 1.59994000e+01)) * fwd_rxn_rates[28] + pres_mod[8] * ((-mw_avg * fwd_rxn_rates[28] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[28] / 1.59994000e+01)));
  Pr = (m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * (exp(1.87345962e+01 - 2.159967 * logT - (2.64188750e+02 / T)));
  Fcent = 5.0000e-01 * exp(-T / 1.0000e-30) + 5.0000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[49] += -1.0 * (pres_mod[9] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 1.59994000e+01 + rho / ((m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * 1.59994000e+01)) * fwd_rxn_rates[29] + pres_mod[9] * ((-mw_avg * fwd_rxn_rates[29] / 1.59994000e+01)));
  jac[49] += -1.0 * ((-mw_avg * fwd_rxn_rates[30] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[30] / 1.59994000e+01));
  jac[49] += ((-mw_avg * fwd_rxn_rates[31] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[31] / 1.59994000e+01));
  jac[49] += -1.0 * ((-mw_avg * fwd_rxn_rates[32] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[32] / 1.59994000e+01));
  jac[49] += (2.0 * (-mw_avg * fwd_rxn_rates[33] / 1.59994000e+01));
  jac[49] += -1.0 * ((-mw_avg * fwd_rxn_rates[34] / 1.59994000e+01 + exp(2.40731699e+01 + 1.0 * logT - (-3.64293641e+02 / T)) * (rho / 1.59994000e+01) * conc[6]) + (-mw_avg * fwd_rxn_rates[34] / 1.59994000e+01));
  jac[49] += ((-mw_avg * fwd_rxn_rates[35] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[35] / 1.59994000e+01));
  jac[49] += -1.0 * ((-mw_avg * fwd_rxn_rates[36] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[36] / 1.59994000e+01));
  jac[49] += ((-mw_avg * fwd_rxn_rates[37] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[37] / 1.59994000e+01));
  jac[49] += -2.0 * (2.0 * (-mw_avg * fwd_rxn_rates[38] / 1.59994000e+01));
  jac[49] += 2.0 * ((-mw_avg * fwd_rxn_rates[39] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[39] / 1.59994000e+01));
  jac[49] += -2.0 * (2.0 * (-mw_avg * fwd_rxn_rates[40] / 1.59994000e+01));
  jac[49] += 2.0 * ((-mw_avg * fwd_rxn_rates[41] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[41] / 1.59994000e+01));
  jac[49] += ((-mw_avg * fwd_rxn_rates[46] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[46] / 1.59994000e+01));
  jac[49] += -1.0 * ((-mw_avg * fwd_rxn_rates[47] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[47] / 1.59994000e+01));
  jac[49] += ((-mw_avg * fwd_rxn_rates[48] / 1.59994000e+01 + exp(1.60720517e+01 + 2.0 * logT - (1.99777016e+03 / T)) * (rho / 1.59994000e+01) * conc[7]) + (-mw_avg * fwd_rxn_rates[48] / 1.59994000e+01));
  jac[49] += -1.0 * ((-mw_avg * fwd_rxn_rates[49] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[49] / 1.59994000e+01));
  jac[49] += ((-mw_avg * fwd_rxn_rates[50] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[50] / 1.59994000e+01));
  jac[49] += -1.0 * ((-mw_avg * fwd_rxn_rates[51] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[51] / 1.59994000e+01));
  jac[49] += ((-mw_avg * fwd_rxn_rates[52] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[52] / 1.59994000e+01));
  jac[49] += -1.0 * ((-mw_avg * fwd_rxn_rates[53] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[53] / 1.59994000e+01));
  jac[49] += sp_rates[6] * mw_avg / 1.59994000e+01;
  jac[49] *= 3.30067700e+01 / rho;

  //partial of omega_HO2 wrt Y_OH;
  Pr = (m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * (exp(1.87346128e+01 - 2.16 * logT - (2.64088106e+02 / T)));
  Fcent = 5.0000e-01 * exp(-T / 1.0000e-30) + 5.0000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[63] = (pres_mod[8] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 1.70073700e+01 + rho / ((m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * 1.70073700e+01)) * fwd_rxn_rates[28] + pres_mod[8] * ((-mw_avg * fwd_rxn_rates[28] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[28] / 1.70073700e+01)));
  Pr = (m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * (exp(1.87345962e+01 - 2.159967 * logT - (2.64188750e+02 / T)));
  Fcent = 5.0000e-01 * exp(-T / 1.0000e-30) + 5.0000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[63] += -1.0 * (pres_mod[9] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 1.70073700e+01 + rho / ((m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * 1.70073700e+01)) * fwd_rxn_rates[29] + pres_mod[9] * ((-mw_avg * fwd_rxn_rates[29] / 1.70073700e+01)));
  jac[63] += -1.0 * ((-mw_avg * fwd_rxn_rates[30] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[30] / 1.70073700e+01));
  jac[63] += ((-mw_avg * fwd_rxn_rates[31] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[31] / 1.70073700e+01));
  jac[63] += -1.0 * ((-mw_avg * fwd_rxn_rates[32] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[32] / 1.70073700e+01));
  jac[63] += (2.0 * (-mw_avg * fwd_rxn_rates[33] / 1.70073700e+01 + exp(2.25013658e+01 + 0.86409 * logT - (1.83206092e+04 / T)) * (rho / 1.70073700e+01) * conc[3]));
  jac[63] += -1.0 * ((-mw_avg * fwd_rxn_rates[34] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[34] / 1.70073700e+01));
  jac[63] += ((-mw_avg * fwd_rxn_rates[35] / 1.70073700e+01 + exp(2.06516517e+01 + 1.4593 * logT - (2.62487877e+04 / T)) * (rho / 1.70073700e+01) * conc[5]) + (-mw_avg * fwd_rxn_rates[35] / 1.70073700e+01));
  jac[63] += -1.0 * ((-mw_avg * fwd_rxn_rates[36] / 1.70073700e+01 + exp(3.09948627e+01 - (-2.50098683e+02 / T)) * (rho / 1.70073700e+01) * conc[6]) + (-mw_avg * fwd_rxn_rates[36] / 1.70073700e+01));
  jac[63] += ((-mw_avg * fwd_rxn_rates[37] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[37] / 1.70073700e+01));
  jac[63] += -2.0 * (2.0 * (-mw_avg * fwd_rxn_rates[38] / 1.70073700e+01));
  jac[63] += 2.0 * ((-mw_avg * fwd_rxn_rates[39] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[39] / 1.70073700e+01));
  jac[63] += -2.0 * (2.0 * (-mw_avg * fwd_rxn_rates[40] / 1.70073700e+01));
  jac[63] += 2.0 * ((-mw_avg * fwd_rxn_rates[41] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[41] / 1.70073700e+01));
  jac[63] += ((-mw_avg * fwd_rxn_rates[46] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[46] / 1.70073700e+01));
  jac[63] += -1.0 * ((-mw_avg * fwd_rxn_rates[47] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[47] / 1.70073700e+01));
  jac[63] += ((-mw_avg * fwd_rxn_rates[48] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[48] / 1.70073700e+01));
  jac[63] += -1.0 * ((-mw_avg * fwd_rxn_rates[49] / 1.70073700e+01 + exp(8.90174786e+00 + 2.6938 * logT - (9.31906943e+03 / T)) * (rho / 1.70073700e+01) * conc[6]) + (-mw_avg * fwd_rxn_rates[49] / 1.70073700e+01));
  jac[63] += ((-mw_avg * fwd_rxn_rates[50] / 1.70073700e+01 + exp(2.81849062e+01 - (1.60022900e+02 / T)) * (rho / 1.70073700e+01) * conc[7]) + (-mw_avg * fwd_rxn_rates[50] / 1.70073700e+01));
  jac[63] += -1.0 * ((-mw_avg * fwd_rxn_rates[51] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[51] / 1.70073700e+01));
  jac[63] += ((-mw_avg * fwd_rxn_rates[52] / 1.70073700e+01 + exp(3.19604378e+01 - (3.65838516e+03 / T)) * (rho / 1.70073700e+01) * conc[7]) + (-mw_avg * fwd_rxn_rates[52] / 1.70073700e+01));
  jac[63] += -1.0 * ((-mw_avg * fwd_rxn_rates[53] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[53] / 1.70073700e+01));
  jac[63] += sp_rates[6] * mw_avg / 1.70073700e+01;
  jac[63] *= 3.30067700e+01 / rho;

  //partial of omega_HO2 wrt Y_H2O;
  Pr = (m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * (exp(1.87346128e+01 - 2.16 * logT - (2.64088106e+02 / T)));
  Fcent = 5.0000e-01 * exp(-T / 1.0000e-30) + 5.0000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[77] = (pres_mod[8] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 1.80153400e+01 + rho * 14.0 / ((m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * 1.80153400e+01)) * fwd_rxn_rates[28] + pres_mod[8] * ((-mw_avg * fwd_rxn_rates[28] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[28] / 1.80153400e+01)));
  Pr = (m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * (exp(1.87345962e+01 - 2.159967 * logT - (2.64188750e+02 / T)));
  Fcent = 5.0000e-01 * exp(-T / 1.0000e-30) + 5.0000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[77] += -1.0 * (pres_mod[9] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 1.80153400e+01 + rho * 14.0 / ((m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * 1.80153400e+01)) * fwd_rxn_rates[29] + pres_mod[9] * ((-mw_avg * fwd_rxn_rates[29] / 1.80153400e+01)));
  jac[77] += -1.0 * ((-mw_avg * fwd_rxn_rates[30] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[30] / 1.80153400e+01));
  jac[77] += ((-mw_avg * fwd_rxn_rates[31] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[31] / 1.80153400e+01));
  jac[77] += -1.0 * ((-mw_avg * fwd_rxn_rates[32] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[32] / 1.80153400e+01));
  jac[77] += (2.0 * (-mw_avg * fwd_rxn_rates[33] / 1.80153400e+01));
  jac[77] += -1.0 * ((-mw_avg * fwd_rxn_rates[34] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[34] / 1.80153400e+01));
  jac[77] += ((-mw_avg * fwd_rxn_rates[35] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[35] / 1.80153400e+01));
  jac[77] += -1.0 * ((-mw_avg * fwd_rxn_rates[36] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[36] / 1.80153400e+01));
  jac[77] += ((-mw_avg * fwd_rxn_rates[37] / 1.80153400e+01 + exp(3.21899484e+01 + 0.18574 * logT - (3.48643603e+04 / T)) * (rho / 1.80153400e+01) * conc[5]) + (-mw_avg * fwd_rxn_rates[37] / 1.80153400e+01));
  jac[77] += -2.0 * (2.0 * (-mw_avg * fwd_rxn_rates[38] / 1.80153400e+01));
  jac[77] += 2.0 * ((-mw_avg * fwd_rxn_rates[39] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[39] / 1.80153400e+01));
  jac[77] += -2.0 * (2.0 * (-mw_avg * fwd_rxn_rates[40] / 1.80153400e+01));
  jac[77] += 2.0 * ((-mw_avg * fwd_rxn_rates[41] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[41] / 1.80153400e+01));
  jac[77] += ((-mw_avg * fwd_rxn_rates[46] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[46] / 1.80153400e+01));
  jac[77] += -1.0 * ((-mw_avg * fwd_rxn_rates[47] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[47] / 1.80153400e+01));
  jac[77] += ((-mw_avg * fwd_rxn_rates[48] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[48] / 1.80153400e+01));
  jac[77] += -1.0 * ((-mw_avg * fwd_rxn_rates[49] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[49] / 1.80153400e+01));
  jac[77] += ((-mw_avg * fwd_rxn_rates[50] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[50] / 1.80153400e+01));
  jac[77] += -1.0 * ((-mw_avg * fwd_rxn_rates[51] / 1.80153400e+01 + exp(2.56312037e+01 + 0.42021 * logT - (1.59826645e+04 / T)) * (rho / 1.80153400e+01) * conc[6]) + (-mw_avg * fwd_rxn_rates[51] / 1.80153400e+01));
  jac[77] += ((-mw_avg * fwd_rxn_rates[52] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[52] / 1.80153400e+01));
  jac[77] += -1.0 * ((-mw_avg * fwd_rxn_rates[53] / 1.80153400e+01 + exp(2.94067528e+01 + 0.42021 * logT - (1.94810268e+04 / T)) * (rho / 1.80153400e+01) * conc[6]) + (-mw_avg * fwd_rxn_rates[53] / 1.80153400e+01));
  jac[77] += sp_rates[6] * mw_avg / 1.80153400e+01;
  jac[77] *= 3.30067700e+01 / rho;

  //partial of omega_HO2 wrt Y_O2;
  Pr = (m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * (exp(1.87346128e+01 - 2.16 * logT - (2.64088106e+02 / T)));
  Fcent = 5.0000e-01 * exp(-T / 1.0000e-30) + 5.0000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[91] = (pres_mod[8] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 3.19988000e+01 + rho * 0.78 / ((m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * 3.19988000e+01)) * fwd_rxn_rates[28] + pres_mod[8] * ((-mw_avg * fwd_rxn_rates[28] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[28] / 3.19988000e+01 + exp(2.91680604e+01 + 0.44 * logT) * (rho / 3.19988000e+01) * conc[0])));
  Pr = (m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * (exp(1.87345962e+01 - 2.159967 * logT - (2.64188750e+02 / T)));
  Fcent = 5.0000e-01 * exp(-T / 1.0000e-30) + 5.0000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[91] += -1.0 * (pres_mod[9] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 3.19988000e+01 + rho * 0.78 / ((m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * 3.19988000e+01)) * fwd_rxn_rates[29] + pres_mod[9] * ((-mw_avg * fwd_rxn_rates[29] / 3.19988000e+01)));
  jac[91] += -1.0 * ((-mw_avg * fwd_rxn_rates[30] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[30] / 3.19988000e+01));
  jac[91] += ((-mw_avg * fwd_rxn_rates[31] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[31] / 3.19988000e+01 + exp(1.17897235e+01 + 2.6028 * logT - (2.65552467e+04 / T)) * (rho / 3.19988000e+01) * conc[1]));
  jac[91] += -1.0 * ((-mw_avg * fwd_rxn_rates[32] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[32] / 3.19988000e+01));
  jac[91] += (2.0 * (-mw_avg * fwd_rxn_rates[33] / 3.19988000e+01));
  jac[91] += -1.0 * ((-mw_avg * fwd_rxn_rates[34] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[34] / 3.19988000e+01));
  jac[91] += ((-mw_avg * fwd_rxn_rates[35] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[35] / 3.19988000e+01 + exp(2.06516517e+01 + 1.4593 * logT - (2.62487877e+04 / T)) * (rho / 3.19988000e+01) * conc[3]));
  jac[91] += -1.0 * ((-mw_avg * fwd_rxn_rates[36] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[36] / 3.19988000e+01));
  jac[91] += ((-mw_avg * fwd_rxn_rates[37] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[37] / 3.19988000e+01 + exp(3.21899484e+01 + 0.18574 * logT - (3.48643603e+04 / T)) * (rho / 3.19988000e+01) * conc[4]));
  jac[91] += -2.0 * (2.0 * (-mw_avg * fwd_rxn_rates[38] / 3.19988000e+01));
  jac[91] += 2.0 * ((-mw_avg * fwd_rxn_rates[39] / 3.19988000e+01 + exp(3.74200513e+01 - 0.23447 * logT - (2.53213594e+04 / T)) * (rho / 3.19988000e+01) * conc[7]) + (-mw_avg * fwd_rxn_rates[39] / 3.19988000e+01));
  jac[91] += -2.0 * (2.0 * (-mw_avg * fwd_rxn_rates[40] / 3.19988000e+01));
  jac[91] += 2.0 * ((-mw_avg * fwd_rxn_rates[41] / 3.19988000e+01 + exp(2.93395801e+01 - 0.23447 * logT - (1.84720774e+04 / T)) * (rho / 3.19988000e+01) * conc[7]) + (-mw_avg * fwd_rxn_rates[41] / 3.19988000e+01));
  jac[91] += ((-mw_avg * fwd_rxn_rates[46] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[46] / 3.19988000e+01));
  jac[91] += -1.0 * ((-mw_avg * fwd_rxn_rates[47] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[47] / 3.19988000e+01));
  jac[91] += ((-mw_avg * fwd_rxn_rates[48] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[48] / 3.19988000e+01));
  jac[91] += -1.0 * ((-mw_avg * fwd_rxn_rates[49] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[49] / 3.19988000e+01));
  jac[91] += ((-mw_avg * fwd_rxn_rates[50] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[50] / 3.19988000e+01));
  jac[91] += -1.0 * ((-mw_avg * fwd_rxn_rates[51] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[51] / 3.19988000e+01));
  jac[91] += ((-mw_avg * fwd_rxn_rates[52] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[52] / 3.19988000e+01));
  jac[91] += -1.0 * ((-mw_avg * fwd_rxn_rates[53] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[53] / 3.19988000e+01));
  jac[91] += sp_rates[6] * mw_avg / 3.19988000e+01;
  jac[91] *= 3.30067700e+01 / rho;

  //partial of omega_HO2 wrt Y_HO2;
  Pr = (m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * (exp(1.87346128e+01 - 2.16 * logT - (2.64088106e+02 / T)));
  Fcent = 5.0000e-01 * exp(-T / 1.0000e-30) + 5.0000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[105] = (pres_mod[8] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 3.30067700e+01 + rho / ((m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * 3.30067700e+01)) * fwd_rxn_rates[28] + pres_mod[8] * ((-mw_avg * fwd_rxn_rates[28] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[28] / 3.30067700e+01)));
  Pr = (m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * (exp(1.87345962e+01 - 2.159967 * logT - (2.64188750e+02 / T)));
  Fcent = 5.0000e-01 * exp(-T / 1.0000e-30) + 5.0000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[105] += -1.0 * (pres_mod[9] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 3.30067700e+01 + rho / ((m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * 3.30067700e+01)) * fwd_rxn_rates[29] + pres_mod[9] * ((-mw_avg * fwd_rxn_rates[29] / 3.30067700e+01 + exp(3.34150001e+01 - 0.049433 * logT - (2.52182000e+04 / T)) * (rho / 3.30067700e+01))));
  jac[105] += -1.0 * ((-mw_avg * fwd_rxn_rates[30] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[30] / 3.30067700e+01 + exp(1.48271115e+01 + 2.09 * logT - (-7.30167382e+02 / T)) * (rho / 3.30067700e+01) * conc[0]));
  jac[105] += ((-mw_avg * fwd_rxn_rates[31] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[31] / 3.30067700e+01));
  jac[105] += -1.0 * ((-mw_avg * fwd_rxn_rates[32] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[32] / 3.30067700e+01 + exp(3.18907389e+01 - (1.48448916e+02 / T)) * (rho / 3.30067700e+01) * conc[0]));
  jac[105] += (2.0 * (-mw_avg * fwd_rxn_rates[33] / 3.30067700e+01));
  jac[105] += -1.0 * ((-mw_avg * fwd_rxn_rates[34] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[34] / 3.30067700e+01 + exp(2.40731699e+01 + 1.0 * logT - (-3.64293641e+02 / T)) * (rho / 3.30067700e+01) * conc[2]));
  jac[105] += ((-mw_avg * fwd_rxn_rates[35] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[35] / 3.30067700e+01));
  jac[105] += -1.0 * ((-mw_avg * fwd_rxn_rates[36] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[36] / 3.30067700e+01 + exp(3.09948627e+01 - (-2.50098683e+02 / T)) * (rho / 3.30067700e+01) * conc[3]));
  jac[105] += ((-mw_avg * fwd_rxn_rates[37] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[37] / 3.30067700e+01));
  jac[105] += -2.0 * (2.0 * (-mw_avg * fwd_rxn_rates[38] / 3.30067700e+01 + exp(3.36712758e+01 - (6.02954209e+03 / T)) * (rho / 3.30067700e+01) * conc[6]));
  jac[105] += 2.0 * ((-mw_avg * fwd_rxn_rates[39] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[39] / 3.30067700e+01));
  jac[105] += -2.0 * (2.0 * (-mw_avg * fwd_rxn_rates[40] / 3.30067700e+01 + exp(2.55908003e+01 - (-8.19890914e+02 / T)) * (rho / 3.30067700e+01) * conc[6]));
  jac[105] += 2.0 * ((-mw_avg * fwd_rxn_rates[41] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[41] / 3.30067700e+01));
  jac[105] += ((-mw_avg * fwd_rxn_rates[46] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[46] / 3.30067700e+01));
  jac[105] += -1.0 * ((-mw_avg * fwd_rxn_rates[47] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[47] / 3.30067700e+01 + exp(2.47202181e+01 + 0.74731 * logT - (1.19941692e+04 / T)) * (rho / 3.30067700e+01) * conc[1]));
  jac[105] += ((-mw_avg * fwd_rxn_rates[48] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[48] / 3.30067700e+01));
  jac[105] += -1.0 * ((-mw_avg * fwd_rxn_rates[49] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[49] / 3.30067700e+01 + exp(8.90174786e+00 + 2.6938 * logT - (9.31906943e+03 / T)) * (rho / 3.30067700e+01) * conc[3]));
  jac[105] += ((-mw_avg * fwd_rxn_rates[50] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[50] / 3.30067700e+01));
  jac[105] += -1.0 * ((-mw_avg * fwd_rxn_rates[51] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[51] / 3.30067700e+01 + exp(2.56312037e+01 + 0.42021 * logT - (1.59826645e+04 / T)) * (rho / 3.30067700e+01) * conc[4]));
  jac[105] += ((-mw_avg * fwd_rxn_rates[52] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[52] / 3.30067700e+01));
  jac[105] += -1.0 * ((-mw_avg * fwd_rxn_rates[53] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[53] / 3.30067700e+01 + exp(2.94067528e+01 + 0.42021 * logT - (1.94810268e+04 / T)) * (rho / 3.30067700e+01) * conc[4]));
  jac[105] += sp_rates[6] * mw_avg / 3.30067700e+01;
  jac[105] *= 3.30067700e+01 / rho;

  //partial of omega_HO2 wrt Y_H2O2;
  Pr = (m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * (exp(1.87346128e+01 - 2.16 * logT - (2.64088106e+02 / T)));
  Fcent = 5.0000e-01 * exp(-T / 1.0000e-30) + 5.0000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[119] = (pres_mod[8] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 3.40147400e+01 + rho / ((m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * 3.40147400e+01)) * fwd_rxn_rates[28] + pres_mod[8] * ((-mw_avg * fwd_rxn_rates[28] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[28] / 3.40147400e+01)));
  Pr = (m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * (exp(1.87345962e+01 - 2.159967 * logT - (2.64188750e+02 / T)));
  Fcent = 5.0000e-01 * exp(-T / 1.0000e-30) + 5.0000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[119] += -1.0 * (pres_mod[9] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 3.40147400e+01 + rho / ((m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * 3.40147400e+01)) * fwd_rxn_rates[29] + pres_mod[9] * ((-mw_avg * fwd_rxn_rates[29] / 3.40147400e+01)));
  jac[119] += -1.0 * ((-mw_avg * fwd_rxn_rates[30] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[30] / 3.40147400e+01));
  jac[119] += ((-mw_avg * fwd_rxn_rates[31] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[31] / 3.40147400e+01));
  jac[119] += -1.0 * ((-mw_avg * fwd_rxn_rates[32] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[32] / 3.40147400e+01));
  jac[119] += (2.0 * (-mw_avg * fwd_rxn_rates[33] / 3.40147400e+01));
  jac[119] += -1.0 * ((-mw_avg * fwd_rxn_rates[34] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[34] / 3.40147400e+01));
  jac[119] += ((-mw_avg * fwd_rxn_rates[35] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[35] / 3.40147400e+01));
  jac[119] += -1.0 * ((-mw_avg * fwd_rxn_rates[36] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[36] / 3.40147400e+01));
  jac[119] += ((-mw_avg * fwd_rxn_rates[37] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[37] / 3.40147400e+01));
  jac[119] += -2.0 * (2.0 * (-mw_avg * fwd_rxn_rates[38] / 3.40147400e+01));
  jac[119] += 2.0 * ((-mw_avg * fwd_rxn_rates[39] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[39] / 3.40147400e+01 + exp(3.74200513e+01 - 0.23447 * logT - (2.53213594e+04 / T)) * (rho / 3.40147400e+01) * conc[5]));
  jac[119] += -2.0 * (2.0 * (-mw_avg * fwd_rxn_rates[40] / 3.40147400e+01));
  jac[119] += 2.0 * ((-mw_avg * fwd_rxn_rates[41] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[41] / 3.40147400e+01 + exp(2.93395801e+01 - 0.23447 * logT - (1.84720774e+04 / T)) * (rho / 3.40147400e+01) * conc[5]));
  jac[119] += ((-mw_avg * fwd_rxn_rates[46] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[46] / 3.40147400e+01 + exp(3.15063801e+01 - (4.00057249e+03 / T)) * (rho / 3.40147400e+01) * conc[0]));
  jac[119] += -1.0 * ((-mw_avg * fwd_rxn_rates[47] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[47] / 3.40147400e+01));
  jac[119] += ((-mw_avg * fwd_rxn_rates[48] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[48] / 3.40147400e+01 + exp(1.60720517e+01 + 2.0 * logT - (1.99777016e+03 / T)) * (rho / 3.40147400e+01) * conc[2]));
  jac[119] += -1.0 * ((-mw_avg * fwd_rxn_rates[49] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[49] / 3.40147400e+01));
  jac[119] += ((-mw_avg * fwd_rxn_rates[50] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[50] / 3.40147400e+01 + exp(2.81849062e+01 - (1.60022900e+02 / T)) * (rho / 3.40147400e+01) * conc[3]));
  jac[119] += -1.0 * ((-mw_avg * fwd_rxn_rates[51] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[51] / 3.40147400e+01));
  jac[119] += ((-mw_avg * fwd_rxn_rates[52] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[52] / 3.40147400e+01 + exp(3.19604378e+01 - (3.65838516e+03 / T)) * (rho / 3.40147400e+01) * conc[3]));
  jac[119] += -1.0 * ((-mw_avg * fwd_rxn_rates[53] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[53] / 3.40147400e+01));
  jac[119] += sp_rates[6] * mw_avg / 3.40147400e+01;
  jac[119] *= 3.30067700e+01 / rho;

  //partial of omega_HO2 wrt Y_N2;
  Pr = (m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * (exp(1.87346128e+01 - 2.16 * logT - (2.64088106e+02 / T)));
  Fcent = 5.0000e-01 * exp(-T / 1.0000e-30) + 5.0000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[133] = (pres_mod[8] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 2.80134000e+01 + rho / ((m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * 2.80134000e+01)) * fwd_rxn_rates[28] + pres_mod[8] * ((-mw_avg * fwd_rxn_rates[28] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[28] / 2.80134000e+01)));
  Pr = (m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * (exp(1.87345962e+01 - 2.159967 * logT - (2.64188750e+02 / T)));
  Fcent = 5.0000e-01 * exp(-T / 1.0000e-30) + 5.0000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[133] += -1.0 * (pres_mod[9] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 2.80134000e+01 + rho / ((m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * 2.80134000e+01)) * fwd_rxn_rates[29] + pres_mod[9] * ((-mw_avg * fwd_rxn_rates[29] / 2.80134000e+01)));
  jac[133] += -1.0 * ((-mw_avg * fwd_rxn_rates[30] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[30] / 2.80134000e+01));
  jac[133] += ((-mw_avg * fwd_rxn_rates[31] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[31] / 2.80134000e+01));
  jac[133] += -1.0 * ((-mw_avg * fwd_rxn_rates[32] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[32] / 2.80134000e+01));
  jac[133] += (2.0 * (-mw_avg * fwd_rxn_rates[33] / 2.80134000e+01));
  jac[133] += -1.0 * ((-mw_avg * fwd_rxn_rates[34] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[34] / 2.80134000e+01));
  jac[133] += ((-mw_avg * fwd_rxn_rates[35] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[35] / 2.80134000e+01));
  jac[133] += -1.0 * ((-mw_avg * fwd_rxn_rates[36] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[36] / 2.80134000e+01));
  jac[133] += ((-mw_avg * fwd_rxn_rates[37] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[37] / 2.80134000e+01));
  jac[133] += -2.0 * (2.0 * (-mw_avg * fwd_rxn_rates[38] / 2.80134000e+01));
  jac[133] += 2.0 * ((-mw_avg * fwd_rxn_rates[39] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[39] / 2.80134000e+01));
  jac[133] += -2.0 * (2.0 * (-mw_avg * fwd_rxn_rates[40] / 2.80134000e+01));
  jac[133] += 2.0 * ((-mw_avg * fwd_rxn_rates[41] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[41] / 2.80134000e+01));
  jac[133] += ((-mw_avg * fwd_rxn_rates[46] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[46] / 2.80134000e+01));
  jac[133] += -1.0 * ((-mw_avg * fwd_rxn_rates[47] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[47] / 2.80134000e+01));
  jac[133] += ((-mw_avg * fwd_rxn_rates[48] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[48] / 2.80134000e+01));
  jac[133] += -1.0 * ((-mw_avg * fwd_rxn_rates[49] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[49] / 2.80134000e+01));
  jac[133] += ((-mw_avg * fwd_rxn_rates[50] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[50] / 2.80134000e+01));
  jac[133] += -1.0 * ((-mw_avg * fwd_rxn_rates[51] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[51] / 2.80134000e+01));
  jac[133] += ((-mw_avg * fwd_rxn_rates[52] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[52] / 2.80134000e+01));
  jac[133] += -1.0 * ((-mw_avg * fwd_rxn_rates[53] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[53] / 2.80134000e+01));
  jac[133] += sp_rates[6] * mw_avg / 2.80134000e+01;
  jac[133] *= 3.30067700e+01 / rho;

  //partial of omega_HO2 wrt Y_AR;
  Pr = (m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * (exp(1.87346128e+01 - 2.16 * logT - (2.64088106e+02 / T)));
  Fcent = 5.0000e-01 * exp(-T / 1.0000e-30) + 5.0000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[147] = (pres_mod[8] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 3.99480000e+01 + rho * 0.67 / ((m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * 3.99480000e+01)) * fwd_rxn_rates[28] + pres_mod[8] * ((-mw_avg * fwd_rxn_rates[28] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[28] / 3.99480000e+01)));
  Pr = (m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * (exp(1.87345962e+01 - 2.159967 * logT - (2.64188750e+02 / T)));
  Fcent = 5.0000e-01 * exp(-T / 1.0000e-30) + 5.0000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[147] += -1.0 * (pres_mod[9] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 3.99480000e+01 + rho * 0.67 / ((m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * 3.99480000e+01)) * fwd_rxn_rates[29] + pres_mod[9] * ((-mw_avg * fwd_rxn_rates[29] / 3.99480000e+01)));
  jac[147] += -1.0 * ((-mw_avg * fwd_rxn_rates[30] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[30] / 3.99480000e+01));
  jac[147] += ((-mw_avg * fwd_rxn_rates[31] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[31] / 3.99480000e+01));
  jac[147] += -1.0 * ((-mw_avg * fwd_rxn_rates[32] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[32] / 3.99480000e+01));
  jac[147] += (2.0 * (-mw_avg * fwd_rxn_rates[33] / 3.99480000e+01));
  jac[147] += -1.0 * ((-mw_avg * fwd_rxn_rates[34] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[34] / 3.99480000e+01));
  jac[147] += ((-mw_avg * fwd_rxn_rates[35] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[35] / 3.99480000e+01));
  jac[147] += -1.0 * ((-mw_avg * fwd_rxn_rates[36] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[36] / 3.99480000e+01));
  jac[147] += ((-mw_avg * fwd_rxn_rates[37] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[37] / 3.99480000e+01));
  jac[147] += -2.0 * (2.0 * (-mw_avg * fwd_rxn_rates[38] / 3.99480000e+01));
  jac[147] += 2.0 * ((-mw_avg * fwd_rxn_rates[39] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[39] / 3.99480000e+01));
  jac[147] += -2.0 * (2.0 * (-mw_avg * fwd_rxn_rates[40] / 3.99480000e+01));
  jac[147] += 2.0 * ((-mw_avg * fwd_rxn_rates[41] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[41] / 3.99480000e+01));
  jac[147] += ((-mw_avg * fwd_rxn_rates[46] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[46] / 3.99480000e+01));
  jac[147] += -1.0 * ((-mw_avg * fwd_rxn_rates[47] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[47] / 3.99480000e+01));
  jac[147] += ((-mw_avg * fwd_rxn_rates[48] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[48] / 3.99480000e+01));
  jac[147] += -1.0 * ((-mw_avg * fwd_rxn_rates[49] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[49] / 3.99480000e+01));
  jac[147] += ((-mw_avg * fwd_rxn_rates[50] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[50] / 3.99480000e+01));
  jac[147] += -1.0 * ((-mw_avg * fwd_rxn_rates[51] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[51] / 3.99480000e+01));
  jac[147] += ((-mw_avg * fwd_rxn_rates[52] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[52] / 3.99480000e+01));
  jac[147] += -1.0 * ((-mw_avg * fwd_rxn_rates[53] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[53] / 3.99480000e+01));
  jac[147] += sp_rates[6] * mw_avg / 3.99480000e+01;
  jac[147] *= 3.30067700e+01 / rho;

  //partial of omega_HO2 wrt Y_HE;
  Pr = (m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * (exp(1.87346128e+01 - 2.16 * logT - (2.64088106e+02 / T)));
  Fcent = 5.0000e-01 * exp(-T / 1.0000e-30) + 5.0000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[161] = (pres_mod[8] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 4.00260000e+00 + rho * 0.8 / ((m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * 4.00260000e+00)) * fwd_rxn_rates[28] + pres_mod[8] * ((-mw_avg * fwd_rxn_rates[28] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[28] / 4.00260000e+00)));
  Pr = (m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * (exp(1.87345962e+01 - 2.159967 * logT - (2.64188750e+02 / T)));
  Fcent = 5.0000e-01 * exp(-T / 1.0000e-30) + 5.0000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[161] += -1.0 * (pres_mod[9] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 4.00260000e+00 + rho * 0.8 / ((m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * 4.00260000e+00)) * fwd_rxn_rates[29] + pres_mod[9] * ((-mw_avg * fwd_rxn_rates[29] / 4.00260000e+00)));
  jac[161] += -1.0 * ((-mw_avg * fwd_rxn_rates[30] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[30] / 4.00260000e+00));
  jac[161] += ((-mw_avg * fwd_rxn_rates[31] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[31] / 4.00260000e+00));
  jac[161] += -1.0 * ((-mw_avg * fwd_rxn_rates[32] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[32] / 4.00260000e+00));
  jac[161] += (2.0 * (-mw_avg * fwd_rxn_rates[33] / 4.00260000e+00));
  jac[161] += -1.0 * ((-mw_avg * fwd_rxn_rates[34] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[34] / 4.00260000e+00));
  jac[161] += ((-mw_avg * fwd_rxn_rates[35] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[35] / 4.00260000e+00));
  jac[161] += -1.0 * ((-mw_avg * fwd_rxn_rates[36] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[36] / 4.00260000e+00));
  jac[161] += ((-mw_avg * fwd_rxn_rates[37] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[37] / 4.00260000e+00));
  jac[161] += -2.0 * (2.0 * (-mw_avg * fwd_rxn_rates[38] / 4.00260000e+00));
  jac[161] += 2.0 * ((-mw_avg * fwd_rxn_rates[39] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[39] / 4.00260000e+00));
  jac[161] += -2.0 * (2.0 * (-mw_avg * fwd_rxn_rates[40] / 4.00260000e+00));
  jac[161] += 2.0 * ((-mw_avg * fwd_rxn_rates[41] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[41] / 4.00260000e+00));
  jac[161] += ((-mw_avg * fwd_rxn_rates[46] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[46] / 4.00260000e+00));
  jac[161] += -1.0 * ((-mw_avg * fwd_rxn_rates[47] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[47] / 4.00260000e+00));
  jac[161] += ((-mw_avg * fwd_rxn_rates[48] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[48] / 4.00260000e+00));
  jac[161] += -1.0 * ((-mw_avg * fwd_rxn_rates[49] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[49] / 4.00260000e+00));
  jac[161] += ((-mw_avg * fwd_rxn_rates[50] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[50] / 4.00260000e+00));
  jac[161] += -1.0 * ((-mw_avg * fwd_rxn_rates[51] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[51] / 4.00260000e+00));
  jac[161] += ((-mw_avg * fwd_rxn_rates[52] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[52] / 4.00260000e+00));
  jac[161] += -1.0 * ((-mw_avg * fwd_rxn_rates[53] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[53] / 4.00260000e+00));
  jac[161] += sp_rates[6] * mw_avg / 4.00260000e+00;
  jac[161] *= 3.30067700e+01 / rho;

  //partial of omega_HO2 wrt Y_CO;
  Pr = (m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * (exp(1.87346128e+01 - 2.16 * logT - (2.64088106e+02 / T)));
  Fcent = 5.0000e-01 * exp(-T / 1.0000e-30) + 5.0000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[175] = (pres_mod[8] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 2.80105500e+01 + rho * 1.9 / ((m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * 2.80105500e+01)) * fwd_rxn_rates[28] + pres_mod[8] * ((-mw_avg * fwd_rxn_rates[28] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[28] / 2.80105500e+01)));
  Pr = (m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * (exp(1.87345962e+01 - 2.159967 * logT - (2.64188750e+02 / T)));
  Fcent = 5.0000e-01 * exp(-T / 1.0000e-30) + 5.0000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[175] += -1.0 * (pres_mod[9] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 2.80105500e+01 + rho * 1.9 / ((m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * 2.80105500e+01)) * fwd_rxn_rates[29] + pres_mod[9] * ((-mw_avg * fwd_rxn_rates[29] / 2.80105500e+01)));
  jac[175] += -1.0 * ((-mw_avg * fwd_rxn_rates[30] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[30] / 2.80105500e+01));
  jac[175] += ((-mw_avg * fwd_rxn_rates[31] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[31] / 2.80105500e+01));
  jac[175] += -1.0 * ((-mw_avg * fwd_rxn_rates[32] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[32] / 2.80105500e+01));
  jac[175] += (2.0 * (-mw_avg * fwd_rxn_rates[33] / 2.80105500e+01));
  jac[175] += -1.0 * ((-mw_avg * fwd_rxn_rates[34] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[34] / 2.80105500e+01));
  jac[175] += ((-mw_avg * fwd_rxn_rates[35] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[35] / 2.80105500e+01));
  jac[175] += -1.0 * ((-mw_avg * fwd_rxn_rates[36] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[36] / 2.80105500e+01));
  jac[175] += ((-mw_avg * fwd_rxn_rates[37] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[37] / 2.80105500e+01));
  jac[175] += -2.0 * (2.0 * (-mw_avg * fwd_rxn_rates[38] / 2.80105500e+01));
  jac[175] += 2.0 * ((-mw_avg * fwd_rxn_rates[39] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[39] / 2.80105500e+01));
  jac[175] += -2.0 * (2.0 * (-mw_avg * fwd_rxn_rates[40] / 2.80105500e+01));
  jac[175] += 2.0 * ((-mw_avg * fwd_rxn_rates[41] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[41] / 2.80105500e+01));
  jac[175] += ((-mw_avg * fwd_rxn_rates[46] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[46] / 2.80105500e+01));
  jac[175] += -1.0 * ((-mw_avg * fwd_rxn_rates[47] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[47] / 2.80105500e+01));
  jac[175] += ((-mw_avg * fwd_rxn_rates[48] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[48] / 2.80105500e+01));
  jac[175] += -1.0 * ((-mw_avg * fwd_rxn_rates[49] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[49] / 2.80105500e+01));
  jac[175] += ((-mw_avg * fwd_rxn_rates[50] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[50] / 2.80105500e+01));
  jac[175] += -1.0 * ((-mw_avg * fwd_rxn_rates[51] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[51] / 2.80105500e+01));
  jac[175] += ((-mw_avg * fwd_rxn_rates[52] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[52] / 2.80105500e+01));
  jac[175] += -1.0 * ((-mw_avg * fwd_rxn_rates[53] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[53] / 2.80105500e+01));
  jac[175] += sp_rates[6] * mw_avg / 2.80105500e+01;
  jac[175] *= 3.30067700e+01 / rho;

  //partial of omega_HO2 wrt Y_CO2;
  Pr = (m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * (exp(1.87346128e+01 - 2.16 * logT - (2.64088106e+02 / T)));
  Fcent = 5.0000e-01 * exp(-T / 1.0000e-30) + 5.0000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[189] = (pres_mod[8] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 4.40099500e+01 + rho * 3.8 / ((m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * 4.40099500e+01)) * fwd_rxn_rates[28] + pres_mod[8] * ((-mw_avg * fwd_rxn_rates[28] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[28] / 4.40099500e+01)));
  Pr = (m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * (exp(1.87345962e+01 - 2.159967 * logT - (2.64188750e+02 / T)));
  Fcent = 5.0000e-01 * exp(-T / 1.0000e-30) + 5.0000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[189] += -1.0 * (pres_mod[9] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 4.40099500e+01 + rho * 3.8 / ((m + 1.0 * conc[1] + 13.0 * conc[4] - 0.22 * conc[5] + 0.9 * conc[11] + 2.8 * conc[12] - 0.33 * conc[9] - 0.2 * conc[10]) * 4.40099500e+01)) * fwd_rxn_rates[29] + pres_mod[9] * ((-mw_avg * fwd_rxn_rates[29] / 4.40099500e+01)));
  jac[189] += -1.0 * ((-mw_avg * fwd_rxn_rates[30] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[30] / 4.40099500e+01));
  jac[189] += ((-mw_avg * fwd_rxn_rates[31] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[31] / 4.40099500e+01));
  jac[189] += -1.0 * ((-mw_avg * fwd_rxn_rates[32] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[32] / 4.40099500e+01));
  jac[189] += (2.0 * (-mw_avg * fwd_rxn_rates[33] / 4.40099500e+01));
  jac[189] += -1.0 * ((-mw_avg * fwd_rxn_rates[34] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[34] / 4.40099500e+01));
  jac[189] += ((-mw_avg * fwd_rxn_rates[35] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[35] / 4.40099500e+01));
  jac[189] += -1.0 * ((-mw_avg * fwd_rxn_rates[36] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[36] / 4.40099500e+01));
  jac[189] += ((-mw_avg * fwd_rxn_rates[37] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[37] / 4.40099500e+01));
  jac[189] += -2.0 * (2.0 * (-mw_avg * fwd_rxn_rates[38] / 4.40099500e+01));
  jac[189] += 2.0 * ((-mw_avg * fwd_rxn_rates[39] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[39] / 4.40099500e+01));
  jac[189] += -2.0 * (2.0 * (-mw_avg * fwd_rxn_rates[40] / 4.40099500e+01));
  jac[189] += 2.0 * ((-mw_avg * fwd_rxn_rates[41] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[41] / 4.40099500e+01));
  jac[189] += ((-mw_avg * fwd_rxn_rates[46] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[46] / 4.40099500e+01));
  jac[189] += -1.0 * ((-mw_avg * fwd_rxn_rates[47] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[47] / 4.40099500e+01));
  jac[189] += ((-mw_avg * fwd_rxn_rates[48] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[48] / 4.40099500e+01));
  jac[189] += -1.0 * ((-mw_avg * fwd_rxn_rates[49] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[49] / 4.40099500e+01));
  jac[189] += ((-mw_avg * fwd_rxn_rates[50] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[50] / 4.40099500e+01));
  jac[189] += -1.0 * ((-mw_avg * fwd_rxn_rates[51] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[51] / 4.40099500e+01));
  jac[189] += ((-mw_avg * fwd_rxn_rates[52] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[52] / 4.40099500e+01));
  jac[189] += -1.0 * ((-mw_avg * fwd_rxn_rates[53] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[53] / 4.40099500e+01));
  jac[189] += sp_rates[6] * mw_avg / 4.40099500e+01;
  jac[189] *= 3.30067700e+01 / rho;

  //partial of omega_H2O2 wrt T;
  jac[8] = ((1.0 / T) * (fwd_rxn_rates[38] * ((6.02954209e+03 / T) + 1.0 - 2.0)));
  jac[8] += -1.0 * ((1.0 / T) * (fwd_rxn_rates[39] * (-2.34470000e-01 + (2.53213594e+04 / T) + 1.0 - 2.0)));
  jac[8] += ((1.0 / T) * (fwd_rxn_rates[40] * ((-8.19890914e+02 / T) + 1.0 - 2.0)));
  jac[8] += -1.0 * ((1.0 / T) * (fwd_rxn_rates[41] * (-2.34470000e-01 + (1.84720774e+04 / T) + 1.0 - 2.0)));
  Pr = (m + 6.5 * conc[4] + 0.6 * conc[12] + 0.5 * conc[8] + 0.2 * conc[5] - 0.35 * conc[10] + 6.7 * conc[7] + 2.7 * conc[1] + 1.8 * conc[11]) * (exp(2.78501566e+01 - 3.2 * logT));
  Fcent = 5.70000000e-01 * exp(-T / 1.00000000e-30) + 4.30000000e-01 * exp(T / 1.00000000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  lnF_AB = 2.0 * log(Fcent) * A / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)));
  jac[8] += -1.0 * (pres_mod[10]* (((-3.2000e+00 + (0.00000000e+00 / T) - 1.0) / (T * (1.0 + Pr))) + (((1.0 / (Fcent * (1.0 + A * A / (B * B)))) - lnF_AB * (-2.90977303e-01 * B + 5.10817170e-01 * A) / Fcent) * (-5.70000000e+29 * exp(-T / 1.00000000e-30) - 4.30000000e-31 * exp(-T / 1.00000000e+30))) - lnF_AB * (4.34294482e-01 * B + 6.08012275e-02 * A) * (-3.20000000e+00 + (0.00000000e+00 / T) - 1.0) / T) * fwd_rxn_rates[42] + (pres_mod[10] / T) * (fwd_rxn_rates[42] * (9.00000000e-01 + (2.45313092e+04 / T) + 1.0 - 1.0)));
  Pr = (m + 6.5 * conc[4] + 0.6 * conc[12] + 0.5 * conc[8] + 0.2 * conc[5] - 0.35 * conc[10] + 6.7 * conc[7] + 2.7 * conc[1] + 1.8 * conc[11]) * (exp(2.78501521e+01 - 3.20001 * logT));
  Fcent = 5.70000000e-01 * exp(-T / 1.00000000e-30) + 4.30000000e-01 * exp(T / 1.00000000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  lnF_AB = 2.0 * log(Fcent) * A / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)));
  jac[8] += (pres_mod[11]* (((-3.2000e+00 + (0.00000000e+00 / T) - 1.0) / (T * (1.0 + Pr))) + (((1.0 / (Fcent * (1.0 + A * A / (B * B)))) - lnF_AB * (-2.90977303e-01 * B + 5.10817170e-01 * A) / Fcent) * (-5.70000000e+29 * exp(-T / 1.00000000e-30) - 4.30000000e-31 * exp(-T / 1.00000000e+30))) - lnF_AB * (4.34294482e-01 * B + 6.08012275e-02 * A) * (-3.20001000e+00 + (0.00000000e+00 / T) - 1.0) / T) * fwd_rxn_rates[43] + (pres_mod[11] / T) * (fwd_rxn_rates[43] * (2.48800000e+00 + (-1.80654783e+03 / T) + 1.0 - 2.0)));
  jac[8] += -1.0 * ((1.0 / T) * (fwd_rxn_rates[44] * ((1.99777016e+03 / T) + 1.0 - 2.0)));
  jac[8] += ((1.0 / T) * (fwd_rxn_rates[45] * (1.28430000e+00 + (3.59925720e+04 / T) + 1.0 - 2.0)));
  jac[8] += -1.0 * ((1.0 / T) * (fwd_rxn_rates[46] * ((4.00057249e+03 / T) + 1.0 - 2.0)));
  jac[8] += ((1.0 / T) * (fwd_rxn_rates[47] * (7.47310000e-01 + (1.19941692e+04 / T) + 1.0 - 2.0)));
  jac[8] += -1.0 * ((1.0 / T) * (fwd_rxn_rates[48] * (2.00000000e+00 + (1.99777016e+03 / T) + 1.0 - 2.0)));
  jac[8] += ((1.0 / T) * (fwd_rxn_rates[49] * (2.69380000e+00 + (9.31906943e+03 / T) + 1.0 - 2.0)));
  jac[8] += -1.0 * ((1.0 / T) * (fwd_rxn_rates[50] * ((1.60022900e+02 / T) + 1.0 - 2.0)));
  jac[8] += ((1.0 / T) * (fwd_rxn_rates[51] * (4.20210000e-01 + (1.59826645e+04 / T) + 1.0 - 2.0)));
  jac[8] += -1.0 * ((1.0 / T) * (fwd_rxn_rates[52] * ((3.65838516e+03 / T) + 1.0 - 2.0)));
  jac[8] += ((1.0 / T) * (fwd_rxn_rates[53] * (4.20210000e-01 + (1.94810268e+04 / T) + 1.0 - 2.0)));
  jac[8] *= 3.40147400e+01 / rho;

  //partial of omega_H2O2 wrt Y_H;
  jac[22] = (2.0 * (-mw_avg * fwd_rxn_rates[38] / 1.00797000e+00));
  jac[22] += -1.0 * ((-mw_avg * fwd_rxn_rates[39] / 1.00797000e+00) + (-mw_avg * fwd_rxn_rates[39] / 1.00797000e+00));
  jac[22] += (2.0 * (-mw_avg * fwd_rxn_rates[40] / 1.00797000e+00));
  jac[22] += -1.0 * ((-mw_avg * fwd_rxn_rates[41] / 1.00797000e+00) + (-mw_avg * fwd_rxn_rates[41] / 1.00797000e+00));
  Pr = (m + 6.5 * conc[4] + 0.6 * conc[12] + 0.5 * conc[8] + 0.2 * conc[5] - 0.35 * conc[10] + 6.7 * conc[7] + 2.7 * conc[1] + 1.8 * conc[11]) * (exp(2.78501566e+01 - 3.2 * logT));
  Fcent = 5.7000e-01 * exp(-T / 1.0000e-30) + 4.3000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[22] += -1.0 * (pres_mod[10] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 1.00797000e+00 + rho / ((m + 6.5 * conc[4] + 0.6 * conc[12] + 0.5 * conc[8] + 0.2 * conc[5] - 0.35 * conc[10] + 6.7 * conc[7] + 2.7 * conc[1] + 1.8 * conc[11]) * 1.00797000e+00)) * fwd_rxn_rates[42] + pres_mod[10] * ((-mw_avg * fwd_rxn_rates[42] / 1.00797000e+00)));
  Pr = (m + 6.5 * conc[4] + 0.6 * conc[12] + 0.5 * conc[8] + 0.2 * conc[5] - 0.35 * conc[10] + 6.7 * conc[7] + 2.7 * conc[1] + 1.8 * conc[11]) * (exp(2.78501521e+01 - 3.20001 * logT));
  Fcent = 5.7000e-01 * exp(-T / 1.0000e-30) + 4.3000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[22] += (pres_mod[11] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 1.00797000e+00 + rho / ((m + 6.5 * conc[4] + 0.6 * conc[12] + 0.5 * conc[8] + 0.2 * conc[5] - 0.35 * conc[10] + 6.7 * conc[7] + 2.7 * conc[1] + 1.8 * conc[11]) * 1.00797000e+00)) * fwd_rxn_rates[43] + pres_mod[11] * (2.0 * (-mw_avg * fwd_rxn_rates[43] / 1.00797000e+00)));
  jac[22] += -1.0 * ((-mw_avg * fwd_rxn_rates[44] / 1.00797000e+00 + exp(3.08132330e+01 - (1.99777016e+03 / T)) * (rho / 1.00797000e+00) * conc[7]) + (-mw_avg * fwd_rxn_rates[44] / 1.00797000e+00));
  jac[22] += ((-mw_avg * fwd_rxn_rates[45] / 1.00797000e+00) + (-mw_avg * fwd_rxn_rates[45] / 1.00797000e+00));
  jac[22] += -1.0 * ((-mw_avg * fwd_rxn_rates[46] / 1.00797000e+00 + exp(3.15063801e+01 - (4.00057249e+03 / T)) * (rho / 1.00797000e+00) * conc[7]) + (-mw_avg * fwd_rxn_rates[46] / 1.00797000e+00));
  jac[22] += ((-mw_avg * fwd_rxn_rates[47] / 1.00797000e+00) + (-mw_avg * fwd_rxn_rates[47] / 1.00797000e+00));
  jac[22] += -1.0 * ((-mw_avg * fwd_rxn_rates[48] / 1.00797000e+00) + (-mw_avg * fwd_rxn_rates[48] / 1.00797000e+00));
  jac[22] += ((-mw_avg * fwd_rxn_rates[49] / 1.00797000e+00) + (-mw_avg * fwd_rxn_rates[49] / 1.00797000e+00));
  jac[22] += -1.0 * ((-mw_avg * fwd_rxn_rates[50] / 1.00797000e+00) + (-mw_avg * fwd_rxn_rates[50] / 1.00797000e+00));
  jac[22] += ((-mw_avg * fwd_rxn_rates[51] / 1.00797000e+00) + (-mw_avg * fwd_rxn_rates[51] / 1.00797000e+00));
  jac[22] += -1.0 * ((-mw_avg * fwd_rxn_rates[52] / 1.00797000e+00) + (-mw_avg * fwd_rxn_rates[52] / 1.00797000e+00));
  jac[22] += ((-mw_avg * fwd_rxn_rates[53] / 1.00797000e+00) + (-mw_avg * fwd_rxn_rates[53] / 1.00797000e+00));
  jac[22] += sp_rates[7] * mw_avg / 1.00797000e+00;
  jac[22] *= 3.40147400e+01 / rho;

  //partial of omega_H2O2 wrt Y_H2;
  jac[36] = (2.0 * (-mw_avg * fwd_rxn_rates[38] / 2.01594000e+00));
  jac[36] += -1.0 * ((-mw_avg * fwd_rxn_rates[39] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[39] / 2.01594000e+00));
  jac[36] += (2.0 * (-mw_avg * fwd_rxn_rates[40] / 2.01594000e+00));
  jac[36] += -1.0 * ((-mw_avg * fwd_rxn_rates[41] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[41] / 2.01594000e+00));
  Pr = (m + 6.5 * conc[4] + 0.6 * conc[12] + 0.5 * conc[8] + 0.2 * conc[5] - 0.35 * conc[10] + 6.7 * conc[7] + 2.7 * conc[1] + 1.8 * conc[11]) * (exp(2.78501566e+01 - 3.2 * logT));
  Fcent = 5.7000e-01 * exp(-T / 1.0000e-30) + 4.3000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[36] += -1.0 * (pres_mod[10] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 2.01594000e+00 + rho * 3.7 / ((m + 6.5 * conc[4] + 0.6 * conc[12] + 0.5 * conc[8] + 0.2 * conc[5] - 0.35 * conc[10] + 6.7 * conc[7] + 2.7 * conc[1] + 1.8 * conc[11]) * 2.01594000e+00)) * fwd_rxn_rates[42] + pres_mod[10] * ((-mw_avg * fwd_rxn_rates[42] / 2.01594000e+00)));
  Pr = (m + 6.5 * conc[4] + 0.6 * conc[12] + 0.5 * conc[8] + 0.2 * conc[5] - 0.35 * conc[10] + 6.7 * conc[7] + 2.7 * conc[1] + 1.8 * conc[11]) * (exp(2.78501521e+01 - 3.20001 * logT));
  Fcent = 5.7000e-01 * exp(-T / 1.0000e-30) + 4.3000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[36] += (pres_mod[11] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 2.01594000e+00 + rho * 3.7 / ((m + 6.5 * conc[4] + 0.6 * conc[12] + 0.5 * conc[8] + 0.2 * conc[5] - 0.35 * conc[10] + 6.7 * conc[7] + 2.7 * conc[1] + 1.8 * conc[11]) * 2.01594000e+00)) * fwd_rxn_rates[43] + pres_mod[11] * (2.0 * (-mw_avg * fwd_rxn_rates[43] / 2.01594000e+00)));
  jac[36] += -1.0 * ((-mw_avg * fwd_rxn_rates[44] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[44] / 2.01594000e+00));
  jac[36] += ((-mw_avg * fwd_rxn_rates[45] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[45] / 2.01594000e+00));
  jac[36] += -1.0 * ((-mw_avg * fwd_rxn_rates[46] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[46] / 2.01594000e+00));
  jac[36] += ((-mw_avg * fwd_rxn_rates[47] / 2.01594000e+00 + exp(2.47202181e+01 + 0.74731 * logT - (1.19941692e+04 / T)) * (rho / 2.01594000e+00) * conc[6]) + (-mw_avg * fwd_rxn_rates[47] / 2.01594000e+00));
  jac[36] += -1.0 * ((-mw_avg * fwd_rxn_rates[48] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[48] / 2.01594000e+00));
  jac[36] += ((-mw_avg * fwd_rxn_rates[49] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[49] / 2.01594000e+00));
  jac[36] += -1.0 * ((-mw_avg * fwd_rxn_rates[50] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[50] / 2.01594000e+00));
  jac[36] += ((-mw_avg * fwd_rxn_rates[51] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[51] / 2.01594000e+00));
  jac[36] += -1.0 * ((-mw_avg * fwd_rxn_rates[52] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[52] / 2.01594000e+00));
  jac[36] += ((-mw_avg * fwd_rxn_rates[53] / 2.01594000e+00) + (-mw_avg * fwd_rxn_rates[53] / 2.01594000e+00));
  jac[36] += sp_rates[7] * mw_avg / 2.01594000e+00;
  jac[36] *= 3.40147400e+01 / rho;

  //partial of omega_H2O2 wrt Y_O;
  jac[50] = (2.0 * (-mw_avg * fwd_rxn_rates[38] / 1.59994000e+01));
  jac[50] += -1.0 * ((-mw_avg * fwd_rxn_rates[39] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[39] / 1.59994000e+01));
  jac[50] += (2.0 * (-mw_avg * fwd_rxn_rates[40] / 1.59994000e+01));
  jac[50] += -1.0 * ((-mw_avg * fwd_rxn_rates[41] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[41] / 1.59994000e+01));
  Pr = (m + 6.5 * conc[4] + 0.6 * conc[12] + 0.5 * conc[8] + 0.2 * conc[5] - 0.35 * conc[10] + 6.7 * conc[7] + 2.7 * conc[1] + 1.8 * conc[11]) * (exp(2.78501566e+01 - 3.2 * logT));
  Fcent = 5.7000e-01 * exp(-T / 1.0000e-30) + 4.3000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[50] += -1.0 * (pres_mod[10] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 1.59994000e+01 + rho / ((m + 6.5 * conc[4] + 0.6 * conc[12] + 0.5 * conc[8] + 0.2 * conc[5] - 0.35 * conc[10] + 6.7 * conc[7] + 2.7 * conc[1] + 1.8 * conc[11]) * 1.59994000e+01)) * fwd_rxn_rates[42] + pres_mod[10] * ((-mw_avg * fwd_rxn_rates[42] / 1.59994000e+01)));
  Pr = (m + 6.5 * conc[4] + 0.6 * conc[12] + 0.5 * conc[8] + 0.2 * conc[5] - 0.35 * conc[10] + 6.7 * conc[7] + 2.7 * conc[1] + 1.8 * conc[11]) * (exp(2.78501521e+01 - 3.20001 * logT));
  Fcent = 5.7000e-01 * exp(-T / 1.0000e-30) + 4.3000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[50] += (pres_mod[11] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 1.59994000e+01 + rho / ((m + 6.5 * conc[4] + 0.6 * conc[12] + 0.5 * conc[8] + 0.2 * conc[5] - 0.35 * conc[10] + 6.7 * conc[7] + 2.7 * conc[1] + 1.8 * conc[11]) * 1.59994000e+01)) * fwd_rxn_rates[43] + pres_mod[11] * (2.0 * (-mw_avg * fwd_rxn_rates[43] / 1.59994000e+01)));
  jac[50] += -1.0 * ((-mw_avg * fwd_rxn_rates[44] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[44] / 1.59994000e+01));
  jac[50] += ((-mw_avg * fwd_rxn_rates[45] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[45] / 1.59994000e+01));
  jac[50] += -1.0 * ((-mw_avg * fwd_rxn_rates[46] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[46] / 1.59994000e+01));
  jac[50] += ((-mw_avg * fwd_rxn_rates[47] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[47] / 1.59994000e+01));
  jac[50] += -1.0 * ((-mw_avg * fwd_rxn_rates[48] / 1.59994000e+01 + exp(1.60720517e+01 + 2.0 * logT - (1.99777016e+03 / T)) * (rho / 1.59994000e+01) * conc[7]) + (-mw_avg * fwd_rxn_rates[48] / 1.59994000e+01));
  jac[50] += ((-mw_avg * fwd_rxn_rates[49] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[49] / 1.59994000e+01));
  jac[50] += -1.0 * ((-mw_avg * fwd_rxn_rates[50] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[50] / 1.59994000e+01));
  jac[50] += ((-mw_avg * fwd_rxn_rates[51] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[51] / 1.59994000e+01));
  jac[50] += -1.0 * ((-mw_avg * fwd_rxn_rates[52] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[52] / 1.59994000e+01));
  jac[50] += ((-mw_avg * fwd_rxn_rates[53] / 1.59994000e+01) + (-mw_avg * fwd_rxn_rates[53] / 1.59994000e+01));
  jac[50] += sp_rates[7] * mw_avg / 1.59994000e+01;
  jac[50] *= 3.40147400e+01 / rho;

  //partial of omega_H2O2 wrt Y_OH;
  jac[64] = (2.0 * (-mw_avg * fwd_rxn_rates[38] / 1.70073700e+01));
  jac[64] += -1.0 * ((-mw_avg * fwd_rxn_rates[39] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[39] / 1.70073700e+01));
  jac[64] += (2.0 * (-mw_avg * fwd_rxn_rates[40] / 1.70073700e+01));
  jac[64] += -1.0 * ((-mw_avg * fwd_rxn_rates[41] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[41] / 1.70073700e+01));
  Pr = (m + 6.5 * conc[4] + 0.6 * conc[12] + 0.5 * conc[8] + 0.2 * conc[5] - 0.35 * conc[10] + 6.7 * conc[7] + 2.7 * conc[1] + 1.8 * conc[11]) * (exp(2.78501566e+01 - 3.2 * logT));
  Fcent = 5.7000e-01 * exp(-T / 1.0000e-30) + 4.3000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[64] += -1.0 * (pres_mod[10] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 1.70073700e+01 + rho / ((m + 6.5 * conc[4] + 0.6 * conc[12] + 0.5 * conc[8] + 0.2 * conc[5] - 0.35 * conc[10] + 6.7 * conc[7] + 2.7 * conc[1] + 1.8 * conc[11]) * 1.70073700e+01)) * fwd_rxn_rates[42] + pres_mod[10] * ((-mw_avg * fwd_rxn_rates[42] / 1.70073700e+01)));
  Pr = (m + 6.5 * conc[4] + 0.6 * conc[12] + 0.5 * conc[8] + 0.2 * conc[5] - 0.35 * conc[10] + 6.7 * conc[7] + 2.7 * conc[1] + 1.8 * conc[11]) * (exp(2.78501521e+01 - 3.20001 * logT));
  Fcent = 5.7000e-01 * exp(-T / 1.0000e-30) + 4.3000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[64] += (pres_mod[11] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 1.70073700e+01 + rho / ((m + 6.5 * conc[4] + 0.6 * conc[12] + 0.5 * conc[8] + 0.2 * conc[5] - 0.35 * conc[10] + 6.7 * conc[7] + 2.7 * conc[1] + 1.8 * conc[11]) * 1.70073700e+01)) * fwd_rxn_rates[43] + pres_mod[11] * (2.0 * (-mw_avg * fwd_rxn_rates[43] / 1.70073700e+01 + exp(1.09390890e+01 + 2.488 * logT - (-1.80654783e+03 / T)) * (rho / 1.70073700e+01) * conc[3])));
  jac[64] += -1.0 * ((-mw_avg * fwd_rxn_rates[44] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[44] / 1.70073700e+01));
  jac[64] += ((-mw_avg * fwd_rxn_rates[45] / 1.70073700e+01 + exp(1.88701627e+01 + 1.2843 * logT - (3.59925720e+04 / T)) * (rho / 1.70073700e+01) * conc[4]) + (-mw_avg * fwd_rxn_rates[45] / 1.70073700e+01));
  jac[64] += -1.0 * ((-mw_avg * fwd_rxn_rates[46] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[46] / 1.70073700e+01));
  jac[64] += ((-mw_avg * fwd_rxn_rates[47] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[47] / 1.70073700e+01));
  jac[64] += -1.0 * ((-mw_avg * fwd_rxn_rates[48] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[48] / 1.70073700e+01));
  jac[64] += ((-mw_avg * fwd_rxn_rates[49] / 1.70073700e+01 + exp(8.90174786e+00 + 2.6938 * logT - (9.31906943e+03 / T)) * (rho / 1.70073700e+01) * conc[6]) + (-mw_avg * fwd_rxn_rates[49] / 1.70073700e+01));
  jac[64] += -1.0 * ((-mw_avg * fwd_rxn_rates[50] / 1.70073700e+01 + exp(2.81849062e+01 - (1.60022900e+02 / T)) * (rho / 1.70073700e+01) * conc[7]) + (-mw_avg * fwd_rxn_rates[50] / 1.70073700e+01));
  jac[64] += ((-mw_avg * fwd_rxn_rates[51] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[51] / 1.70073700e+01));
  jac[64] += -1.0 * ((-mw_avg * fwd_rxn_rates[52] / 1.70073700e+01 + exp(3.19604378e+01 - (3.65838516e+03 / T)) * (rho / 1.70073700e+01) * conc[7]) + (-mw_avg * fwd_rxn_rates[52] / 1.70073700e+01));
  jac[64] += ((-mw_avg * fwd_rxn_rates[53] / 1.70073700e+01) + (-mw_avg * fwd_rxn_rates[53] / 1.70073700e+01));
  jac[64] += sp_rates[7] * mw_avg / 1.70073700e+01;
  jac[64] *= 3.40147400e+01 / rho;

  //partial of omega_H2O2 wrt Y_H2O;
  jac[78] = (2.0 * (-mw_avg * fwd_rxn_rates[38] / 1.80153400e+01));
  jac[78] += -1.0 * ((-mw_avg * fwd_rxn_rates[39] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[39] / 1.80153400e+01));
  jac[78] += (2.0 * (-mw_avg * fwd_rxn_rates[40] / 1.80153400e+01));
  jac[78] += -1.0 * ((-mw_avg * fwd_rxn_rates[41] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[41] / 1.80153400e+01));
  Pr = (m + 6.5 * conc[4] + 0.6 * conc[12] + 0.5 * conc[8] + 0.2 * conc[5] - 0.35 * conc[10] + 6.7 * conc[7] + 2.7 * conc[1] + 1.8 * conc[11]) * (exp(2.78501566e+01 - 3.2 * logT));
  Fcent = 5.7000e-01 * exp(-T / 1.0000e-30) + 4.3000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[78] += -1.0 * (pres_mod[10] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 1.80153400e+01 + rho * 7.5 / ((m + 6.5 * conc[4] + 0.6 * conc[12] + 0.5 * conc[8] + 0.2 * conc[5] - 0.35 * conc[10] + 6.7 * conc[7] + 2.7 * conc[1] + 1.8 * conc[11]) * 1.80153400e+01)) * fwd_rxn_rates[42] + pres_mod[10] * ((-mw_avg * fwd_rxn_rates[42] / 1.80153400e+01)));
  Pr = (m + 6.5 * conc[4] + 0.6 * conc[12] + 0.5 * conc[8] + 0.2 * conc[5] - 0.35 * conc[10] + 6.7 * conc[7] + 2.7 * conc[1] + 1.8 * conc[11]) * (exp(2.78501521e+01 - 3.20001 * logT));
  Fcent = 5.7000e-01 * exp(-T / 1.0000e-30) + 4.3000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[78] += (pres_mod[11] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 1.80153400e+01 + rho * 7.5 / ((m + 6.5 * conc[4] + 0.6 * conc[12] + 0.5 * conc[8] + 0.2 * conc[5] - 0.35 * conc[10] + 6.7 * conc[7] + 2.7 * conc[1] + 1.8 * conc[11]) * 1.80153400e+01)) * fwd_rxn_rates[43] + pres_mod[11] * (2.0 * (-mw_avg * fwd_rxn_rates[43] / 1.80153400e+01)));
  jac[78] += -1.0 * ((-mw_avg * fwd_rxn_rates[44] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[44] / 1.80153400e+01));
  jac[78] += ((-mw_avg * fwd_rxn_rates[45] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[45] / 1.80153400e+01 + exp(1.88701627e+01 + 1.2843 * logT - (3.59925720e+04 / T)) * (rho / 1.80153400e+01) * conc[3]));
  jac[78] += -1.0 * ((-mw_avg * fwd_rxn_rates[46] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[46] / 1.80153400e+01));
  jac[78] += ((-mw_avg * fwd_rxn_rates[47] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[47] / 1.80153400e+01));
  jac[78] += -1.0 * ((-mw_avg * fwd_rxn_rates[48] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[48] / 1.80153400e+01));
  jac[78] += ((-mw_avg * fwd_rxn_rates[49] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[49] / 1.80153400e+01));
  jac[78] += -1.0 * ((-mw_avg * fwd_rxn_rates[50] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[50] / 1.80153400e+01));
  jac[78] += ((-mw_avg * fwd_rxn_rates[51] / 1.80153400e+01 + exp(2.56312037e+01 + 0.42021 * logT - (1.59826645e+04 / T)) * (rho / 1.80153400e+01) * conc[6]) + (-mw_avg * fwd_rxn_rates[51] / 1.80153400e+01));
  jac[78] += -1.0 * ((-mw_avg * fwd_rxn_rates[52] / 1.80153400e+01) + (-mw_avg * fwd_rxn_rates[52] / 1.80153400e+01));
  jac[78] += ((-mw_avg * fwd_rxn_rates[53] / 1.80153400e+01 + exp(2.94067528e+01 + 0.42021 * logT - (1.94810268e+04 / T)) * (rho / 1.80153400e+01) * conc[6]) + (-mw_avg * fwd_rxn_rates[53] / 1.80153400e+01));
  jac[78] += sp_rates[7] * mw_avg / 1.80153400e+01;
  jac[78] *= 3.40147400e+01 / rho;

  //partial of omega_H2O2 wrt Y_O2;
  jac[92] = (2.0 * (-mw_avg * fwd_rxn_rates[38] / 3.19988000e+01));
  jac[92] += -1.0 * ((-mw_avg * fwd_rxn_rates[39] / 3.19988000e+01 + exp(3.74200513e+01 - 0.23447 * logT - (2.53213594e+04 / T)) * (rho / 3.19988000e+01) * conc[7]) + (-mw_avg * fwd_rxn_rates[39] / 3.19988000e+01));
  jac[92] += (2.0 * (-mw_avg * fwd_rxn_rates[40] / 3.19988000e+01));
  jac[92] += -1.0 * ((-mw_avg * fwd_rxn_rates[41] / 3.19988000e+01 + exp(2.93395801e+01 - 0.23447 * logT - (1.84720774e+04 / T)) * (rho / 3.19988000e+01) * conc[7]) + (-mw_avg * fwd_rxn_rates[41] / 3.19988000e+01));
  Pr = (m + 6.5 * conc[4] + 0.6 * conc[12] + 0.5 * conc[8] + 0.2 * conc[5] - 0.35 * conc[10] + 6.7 * conc[7] + 2.7 * conc[1] + 1.8 * conc[11]) * (exp(2.78501566e+01 - 3.2 * logT));
  Fcent = 5.7000e-01 * exp(-T / 1.0000e-30) + 4.3000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[92] += -1.0 * (pres_mod[10] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 3.19988000e+01 + rho * 1.2 / ((m + 6.5 * conc[4] + 0.6 * conc[12] + 0.5 * conc[8] + 0.2 * conc[5] - 0.35 * conc[10] + 6.7 * conc[7] + 2.7 * conc[1] + 1.8 * conc[11]) * 3.19988000e+01)) * fwd_rxn_rates[42] + pres_mod[10] * ((-mw_avg * fwd_rxn_rates[42] / 3.19988000e+01)));
  Pr = (m + 6.5 * conc[4] + 0.6 * conc[12] + 0.5 * conc[8] + 0.2 * conc[5] - 0.35 * conc[10] + 6.7 * conc[7] + 2.7 * conc[1] + 1.8 * conc[11]) * (exp(2.78501521e+01 - 3.20001 * logT));
  Fcent = 5.7000e-01 * exp(-T / 1.0000e-30) + 4.3000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[92] += (pres_mod[11] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 3.19988000e+01 + rho * 1.2 / ((m + 6.5 * conc[4] + 0.6 * conc[12] + 0.5 * conc[8] + 0.2 * conc[5] - 0.35 * conc[10] + 6.7 * conc[7] + 2.7 * conc[1] + 1.8 * conc[11]) * 3.19988000e+01)) * fwd_rxn_rates[43] + pres_mod[11] * (2.0 * (-mw_avg * fwd_rxn_rates[43] / 3.19988000e+01)));
  jac[92] += -1.0 * ((-mw_avg * fwd_rxn_rates[44] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[44] / 3.19988000e+01));
  jac[92] += ((-mw_avg * fwd_rxn_rates[45] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[45] / 3.19988000e+01));
  jac[92] += -1.0 * ((-mw_avg * fwd_rxn_rates[46] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[46] / 3.19988000e+01));
  jac[92] += ((-mw_avg * fwd_rxn_rates[47] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[47] / 3.19988000e+01));
  jac[92] += -1.0 * ((-mw_avg * fwd_rxn_rates[48] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[48] / 3.19988000e+01));
  jac[92] += ((-mw_avg * fwd_rxn_rates[49] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[49] / 3.19988000e+01));
  jac[92] += -1.0 * ((-mw_avg * fwd_rxn_rates[50] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[50] / 3.19988000e+01));
  jac[92] += ((-mw_avg * fwd_rxn_rates[51] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[51] / 3.19988000e+01));
  jac[92] += -1.0 * ((-mw_avg * fwd_rxn_rates[52] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[52] / 3.19988000e+01));
  jac[92] += ((-mw_avg * fwd_rxn_rates[53] / 3.19988000e+01) + (-mw_avg * fwd_rxn_rates[53] / 3.19988000e+01));
  jac[92] += sp_rates[7] * mw_avg / 3.19988000e+01;
  jac[92] *= 3.40147400e+01 / rho;

  //partial of omega_H2O2 wrt Y_HO2;
  jac[106] = (2.0 * (-mw_avg * fwd_rxn_rates[38] / 3.30067700e+01 + exp(3.36712758e+01 - (6.02954209e+03 / T)) * (rho / 3.30067700e+01) * conc[6]));
  jac[106] += -1.0 * ((-mw_avg * fwd_rxn_rates[39] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[39] / 3.30067700e+01));
  jac[106] += (2.0 * (-mw_avg * fwd_rxn_rates[40] / 3.30067700e+01 + exp(2.55908003e+01 - (-8.19890914e+02 / T)) * (rho / 3.30067700e+01) * conc[6]));
  jac[106] += -1.0 * ((-mw_avg * fwd_rxn_rates[41] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[41] / 3.30067700e+01));
  Pr = (m + 6.5 * conc[4] + 0.6 * conc[12] + 0.5 * conc[8] + 0.2 * conc[5] - 0.35 * conc[10] + 6.7 * conc[7] + 2.7 * conc[1] + 1.8 * conc[11]) * (exp(2.78501566e+01 - 3.2 * logT));
  Fcent = 5.7000e-01 * exp(-T / 1.0000e-30) + 4.3000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[106] += -1.0 * (pres_mod[10] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 3.30067700e+01 + rho / ((m + 6.5 * conc[4] + 0.6 * conc[12] + 0.5 * conc[8] + 0.2 * conc[5] - 0.35 * conc[10] + 6.7 * conc[7] + 2.7 * conc[1] + 1.8 * conc[11]) * 3.30067700e+01)) * fwd_rxn_rates[42] + pres_mod[10] * ((-mw_avg * fwd_rxn_rates[42] / 3.30067700e+01)));
  Pr = (m + 6.5 * conc[4] + 0.6 * conc[12] + 0.5 * conc[8] + 0.2 * conc[5] - 0.35 * conc[10] + 6.7 * conc[7] + 2.7 * conc[1] + 1.8 * conc[11]) * (exp(2.78501521e+01 - 3.20001 * logT));
  Fcent = 5.7000e-01 * exp(-T / 1.0000e-30) + 4.3000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[106] += (pres_mod[11] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 3.30067700e+01 + rho / ((m + 6.5 * conc[4] + 0.6 * conc[12] + 0.5 * conc[8] + 0.2 * conc[5] - 0.35 * conc[10] + 6.7 * conc[7] + 2.7 * conc[1] + 1.8 * conc[11]) * 3.30067700e+01)) * fwd_rxn_rates[43] + pres_mod[11] * (2.0 * (-mw_avg * fwd_rxn_rates[43] / 3.30067700e+01)));
  jac[106] += -1.0 * ((-mw_avg * fwd_rxn_rates[44] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[44] / 3.30067700e+01));
  jac[106] += ((-mw_avg * fwd_rxn_rates[45] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[45] / 3.30067700e+01));
  jac[106] += -1.0 * ((-mw_avg * fwd_rxn_rates[46] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[46] / 3.30067700e+01));
  jac[106] += ((-mw_avg * fwd_rxn_rates[47] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[47] / 3.30067700e+01 + exp(2.47202181e+01 + 0.74731 * logT - (1.19941692e+04 / T)) * (rho / 3.30067700e+01) * conc[1]));
  jac[106] += -1.0 * ((-mw_avg * fwd_rxn_rates[48] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[48] / 3.30067700e+01));
  jac[106] += ((-mw_avg * fwd_rxn_rates[49] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[49] / 3.30067700e+01 + exp(8.90174786e+00 + 2.6938 * logT - (9.31906943e+03 / T)) * (rho / 3.30067700e+01) * conc[3]));
  jac[106] += -1.0 * ((-mw_avg * fwd_rxn_rates[50] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[50] / 3.30067700e+01));
  jac[106] += ((-mw_avg * fwd_rxn_rates[51] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[51] / 3.30067700e+01 + exp(2.56312037e+01 + 0.42021 * logT - (1.59826645e+04 / T)) * (rho / 3.30067700e+01) * conc[4]));
  jac[106] += -1.0 * ((-mw_avg * fwd_rxn_rates[52] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[52] / 3.30067700e+01));
  jac[106] += ((-mw_avg * fwd_rxn_rates[53] / 3.30067700e+01) + (-mw_avg * fwd_rxn_rates[53] / 3.30067700e+01 + exp(2.94067528e+01 + 0.42021 * logT - (1.94810268e+04 / T)) * (rho / 3.30067700e+01) * conc[4]));
  jac[106] += sp_rates[7] * mw_avg / 3.30067700e+01;
  jac[106] *= 3.40147400e+01 / rho;

  //partial of omega_H2O2 wrt Y_H2O2;
  jac[120] = (2.0 * (-mw_avg * fwd_rxn_rates[38] / 3.40147400e+01));
  jac[120] += -1.0 * ((-mw_avg * fwd_rxn_rates[39] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[39] / 3.40147400e+01 + exp(3.74200513e+01 - 0.23447 * logT - (2.53213594e+04 / T)) * (rho / 3.40147400e+01) * conc[5]));
  jac[120] += (2.0 * (-mw_avg * fwd_rxn_rates[40] / 3.40147400e+01));
  jac[120] += -1.0 * ((-mw_avg * fwd_rxn_rates[41] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[41] / 3.40147400e+01 + exp(2.93395801e+01 - 0.23447 * logT - (1.84720774e+04 / T)) * (rho / 3.40147400e+01) * conc[5]));
  Pr = (m + 6.5 * conc[4] + 0.6 * conc[12] + 0.5 * conc[8] + 0.2 * conc[5] - 0.35 * conc[10] + 6.7 * conc[7] + 2.7 * conc[1] + 1.8 * conc[11]) * (exp(2.78501566e+01 - 3.2 * logT));
  Fcent = 5.7000e-01 * exp(-T / 1.0000e-30) + 4.3000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[120] += -1.0 * (pres_mod[10] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 3.40147400e+01 + rho * 7.7 / ((m + 6.5 * conc[4] + 0.6 * conc[12] + 0.5 * conc[8] + 0.2 * conc[5] - 0.35 * conc[10] + 6.7 * conc[7] + 2.7 * conc[1] + 1.8 * conc[11]) * 3.40147400e+01)) * fwd_rxn_rates[42] + pres_mod[10] * ((-mw_avg * fwd_rxn_rates[42] / 3.40147400e+01 + exp(2.83241683e+01 + 0.9 * logT - (2.45313092e+04 / T)) * (rho / 3.40147400e+01))));
  Pr = (m + 6.5 * conc[4] + 0.6 * conc[12] + 0.5 * conc[8] + 0.2 * conc[5] - 0.35 * conc[10] + 6.7 * conc[7] + 2.7 * conc[1] + 1.8 * conc[11]) * (exp(2.78501521e+01 - 3.20001 * logT));
  Fcent = 5.7000e-01 * exp(-T / 1.0000e-30) + 4.3000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[120] += (pres_mod[11] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 3.40147400e+01 + rho * 7.7 / ((m + 6.5 * conc[4] + 0.6 * conc[12] + 0.5 * conc[8] + 0.2 * conc[5] - 0.35 * conc[10] + 6.7 * conc[7] + 2.7 * conc[1] + 1.8 * conc[11]) * 3.40147400e+01)) * fwd_rxn_rates[43] + pres_mod[11] * (2.0 * (-mw_avg * fwd_rxn_rates[43] / 3.40147400e+01)));
  jac[120] += -1.0 * ((-mw_avg * fwd_rxn_rates[44] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[44] / 3.40147400e+01 + exp(3.08132330e+01 - (1.99777016e+03 / T)) * (rho / 3.40147400e+01) * conc[0]));
  jac[120] += ((-mw_avg * fwd_rxn_rates[45] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[45] / 3.40147400e+01));
  jac[120] += -1.0 * ((-mw_avg * fwd_rxn_rates[46] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[46] / 3.40147400e+01 + exp(3.15063801e+01 - (4.00057249e+03 / T)) * (rho / 3.40147400e+01) * conc[0]));
  jac[120] += ((-mw_avg * fwd_rxn_rates[47] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[47] / 3.40147400e+01));
  jac[120] += -1.0 * ((-mw_avg * fwd_rxn_rates[48] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[48] / 3.40147400e+01 + exp(1.60720517e+01 + 2.0 * logT - (1.99777016e+03 / T)) * (rho / 3.40147400e+01) * conc[2]));
  jac[120] += ((-mw_avg * fwd_rxn_rates[49] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[49] / 3.40147400e+01));
  jac[120] += -1.0 * ((-mw_avg * fwd_rxn_rates[50] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[50] / 3.40147400e+01 + exp(2.81849062e+01 - (1.60022900e+02 / T)) * (rho / 3.40147400e+01) * conc[3]));
  jac[120] += ((-mw_avg * fwd_rxn_rates[51] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[51] / 3.40147400e+01));
  jac[120] += -1.0 * ((-mw_avg * fwd_rxn_rates[52] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[52] / 3.40147400e+01 + exp(3.19604378e+01 - (3.65838516e+03 / T)) * (rho / 3.40147400e+01) * conc[3]));
  jac[120] += ((-mw_avg * fwd_rxn_rates[53] / 3.40147400e+01) + (-mw_avg * fwd_rxn_rates[53] / 3.40147400e+01));
  jac[120] += sp_rates[7] * mw_avg / 3.40147400e+01;
  jac[120] *= 3.40147400e+01 / rho;

  //partial of omega_H2O2 wrt Y_N2;
  jac[134] = (2.0 * (-mw_avg * fwd_rxn_rates[38] / 2.80134000e+01));
  jac[134] += -1.0 * ((-mw_avg * fwd_rxn_rates[39] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[39] / 2.80134000e+01));
  jac[134] += (2.0 * (-mw_avg * fwd_rxn_rates[40] / 2.80134000e+01));
  jac[134] += -1.0 * ((-mw_avg * fwd_rxn_rates[41] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[41] / 2.80134000e+01));
  Pr = (m + 6.5 * conc[4] + 0.6 * conc[12] + 0.5 * conc[8] + 0.2 * conc[5] - 0.35 * conc[10] + 6.7 * conc[7] + 2.7 * conc[1] + 1.8 * conc[11]) * (exp(2.78501566e+01 - 3.2 * logT));
  Fcent = 5.7000e-01 * exp(-T / 1.0000e-30) + 4.3000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[134] += -1.0 * (pres_mod[10] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 2.80134000e+01 + rho * 1.5 / ((m + 6.5 * conc[4] + 0.6 * conc[12] + 0.5 * conc[8] + 0.2 * conc[5] - 0.35 * conc[10] + 6.7 * conc[7] + 2.7 * conc[1] + 1.8 * conc[11]) * 2.80134000e+01)) * fwd_rxn_rates[42] + pres_mod[10] * ((-mw_avg * fwd_rxn_rates[42] / 2.80134000e+01)));
  Pr = (m + 6.5 * conc[4] + 0.6 * conc[12] + 0.5 * conc[8] + 0.2 * conc[5] - 0.35 * conc[10] + 6.7 * conc[7] + 2.7 * conc[1] + 1.8 * conc[11]) * (exp(2.78501521e+01 - 3.20001 * logT));
  Fcent = 5.7000e-01 * exp(-T / 1.0000e-30) + 4.3000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[134] += (pres_mod[11] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 2.80134000e+01 + rho * 1.5 / ((m + 6.5 * conc[4] + 0.6 * conc[12] + 0.5 * conc[8] + 0.2 * conc[5] - 0.35 * conc[10] + 6.7 * conc[7] + 2.7 * conc[1] + 1.8 * conc[11]) * 2.80134000e+01)) * fwd_rxn_rates[43] + pres_mod[11] * (2.0 * (-mw_avg * fwd_rxn_rates[43] / 2.80134000e+01)));
  jac[134] += -1.0 * ((-mw_avg * fwd_rxn_rates[44] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[44] / 2.80134000e+01));
  jac[134] += ((-mw_avg * fwd_rxn_rates[45] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[45] / 2.80134000e+01));
  jac[134] += -1.0 * ((-mw_avg * fwd_rxn_rates[46] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[46] / 2.80134000e+01));
  jac[134] += ((-mw_avg * fwd_rxn_rates[47] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[47] / 2.80134000e+01));
  jac[134] += -1.0 * ((-mw_avg * fwd_rxn_rates[48] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[48] / 2.80134000e+01));
  jac[134] += ((-mw_avg * fwd_rxn_rates[49] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[49] / 2.80134000e+01));
  jac[134] += -1.0 * ((-mw_avg * fwd_rxn_rates[50] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[50] / 2.80134000e+01));
  jac[134] += ((-mw_avg * fwd_rxn_rates[51] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[51] / 2.80134000e+01));
  jac[134] += -1.0 * ((-mw_avg * fwd_rxn_rates[52] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[52] / 2.80134000e+01));
  jac[134] += ((-mw_avg * fwd_rxn_rates[53] / 2.80134000e+01) + (-mw_avg * fwd_rxn_rates[53] / 2.80134000e+01));
  jac[134] += sp_rates[7] * mw_avg / 2.80134000e+01;
  jac[134] *= 3.40147400e+01 / rho;

  //partial of omega_H2O2 wrt Y_AR;
  jac[148] = (2.0 * (-mw_avg * fwd_rxn_rates[38] / 3.99480000e+01));
  jac[148] += -1.0 * ((-mw_avg * fwd_rxn_rates[39] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[39] / 3.99480000e+01));
  jac[148] += (2.0 * (-mw_avg * fwd_rxn_rates[40] / 3.99480000e+01));
  jac[148] += -1.0 * ((-mw_avg * fwd_rxn_rates[41] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[41] / 3.99480000e+01));
  Pr = (m + 6.5 * conc[4] + 0.6 * conc[12] + 0.5 * conc[8] + 0.2 * conc[5] - 0.35 * conc[10] + 6.7 * conc[7] + 2.7 * conc[1] + 1.8 * conc[11]) * (exp(2.78501566e+01 - 3.2 * logT));
  Fcent = 5.7000e-01 * exp(-T / 1.0000e-30) + 4.3000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[148] += -1.0 * (pres_mod[10] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 3.99480000e+01 + rho / ((m + 6.5 * conc[4] + 0.6 * conc[12] + 0.5 * conc[8] + 0.2 * conc[5] - 0.35 * conc[10] + 6.7 * conc[7] + 2.7 * conc[1] + 1.8 * conc[11]) * 3.99480000e+01)) * fwd_rxn_rates[42] + pres_mod[10] * ((-mw_avg * fwd_rxn_rates[42] / 3.99480000e+01)));
  Pr = (m + 6.5 * conc[4] + 0.6 * conc[12] + 0.5 * conc[8] + 0.2 * conc[5] - 0.35 * conc[10] + 6.7 * conc[7] + 2.7 * conc[1] + 1.8 * conc[11]) * (exp(2.78501521e+01 - 3.20001 * logT));
  Fcent = 5.7000e-01 * exp(-T / 1.0000e-30) + 4.3000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[148] += (pres_mod[11] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 3.99480000e+01 + rho / ((m + 6.5 * conc[4] + 0.6 * conc[12] + 0.5 * conc[8] + 0.2 * conc[5] - 0.35 * conc[10] + 6.7 * conc[7] + 2.7 * conc[1] + 1.8 * conc[11]) * 3.99480000e+01)) * fwd_rxn_rates[43] + pres_mod[11] * (2.0 * (-mw_avg * fwd_rxn_rates[43] / 3.99480000e+01)));
  jac[148] += -1.0 * ((-mw_avg * fwd_rxn_rates[44] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[44] / 3.99480000e+01));
  jac[148] += ((-mw_avg * fwd_rxn_rates[45] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[45] / 3.99480000e+01));
  jac[148] += -1.0 * ((-mw_avg * fwd_rxn_rates[46] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[46] / 3.99480000e+01));
  jac[148] += ((-mw_avg * fwd_rxn_rates[47] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[47] / 3.99480000e+01));
  jac[148] += -1.0 * ((-mw_avg * fwd_rxn_rates[48] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[48] / 3.99480000e+01));
  jac[148] += ((-mw_avg * fwd_rxn_rates[49] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[49] / 3.99480000e+01));
  jac[148] += -1.0 * ((-mw_avg * fwd_rxn_rates[50] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[50] / 3.99480000e+01));
  jac[148] += ((-mw_avg * fwd_rxn_rates[51] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[51] / 3.99480000e+01));
  jac[148] += -1.0 * ((-mw_avg * fwd_rxn_rates[52] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[52] / 3.99480000e+01));
  jac[148] += ((-mw_avg * fwd_rxn_rates[53] / 3.99480000e+01) + (-mw_avg * fwd_rxn_rates[53] / 3.99480000e+01));
  jac[148] += sp_rates[7] * mw_avg / 3.99480000e+01;
  jac[148] *= 3.40147400e+01 / rho;

  //partial of omega_H2O2 wrt Y_HE;
  jac[162] = (2.0 * (-mw_avg * fwd_rxn_rates[38] / 4.00260000e+00));
  jac[162] += -1.0 * ((-mw_avg * fwd_rxn_rates[39] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[39] / 4.00260000e+00));
  jac[162] += (2.0 * (-mw_avg * fwd_rxn_rates[40] / 4.00260000e+00));
  jac[162] += -1.0 * ((-mw_avg * fwd_rxn_rates[41] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[41] / 4.00260000e+00));
  Pr = (m + 6.5 * conc[4] + 0.6 * conc[12] + 0.5 * conc[8] + 0.2 * conc[5] - 0.35 * conc[10] + 6.7 * conc[7] + 2.7 * conc[1] + 1.8 * conc[11]) * (exp(2.78501566e+01 - 3.2 * logT));
  Fcent = 5.7000e-01 * exp(-T / 1.0000e-30) + 4.3000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[162] += -1.0 * (pres_mod[10] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 4.00260000e+00 + rho * 0.65 / ((m + 6.5 * conc[4] + 0.6 * conc[12] + 0.5 * conc[8] + 0.2 * conc[5] - 0.35 * conc[10] + 6.7 * conc[7] + 2.7 * conc[1] + 1.8 * conc[11]) * 4.00260000e+00)) * fwd_rxn_rates[42] + pres_mod[10] * ((-mw_avg * fwd_rxn_rates[42] / 4.00260000e+00)));
  Pr = (m + 6.5 * conc[4] + 0.6 * conc[12] + 0.5 * conc[8] + 0.2 * conc[5] - 0.35 * conc[10] + 6.7 * conc[7] + 2.7 * conc[1] + 1.8 * conc[11]) * (exp(2.78501521e+01 - 3.20001 * logT));
  Fcent = 5.7000e-01 * exp(-T / 1.0000e-30) + 4.3000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[162] += (pres_mod[11] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 4.00260000e+00 + rho * 0.65 / ((m + 6.5 * conc[4] + 0.6 * conc[12] + 0.5 * conc[8] + 0.2 * conc[5] - 0.35 * conc[10] + 6.7 * conc[7] + 2.7 * conc[1] + 1.8 * conc[11]) * 4.00260000e+00)) * fwd_rxn_rates[43] + pres_mod[11] * (2.0 * (-mw_avg * fwd_rxn_rates[43] / 4.00260000e+00)));
  jac[162] += -1.0 * ((-mw_avg * fwd_rxn_rates[44] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[44] / 4.00260000e+00));
  jac[162] += ((-mw_avg * fwd_rxn_rates[45] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[45] / 4.00260000e+00));
  jac[162] += -1.0 * ((-mw_avg * fwd_rxn_rates[46] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[46] / 4.00260000e+00));
  jac[162] += ((-mw_avg * fwd_rxn_rates[47] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[47] / 4.00260000e+00));
  jac[162] += -1.0 * ((-mw_avg * fwd_rxn_rates[48] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[48] / 4.00260000e+00));
  jac[162] += ((-mw_avg * fwd_rxn_rates[49] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[49] / 4.00260000e+00));
  jac[162] += -1.0 * ((-mw_avg * fwd_rxn_rates[50] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[50] / 4.00260000e+00));
  jac[162] += ((-mw_avg * fwd_rxn_rates[51] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[51] / 4.00260000e+00));
  jac[162] += -1.0 * ((-mw_avg * fwd_rxn_rates[52] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[52] / 4.00260000e+00));
  jac[162] += ((-mw_avg * fwd_rxn_rates[53] / 4.00260000e+00) + (-mw_avg * fwd_rxn_rates[53] / 4.00260000e+00));
  jac[162] += sp_rates[7] * mw_avg / 4.00260000e+00;
  jac[162] *= 3.40147400e+01 / rho;

  //partial of omega_H2O2 wrt Y_CO;
  jac[176] = (2.0 * (-mw_avg * fwd_rxn_rates[38] / 2.80105500e+01));
  jac[176] += -1.0 * ((-mw_avg * fwd_rxn_rates[39] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[39] / 2.80105500e+01));
  jac[176] += (2.0 * (-mw_avg * fwd_rxn_rates[40] / 2.80105500e+01));
  jac[176] += -1.0 * ((-mw_avg * fwd_rxn_rates[41] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[41] / 2.80105500e+01));
  Pr = (m + 6.5 * conc[4] + 0.6 * conc[12] + 0.5 * conc[8] + 0.2 * conc[5] - 0.35 * conc[10] + 6.7 * conc[7] + 2.7 * conc[1] + 1.8 * conc[11]) * (exp(2.78501566e+01 - 3.2 * logT));
  Fcent = 5.7000e-01 * exp(-T / 1.0000e-30) + 4.3000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[176] += -1.0 * (pres_mod[10] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 2.80105500e+01 + rho * 2.8 / ((m + 6.5 * conc[4] + 0.6 * conc[12] + 0.5 * conc[8] + 0.2 * conc[5] - 0.35 * conc[10] + 6.7 * conc[7] + 2.7 * conc[1] + 1.8 * conc[11]) * 2.80105500e+01)) * fwd_rxn_rates[42] + pres_mod[10] * ((-mw_avg * fwd_rxn_rates[42] / 2.80105500e+01)));
  Pr = (m + 6.5 * conc[4] + 0.6 * conc[12] + 0.5 * conc[8] + 0.2 * conc[5] - 0.35 * conc[10] + 6.7 * conc[7] + 2.7 * conc[1] + 1.8 * conc[11]) * (exp(2.78501521e+01 - 3.20001 * logT));
  Fcent = 5.7000e-01 * exp(-T / 1.0000e-30) + 4.3000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[176] += (pres_mod[11] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 2.80105500e+01 + rho * 2.8 / ((m + 6.5 * conc[4] + 0.6 * conc[12] + 0.5 * conc[8] + 0.2 * conc[5] - 0.35 * conc[10] + 6.7 * conc[7] + 2.7 * conc[1] + 1.8 * conc[11]) * 2.80105500e+01)) * fwd_rxn_rates[43] + pres_mod[11] * (2.0 * (-mw_avg * fwd_rxn_rates[43] / 2.80105500e+01)));
  jac[176] += -1.0 * ((-mw_avg * fwd_rxn_rates[44] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[44] / 2.80105500e+01));
  jac[176] += ((-mw_avg * fwd_rxn_rates[45] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[45] / 2.80105500e+01));
  jac[176] += -1.0 * ((-mw_avg * fwd_rxn_rates[46] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[46] / 2.80105500e+01));
  jac[176] += ((-mw_avg * fwd_rxn_rates[47] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[47] / 2.80105500e+01));
  jac[176] += -1.0 * ((-mw_avg * fwd_rxn_rates[48] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[48] / 2.80105500e+01));
  jac[176] += ((-mw_avg * fwd_rxn_rates[49] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[49] / 2.80105500e+01));
  jac[176] += -1.0 * ((-mw_avg * fwd_rxn_rates[50] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[50] / 2.80105500e+01));
  jac[176] += ((-mw_avg * fwd_rxn_rates[51] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[51] / 2.80105500e+01));
  jac[176] += -1.0 * ((-mw_avg * fwd_rxn_rates[52] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[52] / 2.80105500e+01));
  jac[176] += ((-mw_avg * fwd_rxn_rates[53] / 2.80105500e+01) + (-mw_avg * fwd_rxn_rates[53] / 2.80105500e+01));
  jac[176] += sp_rates[7] * mw_avg / 2.80105500e+01;
  jac[176] *= 3.40147400e+01 / rho;

  //partial of omega_H2O2 wrt Y_CO2;
  jac[190] = (2.0 * (-mw_avg * fwd_rxn_rates[38] / 4.40099500e+01));
  jac[190] += -1.0 * ((-mw_avg * fwd_rxn_rates[39] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[39] / 4.40099500e+01));
  jac[190] += (2.0 * (-mw_avg * fwd_rxn_rates[40] / 4.40099500e+01));
  jac[190] += -1.0 * ((-mw_avg * fwd_rxn_rates[41] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[41] / 4.40099500e+01));
  Pr = (m + 6.5 * conc[4] + 0.6 * conc[12] + 0.5 * conc[8] + 0.2 * conc[5] - 0.35 * conc[10] + 6.7 * conc[7] + 2.7 * conc[1] + 1.8 * conc[11]) * (exp(2.78501566e+01 - 3.2 * logT));
  Fcent = 5.7000e-01 * exp(-T / 1.0000e-30) + 4.3000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[190] += -1.0 * (pres_mod[10] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 4.40099500e+01 + rho * 1.6 / ((m + 6.5 * conc[4] + 0.6 * conc[12] + 0.5 * conc[8] + 0.2 * conc[5] - 0.35 * conc[10] + 6.7 * conc[7] + 2.7 * conc[1] + 1.8 * conc[11]) * 4.40099500e+01)) * fwd_rxn_rates[42] + pres_mod[10] * ((-mw_avg * fwd_rxn_rates[42] / 4.40099500e+01)));
  Pr = (m + 6.5 * conc[4] + 0.6 * conc[12] + 0.5 * conc[8] + 0.2 * conc[5] - 0.35 * conc[10] + 6.7 * conc[7] + 2.7 * conc[1] + 1.8 * conc[11]) * (exp(2.78501521e+01 - 3.20001 * logT));
  Fcent = 5.7000e-01 * exp(-T / 1.0000e-30) + 4.3000e-01 * exp(T / 1.0000e+30);
  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4;
  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr);
  jac[190] += (pres_mod[11] * ((1.0 / (1.0 + Pr)) - log(Fcent) * 2.0 * A * (B * 0.434294 + A * 0.0608012) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (-mw_avg / 4.40099500e+01 + rho * 1.6 / ((m + 6.5 * conc[4] + 0.6 * conc[12] + 0.5 * conc[8] + 0.2 * conc[5] - 0.35 * conc[10] + 6.7 * conc[7] + 2.7 * conc[1] + 1.8 * conc[11]) * 4.40099500e+01)) * fwd_rxn_rates[43] + pres_mod[11] * (2.0 * (-mw_avg * fwd_rxn_rates[43] / 4.40099500e+01)));
  jac[190] += -1.0 * ((-mw_avg * fwd_rxn_rates[44] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[44] / 4.40099500e+01));
  jac[190] += ((-mw_avg * fwd_rxn_rates[45] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[45] / 4.40099500e+01));
  jac[190] += -1.0 * ((-mw_avg * fwd_rxn_rates[46] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[46] / 4.40099500e+01));
  jac[190] += ((-mw_avg * fwd_rxn_rates[47] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[47] / 4.40099500e+01));
  jac[190] += -1.0 * ((-mw_avg * fwd_rxn_rates[48] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[48] / 4.40099500e+01));
  jac[190] += ((-mw_avg * fwd_rxn_rates[49] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[49] / 4.40099500e+01));
  jac[190] += -1.0 * ((-mw_avg * fwd_rxn_rates[50] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[50] / 4.40099500e+01));
  jac[190] += ((-mw_avg * fwd_rxn_rates[51] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[51] / 4.40099500e+01));
  jac[190] += -1.0 * ((-mw_avg * fwd_rxn_rates[52] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[52] / 4.40099500e+01));
  jac[190] += ((-mw_avg * fwd_rxn_rates[53] / 4.40099500e+01) + (-mw_avg * fwd_rxn_rates[53] / 4.40099500e+01));
  jac[190] += sp_rates[7] * mw_avg / 4.40099500e+01;
  jac[190] *= 3.40147400e+01 / rho;

  //partial of omega_N2 wrt T;
  jac[9] = 0.0;

  //partial of omega_N2 wrt Y_H;
  jac[23] = 0.0;

  //partial of omega_N2 wrt Y_H2;
  jac[37] = 0.0;

  //partial of omega_N2 wrt Y_O;
  jac[51] = 0.0;

  //partial of omega_N2 wrt Y_OH;
  jac[65] = 0.0;

  //partial of omega_N2 wrt Y_H2O;
  jac[79] = 0.0;

  //partial of omega_N2 wrt Y_O2;
  jac[93] = 0.0;

  //partial of omega_N2 wrt Y_HO2;
  jac[107] = 0.0;

  //partial of omega_N2 wrt Y_H2O2;
  jac[121] = 0.0;

  //partial of omega_N2 wrt Y_N2;
  jac[135] = 0.0;

  //partial of omega_N2 wrt Y_AR;
  jac[149] = 0.0;

  //partial of omega_N2 wrt Y_HE;
  jac[163] = 0.0;

  //partial of omega_N2 wrt Y_CO;
  jac[177] = 0.0;

  //partial of omega_N2 wrt Y_CO2;
  jac[191] = 0.0;

  //partial of omega_AR wrt T;
  jac[10] = 0.0;

  //partial of omega_AR wrt Y_H;
  jac[24] = 0.0;

  //partial of omega_AR wrt Y_H2;
  jac[38] = 0.0;

  //partial of omega_AR wrt Y_O;
  jac[52] = 0.0;

  //partial of omega_AR wrt Y_OH;
  jac[66] = 0.0;

  //partial of omega_AR wrt Y_H2O;
  jac[80] = 0.0;

  //partial of omega_AR wrt Y_O2;
  jac[94] = 0.0;

  //partial of omega_AR wrt Y_HO2;
  jac[108] = 0.0;

  //partial of omega_AR wrt Y_H2O2;
  jac[122] = 0.0;

  //partial of omega_AR wrt Y_N2;
  jac[136] = 0.0;

  //partial of omega_AR wrt Y_AR;
  jac[150] = 0.0;

  //partial of omega_AR wrt Y_HE;
  jac[164] = 0.0;

  //partial of omega_AR wrt Y_CO;
  jac[178] = 0.0;

  //partial of omega_AR wrt Y_CO2;
  jac[192] = 0.0;

  //partial of omega_HE wrt T;
  jac[11] = 0.0;

  //partial of omega_HE wrt Y_H;
  jac[25] = 0.0;

  //partial of omega_HE wrt Y_H2;
  jac[39] = 0.0;

  //partial of omega_HE wrt Y_O;
  jac[53] = 0.0;

  //partial of omega_HE wrt Y_OH;
  jac[67] = 0.0;

  //partial of omega_HE wrt Y_H2O;
  jac[81] = 0.0;

  //partial of omega_HE wrt Y_O2;
  jac[95] = 0.0;

  //partial of omega_HE wrt Y_HO2;
  jac[109] = 0.0;

  //partial of omega_HE wrt Y_H2O2;
  jac[123] = 0.0;

  //partial of omega_HE wrt Y_N2;
  jac[137] = 0.0;

  //partial of omega_HE wrt Y_AR;
  jac[151] = 0.0;

  //partial of omega_HE wrt Y_HE;
  jac[165] = 0.0;

  //partial of omega_HE wrt Y_CO;
  jac[179] = 0.0;

  //partial of omega_HE wrt Y_CO2;
  jac[193] = 0.0;

  //partial of omega_CO wrt T;
  jac[12] = 0.0;

  //partial of omega_CO wrt Y_H;
  jac[26] = 0.0;

  //partial of omega_CO wrt Y_H2;
  jac[40] = 0.0;

  //partial of omega_CO wrt Y_O;
  jac[54] = 0.0;

  //partial of omega_CO wrt Y_OH;
  jac[68] = 0.0;

  //partial of omega_CO wrt Y_H2O;
  jac[82] = 0.0;

  //partial of omega_CO wrt Y_O2;
  jac[96] = 0.0;

  //partial of omega_CO wrt Y_HO2;
  jac[110] = 0.0;

  //partial of omega_CO wrt Y_H2O2;
  jac[124] = 0.0;

  //partial of omega_CO wrt Y_N2;
  jac[138] = 0.0;

  //partial of omega_CO wrt Y_AR;
  jac[152] = 0.0;

  //partial of omega_CO wrt Y_HE;
  jac[166] = 0.0;

  //partial of omega_CO wrt Y_CO;
  jac[180] = 0.0;

  //partial of omega_CO wrt Y_CO2;
  jac[194] = 0.0;

  //partial of omega_CO2 wrt T;
  jac[13] = 0.0;

  //partial of omega_CO2 wrt Y_H;
  jac[27] = 0.0;

  //partial of omega_CO2 wrt Y_H2;
  jac[41] = 0.0;

  //partial of omega_CO2 wrt Y_O;
  jac[55] = 0.0;

  //partial of omega_CO2 wrt Y_OH;
  jac[69] = 0.0;

  //partial of omega_CO2 wrt Y_H2O;
  jac[83] = 0.0;

  //partial of omega_CO2 wrt Y_O2;
  jac[97] = 0.0;

  //partial of omega_CO2 wrt Y_HO2;
  jac[111] = 0.0;

  //partial of omega_CO2 wrt Y_H2O2;
  jac[125] = 0.0;

  //partial of omega_CO2 wrt Y_N2;
  jac[139] = 0.0;

  //partial of omega_CO2 wrt Y_AR;
  jac[153] = 0.0;

  //partial of omega_CO2 wrt Y_HE;
  jac[167] = 0.0;

  //partial of omega_CO2 wrt Y_CO;
  jac[181] = 0.0;

  //partial of omega_CO2 wrt Y_CO2;
  jac[195] = 0.0;


  // species enthalpies
  Real h[13];
  eval_h(T, h);

  // species specific heats
  Real cp[13];
  eval_cp(T, cp);

  // average specific heat
  register Real cp_avg;
  cp_avg = (y[1] * cp[0]) + (y[2] * cp[1]) + (y[3] * cp[2]) + (y[4] * cp[3])
      + (y[5] * cp[4]) + (y[6] * cp[5]) + (y[7] * cp[6]) + (y[8] * cp[7])
      + (y[9] * cp[8]) + (y[10] * cp[9]) + (y[11] * cp[10]) + (y[12] * cp[11])
      + (y[13] * cp[12]);
  // sum of enthalpy * species rate * molecular weight for all species
  register Real sum_hwW;

  sum_hwW = (h[0] * sp_rates[0] * 1.00797) + (h[1] * sp_rates[1] * 2.01594)
      + (h[2] * sp_rates[2] * 15.9994) + (h[3] * sp_rates[3] * 17.0074)
      + (h[4] * sp_rates[4] * 18.0153) + (h[5] * sp_rates[5] * 31.9988)
      + (h[6] * sp_rates[6] * 33.0068) + (h[7] * sp_rates[7] * 34.0147)
      + (h[8] * sp_rates[8] * 28.0134) + (h[9] * sp_rates[9] * 39.948) + (h[10] * sp_rates[10] * 4.0026)
      + (h[11] * sp_rates[11] * 28.0106) + (h[12] * sp_rates[12] * 44.01);

  //partial of dT wrt T;
  jac[0] = 0.0;
  if (T <= 1000.0) {
    jac[0] += y[1] * 8.24876732e+07 * (0.00000000e+00 + T * (0.00000000e+00 + T * (0.00000000e+00 + 0.00000000e+00 * T)));
  } else {
    jac[0] += y[1] * 8.24876732e+07 * (0.00000000e+00 + T * (0.00000000e+00 + T * (0.00000000e+00 + 0.00000000e+00 * T)));
  }

  if (T <= 1000.0) {
    jac[0] += y[2] * 4.12438366e+07 * (8.24944200e-04 + T * (-1.62860300e-06 + T * (-2.84263020e-10 + 1.65394880e-12 * T)));
  } else {
    jac[0] += y[2] * 4.12438366e+07 * (7.00064400e-04 + T * (-1.12676580e-07 + T * (-2.76947340e-11 + 6.33100800e-15 * T)));
  }

  if (T <= 1000.0) {
    jac[0] += y[3] * 5.19676363e+06 * (-1.63816600e-03 + T * (4.84206400e-06 + T * (-4.80852900e-09 + 1.55627840e-12 * T)));
  } else {
    jac[0] += y[3] * 5.19676363e+06 * (-2.75506200e-05 + T * (-6.20560600e-09 + T * (1.36532010e-11 + -1.74722080e-15 * T)));
  }

  if (T <= 1000.0) {
    jac[0] += y[4] * 4.88876881e+06 * (-3.22544939e-03 + T * (1.30552938e-05 + T * (-1.73956093e-08 + 8.24949516e-12 * T)));
  } else {
    jac[0] += y[4] * 4.88876881e+06 * (1.05650448e-03 + T * (-5.18165516e-07 + T * (9.15656022e-11 + -5.32783504e-15 * T)));
  }

  if (T <= 1000.0) {
    jac[0] += y[5] * 4.61523901e+06 * (3.47498200e-03 + T * (-1.27093920e-05 + T * (2.09057430e-08 + -1.00263520e-11 * T)));
  } else {
    jac[0] += y[5] * 4.61523901e+06 * (3.05629300e-03 + T * (-1.74605200e-06 + T * (3.60298800e-10 + -2.55664720e-14 * T)));
  }

  if (T <= 1000.0) {
    jac[0] += y[6] * 2.59838181e+06 * (1.12748600e-03 + T * (-1.15123000e-06 + T * (3.94163100e-09 + -3.50742160e-12 * T)));
  } else {
    jac[0] += y[6] * 2.59838181e+06 * (6.13519700e-04 + T * (-2.51768400e-07 + T * (5.32584300e-11 + -4.54574000e-15 * T)));
  }

  if (T <= 1000.0) {
    jac[0] += y[7] * 2.51903170e+06 * (-4.74912051e-03 + T * (4.23165782e-05 + T * (-7.28291682e-08 + 3.71690050e-11 * T)));
  } else {
    jac[0] += y[7] * 2.51903170e+06 * (2.23982013e-03 + T * (-1.26731630e-06 + T * (3.42739110e-10 + -4.31634140e-14 * T)));
  }

  if (T <= 1000.0) {
    jac[0] += y[8] * 2.44438441e+06 * (6.56922600e-03 + T * (-2.97002600e-07 + T * (-1.38774180e-08 + 9.88606000e-12 * T)));
  } else {
    jac[0] += y[8] * 2.44438441e+06 * (4.33613600e-03 + T * (-2.94937800e-06 + T * (7.04671200e-10 + -5.72661600e-14 * T)));
  }

  if (T <= 1000.0) {
    jac[0] += y[9] * 2.96804743e+06 * (1.40824000e-03 + T * (-7.92644400e-06 + T * (1.69245450e-08 + -9.77942000e-12 * T)));
  } else {
    jac[0] += y[9] * 2.96804743e+06 * (1.48797700e-03 + T * (-1.13695220e-06 + T * (3.02911200e-10 + -2.70134040e-14 * T)));
  }

  if (T <= 1000.0) {
    jac[0] += y[10] * 2.08133323e+06 * (0.00000000e+00 + T * (0.00000000e+00 + T * (0.00000000e+00 + 0.00000000e+00 * T)));
  } else {
    jac[0] += y[10] * 2.08133323e+06 * (0.00000000e+00 + T * (0.00000000e+00 + T * (0.00000000e+00 + 0.00000000e+00 * T)));
  }

  if (T <= 1000.0) {
    jac[0] += y[11] * 2.07727727e+07 * (0.00000000e+00 + T * (0.00000000e+00 + T * (0.00000000e+00 + 0.00000000e+00 * T)));
  } else {
    jac[0] += y[11] * 2.07727727e+07 * (0.00000000e+00 + T * (0.00000000e+00 + T * (0.00000000e+00 + 0.00000000e+00 * T)));
  }

  if (T <= 1000.0) {
    jac[0] += y[12] * 2.96834943e+06 * (1.51194100e-03 + T * (-7.76351000e-06 + T * (1.67458320e-08 + -9.89980400e-12 * T)));
  } else {
    jac[0] += y[12] * 2.96834943e+06 * (1.44268900e-03 + T * (-1.12616560e-06 + T * (3.05574300e-10 + -2.76438080e-14 * T)));
  }

  if (T <= 1000.0) {
    jac[0] += y[13] * 1.88923414e+06 * (9.92207200e-03 + T * (-2.08182200e-05 + T * (2.06000610e-08 + -8.46912000e-12 * T)));
  } else {
    jac[0] += y[13] * 1.88923414e+06 * (3.14016900e-03 + T * (-2.55682200e-06 + T * (7.18199100e-10 + -6.67613200e-14 * T)));
  }

  jac[0] *= (-1.0 / (rho * cp_avg)) * (h[0] * sp_rates[0] * 1.00797000e+00
         + h[1] * sp_rates[1] * 2.01594000e+00 + h[2] * sp_rates[2] * 1.59994000e+01
         + h[3] * sp_rates[3] * 1.70073700e+01 + h[4] * sp_rates[4] * 1.80153400e+01
         + h[5] * sp_rates[5] * 3.19988000e+01 + h[6] * sp_rates[6] * 3.30067700e+01
         + h[7] * sp_rates[7] * 3.40147400e+01 + h[8] * sp_rates[8] * 2.80134000e+01
         + h[9] * sp_rates[9] * 3.99480000e+01 + h[10] * sp_rates[10] * 4.00260000e+00
         + h[11] * sp_rates[11] * 2.80105500e+01 + h[12] * sp_rates[12] * 4.40099500e+01);
  jac[0] += ((cp[0] * sp_rates[0] * 1.00797000e+00 / rho + h[0] * jac[1])
          + (cp[1] * sp_rates[1] * 2.01594000e+00 / rho + h[1] * jac[2])
          + (cp[2] * sp_rates[2] * 1.59994000e+01 / rho + h[2] * jac[3])
          + (cp[3] * sp_rates[3] * 1.70073700e+01 / rho + h[3] * jac[4])
          + (cp[4] * sp_rates[4] * 1.80153400e+01 / rho + h[4] * jac[5])
          + (cp[5] * sp_rates[5] * 3.19988000e+01 / rho + h[5] * jac[6])
          + (cp[6] * sp_rates[6] * 3.30067700e+01 / rho + h[6] * jac[7])
          + (cp[7] * sp_rates[7] * 3.40147400e+01 / rho + h[7] * jac[8])
          + (cp[8] * sp_rates[8] * 2.80134000e+01 / rho + h[8] * jac[9])
          + (cp[9] * sp_rates[9] * 3.99480000e+01 / rho + h[9] * jac[10])
          + (cp[10] * sp_rates[10] * 4.00260000e+00 / rho + h[10] * jac[11])
          + (cp[11] * sp_rates[11] * 2.80105500e+01 / rho + h[11] * jac[12])
          + (cp[12] * sp_rates[12] * 4.40099500e+01 / rho + h[12] * jac[13]));
  jac[0] /= (-cp_avg);

  //partial of dT wrt Y_H;
  jac[14] = -(h[0] * (jac[15] - (cp[0] * sp_rates[0] * 1.00797000e+00 / (rho * cp_avg)))
         + h[1] * (jac[16] - (cp[0] * sp_rates[1] * 2.01594000e+00 / (rho * cp_avg)))
         + h[2] * (jac[17] - (cp[0] * sp_rates[2] * 1.59994000e+01 / (rho * cp_avg)))
         + h[3] * (jac[18] - (cp[0] * sp_rates[3] * 1.70073700e+01 / (rho * cp_avg)))
         + h[4] * (jac[19] - (cp[0] * sp_rates[4] * 1.80153400e+01 / (rho * cp_avg)))
         + h[5] * (jac[20] - (cp[0] * sp_rates[5] * 3.19988000e+01 / (rho * cp_avg)))
         + h[6] * (jac[21] - (cp[0] * sp_rates[6] * 3.30067700e+01 / (rho * cp_avg)))
         + h[7] * (jac[22] - (cp[0] * sp_rates[7] * 3.40147400e+01 / (rho * cp_avg)))
         + h[8] * (jac[23] - (cp[0] * sp_rates[8] * 2.80134000e+01 / (rho * cp_avg)))
         + h[9] * (jac[24] - (cp[0] * sp_rates[9] * 3.99480000e+01 / (rho * cp_avg)))
         + h[10] * (jac[25] - (cp[0] * sp_rates[10] * 4.00260000e+00 / (rho * cp_avg)))
         + h[11] * (jac[26] - (cp[0] * sp_rates[11] * 2.80105500e+01 / (rho * cp_avg)))
         + h[12] * (jac[27] - (cp[0] * sp_rates[12] * 4.40099500e+01 / (rho * cp_avg)))) / cp_avg;
  //partial of dT wrt Y_H2;
  jac[28] = -(h[0] * (jac[29] - (cp[1] * sp_rates[0] * 1.00797000e+00 / (rho * cp_avg)))
         + h[1] * (jac[30] - (cp[1] * sp_rates[1] * 2.01594000e+00 / (rho * cp_avg)))
         + h[2] * (jac[31] - (cp[1] * sp_rates[2] * 1.59994000e+01 / (rho * cp_avg)))
         + h[3] * (jac[32] - (cp[1] * sp_rates[3] * 1.70073700e+01 / (rho * cp_avg)))
         + h[4] * (jac[33] - (cp[1] * sp_rates[4] * 1.80153400e+01 / (rho * cp_avg)))
         + h[5] * (jac[34] - (cp[1] * sp_rates[5] * 3.19988000e+01 / (rho * cp_avg)))
         + h[6] * (jac[35] - (cp[1] * sp_rates[6] * 3.30067700e+01 / (rho * cp_avg)))
         + h[7] * (jac[36] - (cp[1] * sp_rates[7] * 3.40147400e+01 / (rho * cp_avg)))
         + h[8] * (jac[37] - (cp[1] * sp_rates[8] * 2.80134000e+01 / (rho * cp_avg)))
         + h[9] * (jac[38] - (cp[1] * sp_rates[9] * 3.99480000e+01 / (rho * cp_avg)))
         + h[10] * (jac[39] - (cp[1] * sp_rates[10] * 4.00260000e+00 / (rho * cp_avg)))
         + h[11] * (jac[40] - (cp[1] * sp_rates[11] * 2.80105500e+01 / (rho * cp_avg)))
         + h[12] * (jac[41] - (cp[1] * sp_rates[12] * 4.40099500e+01 / (rho * cp_avg)))) / cp_avg;
  //partial of dT wrt Y_O;
  jac[42] = -(h[0] * (jac[43] - (cp[2] * sp_rates[0] * 1.00797000e+00 / (rho * cp_avg)))
         + h[1] * (jac[44] - (cp[2] * sp_rates[1] * 2.01594000e+00 / (rho * cp_avg)))
         + h[2] * (jac[45] - (cp[2] * sp_rates[2] * 1.59994000e+01 / (rho * cp_avg)))
         + h[3] * (jac[46] - (cp[2] * sp_rates[3] * 1.70073700e+01 / (rho * cp_avg)))
         + h[4] * (jac[47] - (cp[2] * sp_rates[4] * 1.80153400e+01 / (rho * cp_avg)))
         + h[5] * (jac[48] - (cp[2] * sp_rates[5] * 3.19988000e+01 / (rho * cp_avg)))
         + h[6] * (jac[49] - (cp[2] * sp_rates[6] * 3.30067700e+01 / (rho * cp_avg)))
         + h[7] * (jac[50] - (cp[2] * sp_rates[7] * 3.40147400e+01 / (rho * cp_avg)))
         + h[8] * (jac[51] - (cp[2] * sp_rates[8] * 2.80134000e+01 / (rho * cp_avg)))
         + h[9] * (jac[52] - (cp[2] * sp_rates[9] * 3.99480000e+01 / (rho * cp_avg)))
         + h[10] * (jac[53] - (cp[2] * sp_rates[10] * 4.00260000e+00 / (rho * cp_avg)))
         + h[11] * (jac[54] - (cp[2] * sp_rates[11] * 2.80105500e+01 / (rho * cp_avg)))
         + h[12] * (jac[55] - (cp[2] * sp_rates[12] * 4.40099500e+01 / (rho * cp_avg)))) / cp_avg;
  //partial of dT wrt Y_OH;
  jac[56] = -(h[0] * (jac[57] - (cp[3] * sp_rates[0] * 1.00797000e+00 / (rho * cp_avg)))
         + h[1] * (jac[58] - (cp[3] * sp_rates[1] * 2.01594000e+00 / (rho * cp_avg)))
         + h[2] * (jac[59] - (cp[3] * sp_rates[2] * 1.59994000e+01 / (rho * cp_avg)))
         + h[3] * (jac[60] - (cp[3] * sp_rates[3] * 1.70073700e+01 / (rho * cp_avg)))
         + h[4] * (jac[61] - (cp[3] * sp_rates[4] * 1.80153400e+01 / (rho * cp_avg)))
         + h[5] * (jac[62] - (cp[3] * sp_rates[5] * 3.19988000e+01 / (rho * cp_avg)))
         + h[6] * (jac[63] - (cp[3] * sp_rates[6] * 3.30067700e+01 / (rho * cp_avg)))
         + h[7] * (jac[64] - (cp[3] * sp_rates[7] * 3.40147400e+01 / (rho * cp_avg)))
         + h[8] * (jac[65] - (cp[3] * sp_rates[8] * 2.80134000e+01 / (rho * cp_avg)))
         + h[9] * (jac[66] - (cp[3] * sp_rates[9] * 3.99480000e+01 / (rho * cp_avg)))
         + h[10] * (jac[67] - (cp[3] * sp_rates[10] * 4.00260000e+00 / (rho * cp_avg)))
         + h[11] * (jac[68] - (cp[3] * sp_rates[11] * 2.80105500e+01 / (rho * cp_avg)))
         + h[12] * (jac[69] - (cp[3] * sp_rates[12] * 4.40099500e+01 / (rho * cp_avg)))) / cp_avg;
  //partial of dT wrt Y_H2O;
  jac[70] = -(h[0] * (jac[71] - (cp[4] * sp_rates[0] * 1.00797000e+00 / (rho * cp_avg)))
         + h[1] * (jac[72] - (cp[4] * sp_rates[1] * 2.01594000e+00 / (rho * cp_avg)))
         + h[2] * (jac[73] - (cp[4] * sp_rates[2] * 1.59994000e+01 / (rho * cp_avg)))
         + h[3] * (jac[74] - (cp[4] * sp_rates[3] * 1.70073700e+01 / (rho * cp_avg)))
         + h[4] * (jac[75] - (cp[4] * sp_rates[4] * 1.80153400e+01 / (rho * cp_avg)))
         + h[5] * (jac[76] - (cp[4] * sp_rates[5] * 3.19988000e+01 / (rho * cp_avg)))
         + h[6] * (jac[77] - (cp[4] * sp_rates[6] * 3.30067700e+01 / (rho * cp_avg)))
         + h[7] * (jac[78] - (cp[4] * sp_rates[7] * 3.40147400e+01 / (rho * cp_avg)))
         + h[8] * (jac[79] - (cp[4] * sp_rates[8] * 2.80134000e+01 / (rho * cp_avg)))
         + h[9] * (jac[80] - (cp[4] * sp_rates[9] * 3.99480000e+01 / (rho * cp_avg)))
         + h[10] * (jac[81] - (cp[4] * sp_rates[10] * 4.00260000e+00 / (rho * cp_avg)))
         + h[11] * (jac[82] - (cp[4] * sp_rates[11] * 2.80105500e+01 / (rho * cp_avg)))
         + h[12] * (jac[83] - (cp[4] * sp_rates[12] * 4.40099500e+01 / (rho * cp_avg)))) / cp_avg;
  //partial of dT wrt Y_O2;
  jac[84] = -(h[0] * (jac[85] - (cp[5] * sp_rates[0] * 1.00797000e+00 / (rho * cp_avg)))
         + h[1] * (jac[86] - (cp[5] * sp_rates[1] * 2.01594000e+00 / (rho * cp_avg)))
         + h[2] * (jac[87] - (cp[5] * sp_rates[2] * 1.59994000e+01 / (rho * cp_avg)))
         + h[3] * (jac[88] - (cp[5] * sp_rates[3] * 1.70073700e+01 / (rho * cp_avg)))
         + h[4] * (jac[89] - (cp[5] * sp_rates[4] * 1.80153400e+01 / (rho * cp_avg)))
         + h[5] * (jac[90] - (cp[5] * sp_rates[5] * 3.19988000e+01 / (rho * cp_avg)))
         + h[6] * (jac[91] - (cp[5] * sp_rates[6] * 3.30067700e+01 / (rho * cp_avg)))
         + h[7] * (jac[92] - (cp[5] * sp_rates[7] * 3.40147400e+01 / (rho * cp_avg)))
         + h[8] * (jac[93] - (cp[5] * sp_rates[8] * 2.80134000e+01 / (rho * cp_avg)))
         + h[9] * (jac[94] - (cp[5] * sp_rates[9] * 3.99480000e+01 / (rho * cp_avg)))
         + h[10] * (jac[95] - (cp[5] * sp_rates[10] * 4.00260000e+00 / (rho * cp_avg)))
         + h[11] * (jac[96] - (cp[5] * sp_rates[11] * 2.80105500e+01 / (rho * cp_avg)))
         + h[12] * (jac[97] - (cp[5] * sp_rates[12] * 4.40099500e+01 / (rho * cp_avg)))) / cp_avg;
  //partial of dT wrt Y_HO2;
  jac[98] = -(h[0] * (jac[99] - (cp[6] * sp_rates[0] * 1.00797000e+00 / (rho * cp_avg)))
         + h[1] * (jac[100] - (cp[6] * sp_rates[1] * 2.01594000e+00 / (rho * cp_avg)))
         + h[2] * (jac[101] - (cp[6] * sp_rates[2] * 1.59994000e+01 / (rho * cp_avg)))
         + h[3] * (jac[102] - (cp[6] * sp_rates[3] * 1.70073700e+01 / (rho * cp_avg)))
         + h[4] * (jac[103] - (cp[6] * sp_rates[4] * 1.80153400e+01 / (rho * cp_avg)))
         + h[5] * (jac[104] - (cp[6] * sp_rates[5] * 3.19988000e+01 / (rho * cp_avg)))
         + h[6] * (jac[105] - (cp[6] * sp_rates[6] * 3.30067700e+01 / (rho * cp_avg)))
         + h[7] * (jac[106] - (cp[6] * sp_rates[7] * 3.40147400e+01 / (rho * cp_avg)))
         + h[8] * (jac[107] - (cp[6] * sp_rates[8] * 2.80134000e+01 / (rho * cp_avg)))
         + h[9] * (jac[108] - (cp[6] * sp_rates[9] * 3.99480000e+01 / (rho * cp_avg)))
         + h[10] * (jac[109] - (cp[6] * sp_rates[10] * 4.00260000e+00 / (rho * cp_avg)))
         + h[11] * (jac[110] - (cp[6] * sp_rates[11] * 2.80105500e+01 / (rho * cp_avg)))
         + h[12] * (jac[111] - (cp[6] * sp_rates[12] * 4.40099500e+01 / (rho * cp_avg)))) / cp_avg;
  //partial of dT wrt Y_H2O2;
  jac[112] = -(h[0] * (jac[113] - (cp[7] * sp_rates[0] * 1.00797000e+00 / (rho * cp_avg)))
         + h[1] * (jac[114] - (cp[7] * sp_rates[1] * 2.01594000e+00 / (rho * cp_avg)))
         + h[2] * (jac[115] - (cp[7] * sp_rates[2] * 1.59994000e+01 / (rho * cp_avg)))
         + h[3] * (jac[116] - (cp[7] * sp_rates[3] * 1.70073700e+01 / (rho * cp_avg)))
         + h[4] * (jac[117] - (cp[7] * sp_rates[4] * 1.80153400e+01 / (rho * cp_avg)))
         + h[5] * (jac[118] - (cp[7] * sp_rates[5] * 3.19988000e+01 / (rho * cp_avg)))
         + h[6] * (jac[119] - (cp[7] * sp_rates[6] * 3.30067700e+01 / (rho * cp_avg)))
         + h[7] * (jac[120] - (cp[7] * sp_rates[7] * 3.40147400e+01 / (rho * cp_avg)))
         + h[8] * (jac[121] - (cp[7] * sp_rates[8] * 2.80134000e+01 / (rho * cp_avg)))
         + h[9] * (jac[122] - (cp[7] * sp_rates[9] * 3.99480000e+01 / (rho * cp_avg)))
         + h[10] * (jac[123] - (cp[7] * sp_rates[10] * 4.00260000e+00 / (rho * cp_avg)))
         + h[11] * (jac[124] - (cp[7] * sp_rates[11] * 2.80105500e+01 / (rho * cp_avg)))
         + h[12] * (jac[125] - (cp[7] * sp_rates[12] * 4.40099500e+01 / (rho * cp_avg)))) / cp_avg;
  //partial of dT wrt Y_N2;
  jac[126] = -(h[0] * (jac[127] - (cp[8] * sp_rates[0] * 1.00797000e+00 / (rho * cp_avg)))
         + h[1] * (jac[128] - (cp[8] * sp_rates[1] * 2.01594000e+00 / (rho * cp_avg)))
         + h[2] * (jac[129] - (cp[8] * sp_rates[2] * 1.59994000e+01 / (rho * cp_avg)))
         + h[3] * (jac[130] - (cp[8] * sp_rates[3] * 1.70073700e+01 / (rho * cp_avg)))
         + h[4] * (jac[131] - (cp[8] * sp_rates[4] * 1.80153400e+01 / (rho * cp_avg)))
         + h[5] * (jac[132] - (cp[8] * sp_rates[5] * 3.19988000e+01 / (rho * cp_avg)))
         + h[6] * (jac[133] - (cp[8] * sp_rates[6] * 3.30067700e+01 / (rho * cp_avg)))
         + h[7] * (jac[134] - (cp[8] * sp_rates[7] * 3.40147400e+01 / (rho * cp_avg)))
         + h[8] * (jac[135] - (cp[8] * sp_rates[8] * 2.80134000e+01 / (rho * cp_avg)))
         + h[9] * (jac[136] - (cp[8] * sp_rates[9] * 3.99480000e+01 / (rho * cp_avg)))
         + h[10] * (jac[137] - (cp[8] * sp_rates[10] * 4.00260000e+00 / (rho * cp_avg)))
         + h[11] * (jac[138] - (cp[8] * sp_rates[11] * 2.80105500e+01 / (rho * cp_avg)))
         + h[12] * (jac[139] - (cp[8] * sp_rates[12] * 4.40099500e+01 / (rho * cp_avg)))) / cp_avg;
  //partial of dT wrt Y_AR;
  jac[140] = -(h[0] * (jac[141] - (cp[9] * sp_rates[0] * 1.00797000e+00 / (rho * cp_avg)))
         + h[1] * (jac[142] - (cp[9] * sp_rates[1] * 2.01594000e+00 / (rho * cp_avg)))
         + h[2] * (jac[143] - (cp[9] * sp_rates[2] * 1.59994000e+01 / (rho * cp_avg)))
         + h[3] * (jac[144] - (cp[9] * sp_rates[3] * 1.70073700e+01 / (rho * cp_avg)))
         + h[4] * (jac[145] - (cp[9] * sp_rates[4] * 1.80153400e+01 / (rho * cp_avg)))
         + h[5] * (jac[146] - (cp[9] * sp_rates[5] * 3.19988000e+01 / (rho * cp_avg)))
         + h[6] * (jac[147] - (cp[9] * sp_rates[6] * 3.30067700e+01 / (rho * cp_avg)))
         + h[7] * (jac[148] - (cp[9] * sp_rates[7] * 3.40147400e+01 / (rho * cp_avg)))
         + h[8] * (jac[149] - (cp[9] * sp_rates[8] * 2.80134000e+01 / (rho * cp_avg)))
         + h[9] * (jac[150] - (cp[9] * sp_rates[9] * 3.99480000e+01 / (rho * cp_avg)))
         + h[10] * (jac[151] - (cp[9] * sp_rates[10] * 4.00260000e+00 / (rho * cp_avg)))
         + h[11] * (jac[152] - (cp[9] * sp_rates[11] * 2.80105500e+01 / (rho * cp_avg)))
         + h[12] * (jac[153] - (cp[9] * sp_rates[12] * 4.40099500e+01 / (rho * cp_avg)))) / cp_avg;
  //partial of dT wrt Y_HE;
  jac[154] = -(h[0] * (jac[155] - (cp[10] * sp_rates[0] * 1.00797000e+00 / (rho * cp_avg)))
         + h[1] * (jac[156] - (cp[10] * sp_rates[1] * 2.01594000e+00 / (rho * cp_avg)))
         + h[2] * (jac[157] - (cp[10] * sp_rates[2] * 1.59994000e+01 / (rho * cp_avg)))
         + h[3] * (jac[158] - (cp[10] * sp_rates[3] * 1.70073700e+01 / (rho * cp_avg)))
         + h[4] * (jac[159] - (cp[10] * sp_rates[4] * 1.80153400e+01 / (rho * cp_avg)))
         + h[5] * (jac[160] - (cp[10] * sp_rates[5] * 3.19988000e+01 / (rho * cp_avg)))
         + h[6] * (jac[161] - (cp[10] * sp_rates[6] * 3.30067700e+01 / (rho * cp_avg)))
         + h[7] * (jac[162] - (cp[10] * sp_rates[7] * 3.40147400e+01 / (rho * cp_avg)))
         + h[8] * (jac[163] - (cp[10] * sp_rates[8] * 2.80134000e+01 / (rho * cp_avg)))
         + h[9] * (jac[164] - (cp[10] * sp_rates[9] * 3.99480000e+01 / (rho * cp_avg)))
         + h[10] * (jac[165] - (cp[10] * sp_rates[10] * 4.00260000e+00 / (rho * cp_avg)))
         + h[11] * (jac[166] - (cp[10] * sp_rates[11] * 2.80105500e+01 / (rho * cp_avg)))
         + h[12] * (jac[167] - (cp[10] * sp_rates[12] * 4.40099500e+01 / (rho * cp_avg)))) / cp_avg;
  //partial of dT wrt Y_CO;
  jac[168] = -(h[0] * (jac[169] - (cp[11] * sp_rates[0] * 1.00797000e+00 / (rho * cp_avg)))
         + h[1] * (jac[170] - (cp[11] * sp_rates[1] * 2.01594000e+00 / (rho * cp_avg)))
         + h[2] * (jac[171] - (cp[11] * sp_rates[2] * 1.59994000e+01 / (rho * cp_avg)))
         + h[3] * (jac[172] - (cp[11] * sp_rates[3] * 1.70073700e+01 / (rho * cp_avg)))
         + h[4] * (jac[173] - (cp[11] * sp_rates[4] * 1.80153400e+01 / (rho * cp_avg)))
         + h[5] * (jac[174] - (cp[11] * sp_rates[5] * 3.19988000e+01 / (rho * cp_avg)))
         + h[6] * (jac[175] - (cp[11] * sp_rates[6] * 3.30067700e+01 / (rho * cp_avg)))
         + h[7] * (jac[176] - (cp[11] * sp_rates[7] * 3.40147400e+01 / (rho * cp_avg)))
         + h[8] * (jac[177] - (cp[11] * sp_rates[8] * 2.80134000e+01 / (rho * cp_avg)))
         + h[9] * (jac[178] - (cp[11] * sp_rates[9] * 3.99480000e+01 / (rho * cp_avg)))
         + h[10] * (jac[179] - (cp[11] * sp_rates[10] * 4.00260000e+00 / (rho * cp_avg)))
         + h[11] * (jac[180] - (cp[11] * sp_rates[11] * 2.80105500e+01 / (rho * cp_avg)))
         + h[12] * (jac[181] - (cp[11] * sp_rates[12] * 4.40099500e+01 / (rho * cp_avg)))) / cp_avg;
  //partial of dT wrt Y_CO2;
  jac[182] = -(h[0] * (jac[183] - (cp[12] * sp_rates[0] * 1.00797000e+00 / (rho * cp_avg)))
         + h[1] * (jac[184] - (cp[12] * sp_rates[1] * 2.01594000e+00 / (rho * cp_avg)))
         + h[2] * (jac[185] - (cp[12] * sp_rates[2] * 1.59994000e+01 / (rho * cp_avg)))
         + h[3] * (jac[186] - (cp[12] * sp_rates[3] * 1.70073700e+01 / (rho * cp_avg)))
         + h[4] * (jac[187] - (cp[12] * sp_rates[4] * 1.80153400e+01 / (rho * cp_avg)))
         + h[5] * (jac[188] - (cp[12] * sp_rates[5] * 3.19988000e+01 / (rho * cp_avg)))
         + h[6] * (jac[189] - (cp[12] * sp_rates[6] * 3.30067700e+01 / (rho * cp_avg)))
         + h[7] * (jac[190] - (cp[12] * sp_rates[7] * 3.40147400e+01 / (rho * cp_avg)))
         + h[8] * (jac[191] - (cp[12] * sp_rates[8] * 2.80134000e+01 / (rho * cp_avg)))
         + h[9] * (jac[192] - (cp[12] * sp_rates[9] * 3.99480000e+01 / (rho * cp_avg)))
         + h[10] * (jac[193] - (cp[12] * sp_rates[10] * 4.00260000e+00 / (rho * cp_avg)))
         + h[11] * (jac[194] - (cp[12] * sp_rates[11] * 2.80105500e+01 / (rho * cp_avg)))
         + h[12] * (jac[195] - (cp[12] * sp_rates[12] * 4.40099500e+01 / (rho * cp_avg)))) / cp_avg;
} // end eval_jacob

