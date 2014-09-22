#include <math.h>
#include "header.h"

void get_rxn_pres_mod (const Real T, const Real pres, const Real * C, Real * pres_mod) {
  // third body variable declaration
  Real thd;

  // pressure dependence variable declarations
  Real k0;
  Real kinf;
  Real Pr;

  // troe variable declarations
  Real logFcent;
  Real A;
  Real B;

  Real logT = log(T);
  Real m = pres / (8.31451000e+07 * T);

  // reaction 0;
  pres_mod[0] = m + 1.4 * C[0] + 14.4 * C[5] + 1.0 * C[13] + 0.75 * C[14] + 2.6 * C[15] + 2.0 * C[26] - 0.17 * C[48];

  // reaction 1;
  pres_mod[1] = m + 1.0 * C[0] + 5.0 * C[5] + 1.0 * C[13] + 0.5 * C[14] + 1.0 * C[15] + 2.0 * C[26] - 0.3 * C[48];

  // reaction 11;
  thd = m + 1.0 * C[0] + 5.0 * C[3] + 5.0 * C[5] + 1.0 * C[13] + 0.5 * C[14] + 2.5 * C[15] + 2.0 * C[26] - 0.5 * C[48];
  k0 = exp(3.40312786e+01 - (1.50965000e+03 / T));
  kinf = exp(2.36136376e+01 - (1.20017175e+03 / T));
  Pr = k0 * thd / kinf;
  Pr = k0 * thd / kinf;
* Pr / (1.0 + Pr);

  // reaction 32;
  pres_mod[3] = m - 1.0 * C[3] - 1.0 * C[5] - 0.25 * C[14] + 0.5 * C[15] + 0.5 * C[26] - 1.0 * C[47] - 1.0 * C[48];

  // reaction 38;
  pres_mod[4] = m - 1.0 * C[0] - 1.0 * C[5] + 1.0 * C[13] - 1.0 * C[15] + 2.0 * C[26] - 0.37 * C[48];

  // reaction 42;
  pres_mod[5] = m - 0.27 * C[0] + 2.65 * C[5] + 1.0 * C[13] + 2.0 * C[26] - 0.62 * C[48];

  // reaction 49;
  thd = m + 1.0 * C[0] + 5.0 * C[5] + 1.0 * C[13] + 0.5 * C[14] + 1.0 * C[15] + 2.0 * C[26] - 0.3 * C[48];
  k0 = exp(5.99064331e+01 - 2.76 * logT - (8.05146665e+02 / T));
  kinf = 6e+14;
  Pr = k0 * thd / kinf;
  logFcent = log10( fmax(4.38000000e-01 * exp(-T / 9.10000000e+01) + 5.62000000e-01 * exp(-T / 5.83600000e+03) + exp(-8.55200000e+03 / T), 1.0e-300));
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * logFcent - 0.4;
  B = 0.806 - 1.1762 * logFcent - 0.14 * log10(fmax(Pr, 1.0e-300));
  pres_mod[6] = pow(10.0, logFcent / (1.0 + A * A / (B * B))) * Pr / (1.0 + Pr);

  // reaction 51;
  thd = m + 1.0 * C[0] + 5.0 * C[5] + 2.0 * C[13] + 0.5 * C[14] + 1.0 * C[15] + 2.0 * C[26] - 0.3 * C[48];
  k0 = exp(7.69484824e+01 - 4.76 * logT - (1.22784866e+03 / T));
  kinf = exp(3.71706652e+01 - 0.534 * logT - (2.69724133e+02 / T));
  Pr = k0 * thd / kinf;
  logFcent = log10( fmax(2.17000000e-01 * exp(-T / 7.40000000e+01) + 7.83000000e-01 * exp(-T / 2.94100000e+03) + exp(-6.96400000e+03 / T), 1.0e-300));
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * logFcent - 0.4;
  B = 0.806 - 1.1762 * logFcent - 0.14 * log10(fmax(Pr, 1.0e-300));
  pres_mod[7] = pow(10.0, logFcent / (1.0 + A * A / (B * B))) * Pr / (1.0 + Pr);

  // reaction 53;
  thd = m + 1.0 * C[0] + 5.0 * C[5] + 1.0 * C[13] + 0.5 * C[14] + 1.0 * C[15] + 2.0 * C[26] - 0.3 * C[48];
  k0 = exp(5.61662604e+01 - 2.57 * logT - (2.13867083e+02 / T));
  kinf = exp(2.77171988e+01 + 0.48 * logT - (-1.30836333e+02 / T));
  Pr = k0 * thd / kinf;
  logFcent = log10( fmax(2.17600000e-01 * exp(-T / 2.71000000e+02) + 7.82400000e-01 * exp(-T / 2.75500000e+03) + exp(-6.57000000e+03 / T), 1.0e-300));
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * logFcent - 0.4;
  B = 0.806 - 1.1762 * logFcent - 0.14 * log10(fmax(Pr, 1.0e-300));
  pres_mod[8] = pow(10.0, logFcent / (1.0 + A * A / (B * B))) * Pr / (1.0 + Pr);

  // reaction 55;
  thd = m + 1.0 * C[0] + 5.0 * C[5] + 1.0 * C[13] + 0.5 * C[14] + 1.0 * C[15] + 2.0 * C[26];
  k0 = exp(7.39217399e+01 - 4.82 * logT - (3.28600483e+03 / T));
  kinf = exp(2.70148350e+01 + 0.454 * logT - (1.81158000e+03 / T));
  Pr = k0 * thd / kinf;
  logFcent = log10( fmax(2.81300000e-01 * exp(-T / 1.03000000e+02) + 7.18700000e-01 * exp(-T / 1.29100000e+03) + exp(-4.16000000e+03 / T), 1.0e-300));
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * logFcent - 0.4;
  B = 0.806 - 1.1762 * logFcent - 0.14 * log10(fmax(Pr, 1.0e-300));
  pres_mod[9] = pow(10.0, logFcent / (1.0 + A * A / (B * B))) * Pr / (1.0 + Pr);

  // reaction 56;
  thd = m + 1.0 * C[0] + 5.0 * C[5] + 1.0 * C[13] + 0.5 * C[14] + 1.0 * C[15] + 2.0 * C[26];
  k0 = exp(6.98660102e+01 - 4.8 * logT - (2.79788466e+03 / T));
  kinf = exp(2.70148350e+01 + 0.454 * logT - (1.30836333e+03 / T));
  Pr = k0 * thd / kinf;
  logFcent = log10( fmax(2.42000000e-01 * exp(-T / 9.40000000e+01) + 7.58000000e-01 * exp(-T / 1.55500000e+03) + exp(-4.20000000e+03 / T), 1.0e-300));
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * logFcent - 0.4;
  B = 0.806 - 1.1762 * logFcent - 0.14 * log10(fmax(Pr, 1.0e-300));
  pres_mod[10] = pow(10.0, logFcent / (1.0 + A * A / (B * B))) * Pr / (1.0 + Pr);

  // reaction 58;
  thd = m + 1.0 * C[0] + 5.0 * C[5] + 1.0 * C[13] + 0.5 * C[14] + 1.0 * C[15] + 2.0 * C[26];
  k0 = exp(7.28526099e+01 - 4.65 * logT - (2.55634066e+03 / T));
  kinf = exp(2.76845619e+01 + 0.5 * logT - (4.32766333e+01 / T));
  Pr = k0 * thd / kinf;
  logFcent = log10( fmax(4.00000000e-01 * exp(-T / 1.00000000e+02) + 6.00000000e-01 * exp(-T / 9.00000000e+04) + exp(-1.00000000e+04 / T), 1.0e-300));
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * logFcent - 0.4;
  B = 0.806 - 1.1762 * logFcent - 0.14 * log10(fmax(Pr, 1.0e-300));
  pres_mod[11] = pow(10.0, logFcent / (1.0 + A * A / (B * B))) * Pr / (1.0 + Pr);

  // reaction 62;
  thd = m + 1.0 * C[0] + 5.0 * C[5] + 1.0 * C[13] + 0.5 * C[14] + 1.0 * C[15] + 2.0 * C[26];
  k0 = exp(9.59450043e+01 - 7.44 * logT - (7.08529065e+03 / T));
  kinf = exp(2.85189124e+01 + 0.515 * logT - (2.51608333e+01 / T));
  Pr = k0 * thd / kinf;
  logFcent = log10( fmax(3.00000000e-01 * exp(-T / 1.00000000e+02) + 7.00000000e-01 * exp(-T / 9.00000000e+04) + exp(-1.00000000e+04 / T), 1.0e-300));
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * logFcent - 0.4;
  B = 0.806 - 1.1762 * logFcent - 0.14 * log10(fmax(Pr, 1.0e-300));
  pres_mod[12] = pow(10.0, logFcent / (1.0 + A * A / (B * B))) * Pr / (1.0 + Pr);

  // reaction 69;
  thd = m + 1.0 * C[0] + 5.0 * C[5] + 1.0 * C[13] + 0.5 * C[14] + 1.0 * C[15] + 2.0 * C[26] - 0.3 * C[48];
  k0 = exp(7.73070639e+01 - 4.8 * logT - (9.56111665e+02 / T));
  kinf = exp(3.91439466e+01 - 1.0 * logT);
  Pr = k0 * thd / kinf;
  logFcent = log10( fmax(3.53600000e-01 * exp(-T / 1.32000000e+02) + 6.46400000e-01 * exp(-T / 1.31500000e+03) + exp(-5.56600000e+03 / T), 1.0e-300));
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * logFcent - 0.4;
  B = 0.806 - 1.1762 * logFcent - 0.14 * log10(fmax(Pr, 1.0e-300));
  pres_mod[13] = pow(10.0, logFcent / (1.0 + A * A / (B * B))) * Pr / (1.0 + Pr);

  // reaction 70;
  thd = m + 1.0 * C[0] + 5.0 * C[5] + 1.0 * C[13] + 0.5 * C[14] + 1.0 * C[15] + 2.0 * C[26] - 0.3 * C[48];
  k0 = exp(9.34384048e+01 - 7.27 * logT - (3.63322433e+03 / T));
  kinf = exp(2.93537877e+01 - (1.20772000e+03 / T));
  Pr = k0 * thd / kinf;
  logFcent = log10( fmax(2.49300000e-01 * exp(-T / 9.85000000e+01) + 7.50700000e-01 * exp(-T / 1.30200000e+03) + exp(-4.16700000e+03 / T), 1.0e-300));
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * logFcent - 0.4;
  B = 0.806 - 1.1762 * logFcent - 0.14 * log10(fmax(Pr, 1.0e-300));
  pres_mod[14] = pow(10.0, logFcent / (1.0 + A * A / (B * B))) * Pr / (1.0 + Pr);

  // reaction 71;
  thd = m + 1.0 * C[0] + 5.0 * C[5] + 1.0 * C[13] + 0.5 * C[14] + 1.0 * C[15] + 2.0 * C[26] - 0.3 * C[48];
  k0 = exp(6.94140250e+01 - 3.86 * logT - (1.67067933e+03 / T));
  kinf = exp(2.94360258e+01 + 0.27 * logT - (1.40900666e+02 / T));
  Pr = k0 * thd / kinf;
  logFcent = log10( fmax(2.18000000e-01 * exp(-T / 2.07500000e+02) + 7.82000000e-01 * exp(-T / 2.66300000e+03) + exp(-6.09500000e+03 / T), 1.0e-300));
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * logFcent - 0.4;
  B = 0.806 - 1.1762 * logFcent - 0.14 * log10(fmax(Pr, 1.0e-300));
  pres_mod[15] = pow(10.0, logFcent / (1.0 + A * A / (B * B))) * Pr / (1.0 + Pr);

  // reaction 73;
  thd = m + 1.0 * C[0] + 5.0 * C[5] + 1.0 * C[13] + 0.5 * C[14] + 1.0 * C[15] + 2.0 * C[26] - 0.3 * C[48];
  k0 = exp(9.61977483e+01 - 7.62 * logT - (3.50742016e+03 / T));
  kinf = exp(2.70148350e+01 + 0.454 * logT - (9.15854332e+02 / T));
  Pr = k0 * thd / kinf;
  logFcent = log10( fmax(2.47000000e-02 * exp(-T / 2.10000000e+02) + 9.75300000e-01 * exp(-T / 9.84000000e+02) + exp(-4.37400000e+03 / T), 1.0e-300));
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * logFcent - 0.4;
  B = 0.806 - 1.1762 * logFcent - 0.14 * log10(fmax(Pr, 1.0e-300));
  pres_mod[16] = pow(10.0, logFcent / (1.0 + A * A / (B * B))) * Pr / (1.0 + Pr);

  // reaction 75;
  thd = m + 1.0 * C[0] + 5.0 * C[5] + 1.0 * C[13] + 0.5 * C[14] + 1.0 * C[15] + 2.0 * C[26] - 0.3 * C[48];
  k0 = exp(9.50941235e+01 - 7.08 * logT - (3.36400341e+03 / T));
  kinf = exp(4.07945264e+01 - 0.99 * logT - (7.95082332e+02 / T));
  Pr = k0 * thd / kinf;
  logFcent = log10( fmax(1.57800000e-01 * exp(-T / 1.25000000e+02) + 8.42200000e-01 * exp(-T / 2.21900000e+03) + exp(-6.88200000e+03 / T), 1.0e-300));
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * logFcent - 0.4;
  B = 0.806 - 1.1762 * logFcent - 0.14 * log10(fmax(Pr, 1.0e-300));
  pres_mod[17] = pow(10.0, logFcent / (1.0 + A * A / (B * B))) * Pr / (1.0 + Pr);

  // reaction 82;
  thd = m + 1.0 * C[0] + 5.0 * C[5] + 1.0 * C[13] + 0.5 * C[14] + 1.0 * C[15] + 2.0 * C[26] - 0.3 * C[48];
  k0 = exp(6.37931383e+01 - 3.42 * logT - (4.24463258e+04 / T));
  kinf = exp(1.75767107e+01 + 1.5 * logT - (4.00560466e+04 / T));
  Pr = k0 * thd / kinf;
  logFcent = log10( fmax(6.80000000e-02 * exp(-T / 1.97000000e+02) + 9.32000000e-01 * exp(-T / 1.54000000e+03) + exp(-1.03000000e+04 / T), 1.0e-300));
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * logFcent - 0.4;
  B = 0.806 - 1.1762 * logFcent - 0.14 * log10(fmax(Pr, 1.0e-300));
  pres_mod[18] = pow(10.0, logFcent / (1.0 + A * A / (B * B))) * Pr / (1.0 + Pr);

  // reaction 84;
  thd = m + 1.0 * C[0] + 5.0 * C[5] + 1.0 * C[13] + 0.5 * C[14] + 1.0 * C[15] + 2.0 * C[26] - 0.3 * C[48];
  k0 = exp(4.22794408e+01 - 0.9 * logT - (-8.55468332e+02 / T));
  kinf = exp(3.19350862e+01 - 0.37 * logT);
  Pr = k0 * thd / kinf;
  logFcent = log10( fmax(2.65400000e-01 * exp(-T / 9.40000000e+01) + 7.34600000e-01 * exp(-T / 1.75600000e+03) + exp(-5.18200000e+03 / T), 1.0e-300));
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * logFcent - 0.4;
  B = 0.806 - 1.1762 * logFcent - 0.14 * log10(fmax(Pr, 1.0e-300));
  pres_mod[19] = pow(10.0, logFcent / (1.0 + A * A / (B * B))) * Pr / (1.0 + Pr);

  // reaction 94;
  thd = m + 1.0 * C[0] + 5.0 * C[5] + 1.0 * C[13] + 0.5 * C[14] + 1.0 * C[15] + 2.0 * C[26];
  k0 = exp(8.42793577e+01 - 5.92 * logT - (1.58010033e+03 / T));
  kinf = exp(4.24725733e+01 - 1.43 * logT - (6.69278166e+02 / T));
  Pr = k0 * thd / kinf;
  logFcent = log10( fmax(5.88000000e-01 * exp(-T / 1.95000000e+02) + 4.12000000e-01 * exp(-T / 5.90000000e+03) + exp(-6.39400000e+03 / T), 1.0e-300));
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * logFcent - 0.4;
  B = 0.806 - 1.1762 * logFcent - 0.14 * log10(fmax(Pr, 1.0e-300));
  pres_mod[20] = pow(10.0, logFcent / (1.0 + A * A / (B * B))) * Pr / (1.0 + Pr);

  // reaction 130;
  thd = m + 1.0 * C[0] + 5.0 * C[5] + 1.0 * C[13] + 0.5 * C[14] + 1.0 * C[15] + 2.0 * C[26] - 0.3 * C[48];
  k0 = exp(6.54619238e+01 - 3.74 * logT - (9.74227465e+02 / T));
  kinf = 5e+13;
  Pr = k0 * thd / kinf;
  logFcent = log10( fmax(4.24300000e-01 * exp(-T / 2.37000000e+02) + 5.75700000e-01 * exp(-T / 1.65200000e+03) + exp(-5.06900000e+03 / T), 1.0e-300));
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * logFcent - 0.4;
  B = 0.806 - 1.1762 * logFcent - 0.14 * log10(fmax(Pr, 1.0e-300));
  pres_mod[21] = pow(10.0, logFcent / (1.0 + A * A / (B * B))) * Pr / (1.0 + Pr);

  // reaction 139;
  thd = m + 1.0 * C[0] + 5.0 * C[5] + 1.0 * C[13] + 0.5 * C[14] + 1.0 * C[15] + 2.0 * C[26] - 0.3 * C[48];
  k0 = exp(7.69748493e+01 - 5.11 * logT - (3.57032224e+03 / T));
  kinf = exp(2.74203001e+01 + 0.5 * logT - (2.26950716e+03 / T));
  Pr = k0 * thd / kinf;
  logFcent = log10( fmax(4.09300000e-01 * exp(-T / 2.75000000e+02) + 5.90700000e-01 * exp(-T / 1.22600000e+03) + exp(-5.18500000e+03 / T), 1.0e-300));
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * logFcent - 0.4;
  B = 0.806 - 1.1762 * logFcent - 0.14 * log10(fmax(Pr, 1.0e-300));
  pres_mod[22] = pow(10.0, logFcent / (1.0 + A * A / (B * B))) * Pr / (1.0 + Pr);

  // reaction 146;
  thd = m + 1.0 * C[0] + 5.0 * C[5] + 1.0 * C[13] + 0.5 * C[14] + 1.0 * C[15] + 2.0 * C[26];
  k0 = exp(8.81295053e+01 - 6.36 * logT - (2.53621200e+03 / T));
  kinf = exp(4.07167205e+01 - 1.16 * logT - (5.76183082e+02 / T));
  Pr = k0 * thd / kinf;
  logFcent = log10( fmax(3.97300000e-01 * exp(-T / 2.08000000e+02) + 6.02700000e-01 * exp(-T / 3.92200000e+03) + exp(-1.01800000e+04 / T), 1.0e-300));
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * logFcent - 0.4;
  B = 0.806 - 1.1762 * logFcent - 0.14 * log10(fmax(Pr, 1.0e-300));
  pres_mod[23] = pow(10.0, logFcent / (1.0 + A * A / (B * B))) * Pr / (1.0 + Pr);

  // reaction 157;
  thd = m + 1.0 * C[0] + 5.0 * C[5] + 1.0 * C[13] + 0.5 * C[14] + 1.0 * C[15] + 2.0 * C[26] - 0.3 * C[48];
  k0 = exp(9.56297642e+01 - 7.03 * logT - (1.38988443e+03 / T));
  kinf = exp(3.87538626e+01 - 1.18 * logT - (3.29103699e+02 / T));
  Pr = k0 * thd / kinf;
  logFcent = log10( fmax(3.81000000e-01 * exp(-T / 7.32000000e+01) + 6.19000000e-01 * exp(-T / 1.18000000e+03) + exp(-9.99900000e+03 / T), 1.0e-300));
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * logFcent - 0.4;
  B = 0.806 - 1.1762 * logFcent - 0.14 * log10(fmax(Pr, 1.0e-300));
  pres_mod[24] = pow(10.0, logFcent / (1.0 + A * A / (B * B))) * Pr / (1.0 + Pr);

  // reaction 166;
  pres_mod[25] = m + 1.0 * C[0] - 1.0 * C[5] + 1.0 * C[13] + 0.5 * C[14] + 1.0 * C[15] + 2.0 * C[26];

  // reaction 173;
  thd = m + 1.0 * C[0] + 5.0 * C[5] + 1.0 * C[13] + 0.5 * C[14] + 1.0 * C[15] + 2.0 * C[26] - 0.3 * C[48];
  k0 = exp(1.17889265e+02 - 9.3 * logT - (4.92145899e+04 / T));
  kinf = exp(2.97104627e+01 + 0.44 * logT - (4.36641101e+04 / T));
  Pr = k0 * thd / kinf;
  logFcent = log10( fmax(2.65500000e-01 * exp(-T / 1.80000000e+02) + 7.34500000e-01 * exp(-T / 1.03500000e+03) + exp(-5.41700000e+03 / T), 1.0e-300));
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * logFcent - 0.4;
  B = 0.806 - 1.1762 * logFcent - 0.14 * log10(fmax(Pr, 1.0e-300));
  pres_mod[26] = pow(10.0, logFcent / (1.0 + A * A / (B * B))) * Pr / (1.0 + Pr);

  // reaction 184;
  thd = m + 1.0 * C[0] + 5.0 * C[5] + 1.0 * C[13] + 0.5 * C[14] + 1.0 * C[15] + 2.0 * C[26] - 0.375 * C[48];
  k0 = exp(3.40877908e+01 - (2.85021920e+04 / T));
  kinf = exp(2.50939787e+01 - (2.81901976e+04 / T));
  Pr = k0 * thd / kinf;
  Pr = k0 * thd / kinf;
* Pr / (1.0 + Pr);

  // reaction 186;
  pres_mod[28] = m + 1.0 * C[0] + 5.0 * C[5] + 1.0 * C[13] + 0.5 * C[14] + 1.0 * C[15] + 2.0 * C[26] - 0.3 * C[48];

  // reaction 204;
  pres_mod[29] = m + 1.0 * C[0] + 5.0 * C[5] + 1.0 * C[13] + 0.5 * C[14] + 1.0 * C[15] + 2.0 * C[26] - 0.3 * C[48];

  // reaction 211;
  pres_mod[30] = m + 1.0 * C[0] + 5.0 * C[5] + 1.0 * C[13] + 0.5 * C[14] + 1.0 * C[15] + 2.0 * C[26] - 0.3 * C[48];

  // reaction 226;
  pres_mod[31] = m + 1.0 * C[0] + 5.0 * C[5] + 1.0 * C[13] + 0.5 * C[14] + 1.0 * C[15] + 2.0 * C[26] - 0.3 * C[48];

  // reaction 229;
  pres_mod[32] = m + 1.0 * C[0] + 5.0 * C[5] + 1.0 * C[13] + 0.5 * C[14] + 1.0 * C[15] + 2.0 * C[26] - 0.3 * C[48];

  // reaction 236;
  thd = m + 1.0 * C[0] + 5.0 * C[5] + 1.0 * C[13] + 0.5 * C[14] + 1.0 * C[15] + 2.0 * C[26] - 0.3 * C[48];
  k0 = exp(6.02036847e+01 - 3.4 * logT - (9.56111665e+02 / T));
  kinf = 3.3e+13;
  Pr = k0 * thd / kinf;
  Pr = k0 * thd / kinf;
* Pr / (1.0 + Pr);

  // reaction 240;
  thd = m + 1.0 * C[0] + 5.0 * C[5] + 1.0 * C[13] + 0.5 * C[14] + 1.0 * C[15] + 2.0 * C[26] * C[48];
  k0 = exp(5.78269916e+01 - 3.16 * logT - (3.72380333e+02 / T));
  kinf = exp(2.87624232e+01 + 0.15 * logT);
  Pr = k0 * thd / kinf;
  logFcent = log10( fmax(3.33000000e-01 * exp(-T / 2.35000000e+02) + 6.67000000e-01 * exp(-T / 2.11700000e+03) + exp(-4.53600000e+03 / T), 1.0e-300));
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * logFcent - 0.4;
  B = 0.806 - 1.1762 * logFcent - 0.14 * log10(fmax(Pr, 1.0e-300));
  pres_mod[34] = pow(10.0, logFcent / (1.0 + A * A / (B * B))) * Pr / (1.0 + Pr);

  // reaction 268;
  pres_mod[35] = m + 1.0 * C[0] + 5.0 * C[5] + 1.0 * C[13] + 0.5 * C[14] + 1.0 * C[15] + 2.0 * C[26] - 0.3 * C[48];

  // reaction 288;
  thd = m + 1.0 * C[0] + 5.0 * C[5] + 1.0 * C[13] + 0.5 * C[14] + 1.0 * C[15] + 2.0 * C[26] - 0.3 * C[48];
  k0 = exp(5.91374013e+01 - 2.8 * logT - (2.96897833e+02 / T));
  kinf = exp(2.83090547e+01 + 0.43 * logT - (-1.86190166e+02 / T));
  Pr = k0 * thd / kinf;
  logFcent = log10( fmax(4.22000000e-01 * exp(-T / 1.22000000e+02) + 5.78000000e-01 * exp(-T / 2.53500000e+03) + exp(-9.36500000e+03 / T), 1.0e-300));
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * logFcent - 0.4;
  B = 0.806 - 1.1762 * logFcent - 0.14 * log10(fmax(Pr, 1.0e-300));
  pres_mod[36] = pow(10.0, logFcent / (1.0 + A * A / (B * B))) * Pr / (1.0 + Pr);

  // reaction 303;
  thd = m + 1.0 * C[0] + 5.0 * C[5] + 1.0 * C[13] + 0.5 * C[14] + 1.0 * C[15] + 2.0 * C[26] - 0.3 * C[48];
  k0 = exp(9.67205025e+01 - 7.63 * logT - (1.93939703e+03 / T));
  kinf = exp(2.69105027e+01 + 0.422 * logT - (-8.83145248e+02 / T));
  Pr = k0 * thd / kinf;
  logFcent = log10( fmax(5.35000000e-01 * exp(-T / 2.01000000e+02) + 4.65000000e-01 * exp(-T / 1.77300000e+03) + exp(-5.33300000e+03 / T), 1.0e-300));
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * logFcent - 0.4;
  B = 0.806 - 1.1762 * logFcent - 0.14 * log10(fmax(Pr, 1.0e-300));
  pres_mod[37] = pow(10.0, logFcent / (1.0 + A * A / (B * B))) * Pr / (1.0 + Pr);

  // reaction 311;
  thd = m + 1.0 * C[0] + 5.0 * C[5] + 1.0 * C[13] + 0.5 * C[14] + 1.0 * C[15] + 2.0 * C[26] - 0.3 * C[48];
  k0 = exp(1.71388246e+02 - 16.82 * logT - (6.57452574e+03 / T));
  kinf = 9.43e+12;
  Pr = k0 * thd / kinf;
  logFcent = log10( fmax(8.47300000e-01 * exp(-T / 2.91000000e+02) + 1.52700000e-01 * exp(-T / 2.74200000e+03) + exp(-7.74800000e+03 / T), 1.0e-300));
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * logFcent - 0.4;
  B = 0.806 - 1.1762 * logFcent - 0.14 * log10(fmax(Pr, 1.0e-300));
  pres_mod[38] = pow(10.0, logFcent / (1.0 + A * A / (B * B))) * Pr / (1.0 + Pr);

  // reaction 317;
  thd = m + 1.0 * C[0] + 5.0 * C[5] + 1.0 * C[13] + 0.5 * C[14] + 1.0 * C[15] + 2.0 * C[26] - 0.3 * C[48];
  k0 = exp(1.46161473e+02 - 14.6 * logT - (9.14344682e+03 / T));
  kinf = exp(1.47516039e+01 + 1.6 * logT - (2.86833500e+03 / T));
  Pr = k0 * thd / kinf;
  logFcent = log10( fmax(8.10600000e-01 * exp(-T / 2.77000000e+02) + 1.89400000e-01 * exp(-T / 8.74800000e+03) + exp(-7.89100000e+03 / T), 1.0e-300));
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * logFcent - 0.4;
  B = 0.806 - 1.1762 * logFcent - 0.14 * log10(fmax(Pr, 1.0e-300));
  pres_mod[39] = pow(10.0, logFcent / (1.0 + A * A / (B * B))) * Pr / (1.0 + Pr);

  // reaction 319;
  thd = m + 1.0 * C[0] + 5.0 * C[5] + 1.0 * C[13] + 0.5 * C[14] + 1.0 * C[15] + 2.0 * C[26] - 0.3 * C[48];
  k0 = exp(1.41943830e+02 - 13.545 * logT - (5.71503167e+03 / T));
  kinf = 3.613e+13;
  Pr = k0 * thd / kinf;
  logFcent = log10( fmax(6.85000000e-01 * exp(-T / 3.69000000e+02) + 3.15000000e-01 * exp(-T / 3.28500000e+03) + exp(-6.66700000e+03 / T), 1.0e-300));
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * logFcent - 0.4;
  B = 0.806 - 1.1762 * logFcent - 0.14 * log10(fmax(Pr, 1.0e-300));
  pres_mod[40] = pow(10.0, logFcent / (1.0 + A * A / (B * B))) * Pr / (1.0 + Pr);

} // end get_rxn_pres_mod

