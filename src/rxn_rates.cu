#include <math.h>
#include "header.h"

__device__ void eval_rxn_rates (const Real T, const Real * C, Real * fwd_rxn_rates) {
  register Real logT = log(T);


  fwd_rxn_rates[0] = C[0] * C[5] * exp(3.22754120e+01 - (7.69216995e+03 / T));
  fwd_rxn_rates[1] = C[2] * C[3] * exp(2.63075513e+01 + 0.40478 * logT - (-7.48736077e+02 / T));
  fwd_rxn_rates[2] = C[2] * C[1] * exp(2.89707478e+01 - (3.99956606e+03 / T));
  fwd_rxn_rates[3] = C[0] * C[3] * exp(2.85866095e+01 - 0.053533 * logT - (3.32706731e+03 / T));
  fwd_rxn_rates[4] = C[2] * C[1] * exp(3.44100335e+01 - (9.64666348e+03 / T));
  fwd_rxn_rates[5] = C[0] * C[3] * exp(3.40258987e+01 - 0.053533 * logT - (8.97436602e+03 / T));
  fwd_rxn_rates[6] = C[1] * C[3] * exp(1.91907890e+01 + 1.51 * logT - (1.72603316e+03 / T));
  fwd_rxn_rates[7] = C[4] * C[0] * exp(2.34232839e+01 + 1.1829 * logT - (9.55507805e+03 / T));
  fwd_rxn_rates[8] = C[3] * C[3] * exp(1.04163112e+01 + 2.42 * logT - (-9.71208165e+02 / T));
  fwd_rxn_rates[9] = C[2] * C[4] * exp(1.50329128e+01 + 2.1464 * logT - (7.53063740e+03 / T));
  fwd_rxn_rates[10] = C[1] * exp(4.52701605e+01 - 1.4 * logT - (5.25257556e+04 / T));
  fwd_rxn_rates[11] = C[0] * C[0] * exp(4.40606374e+01 - 1.4234 * logT - (2.21510944e+01 / T));
  fwd_rxn_rates[12] = C[1] * C[9] * exp(4.32112625e+01 - 1.1 * logT - (5.25257556e+04 / T));
  fwd_rxn_rates[13] = C[0] * C[0] * C[9] * exp(4.20017378e+01 - 1.1234 * logT - (2.21510944e+01 / T));
  fwd_rxn_rates[14] = C[1] * C[10] * exp(4.32112625e+01 - 1.1 * logT - (5.25257556e+04 / T));
  fwd_rxn_rates[15] = C[0] * C[0] * C[10] * exp(4.20017378e+01 - 1.1234 * logT - (2.21510944e+01 / T));
  fwd_rxn_rates[16] = C[2] * C[2] * exp(3.63576645e+01 - 0.5 * logT);
  fwd_rxn_rates[17] = C[5] * exp(4.31509343e+01 - 0.93491 * logT - (6.02702601e+04 / T));
  fwd_rxn_rates[18] = C[2] * C[2] * C[9] * exp(3.05680644e+01 - (-8.99751398e+02 / T));
  fwd_rxn_rates[19] = C[5] * C[9] * exp(3.73613450e+01 - 0.43491 * logT - (5.93745344e+04 / T));
  fwd_rxn_rates[20] = C[2] * C[2] * C[10] * exp(3.05680644e+01 - (-8.99751398e+02 / T));
  fwd_rxn_rates[21] = C[5] * C[10] * exp(3.73613450e+01 - 0.43491 * logT - (5.93745344e+04 / T));
  fwd_rxn_rates[22] = C[2] * C[0] * exp(4.29970685e+01 - 1.0 * logT);
  fwd_rxn_rates[23] = C[3] * exp(4.38224602e+01 - 1.0301 * logT - (5.18313166e+04 / T));
  fwd_rxn_rates[24] = C[4] * exp(6.39721672e+01 - 3.322 * logT - (6.07835411e+04 / T));
  fwd_rxn_rates[25] = C[0] * C[3] * exp(5.85301272e+01 - 3.0183 * logT - (4.50771425e+02 / T));
  fwd_rxn_rates[26] = C[4] * C[4] * exp(5.98731945e+01 - 2.44 * logT - (6.04765789e+04 / T));
  fwd_rxn_rates[27] = C[0] * C[3] * C[4] * exp(5.44311720e+01 - 2.1363 * logT - (1.43809259e+02 / T));
  fwd_rxn_rates[28] = C[0] * C[5] * exp(2.91680604e+01 + 0.44 * logT);
  fwd_rxn_rates[29] = C[6] * exp(3.34150001e+01 - 0.049433 * logT - (2.52182000e+04 / T));
  fwd_rxn_rates[30] = C[6] * C[0] * exp(1.48271115e+01 + 2.09 * logT - (-7.30167382e+02 / T));
  fwd_rxn_rates[31] = C[1] * C[5] * exp(1.17897235e+01 + 2.6028 * logT - (2.65552467e+04 / T));
  fwd_rxn_rates[32] = C[6] * C[0] * exp(3.18907389e+01 - (1.48448916e+02 / T));
  fwd_rxn_rates[33] = C[3] * C[3] * exp(2.25013658e+01 + 0.86409 * logT - (1.83206092e+04 / T));
  fwd_rxn_rates[34] = C[6] * C[2] * exp(2.40731699e+01 + 1.0 * logT - (-3.64293641e+02 / T));
  fwd_rxn_rates[35] = C[5] * C[3] * exp(2.06516517e+01 + 1.4593 * logT - (2.62487877e+04 / T));
  fwd_rxn_rates[36] = C[6] * C[3] * exp(3.09948627e+01 - (-2.50098683e+02 / T));
  fwd_rxn_rates[37] = C[4] * C[5] * exp(3.21899484e+01 + 0.18574 * logT - (3.48643603e+04 / T));
  fwd_rxn_rates[38] = C[6] * C[6] * exp(3.36712758e+01 - (6.02954209e+03 / T));
  fwd_rxn_rates[39] = C[7] * C[5] * exp(3.74200513e+01 - 0.23447 * logT - (2.53213594e+04 / T));
  fwd_rxn_rates[40] = C[6] * C[6] * exp(2.55908003e+01 - (-8.19890914e+02 / T));
  fwd_rxn_rates[41] = C[7] * C[5] * exp(2.93395801e+01 - 0.23447 * logT - (1.84720774e+04 / T));
  fwd_rxn_rates[42] = C[7] * exp(2.83241683e+01 + 0.9 * logT - (2.45313092e+04 / T));
  fwd_rxn_rates[43] = C[3] * C[3] * exp(1.09390890e+01 + 2.488 * logT - (-1.80654783e+03 / T));
  fwd_rxn_rates[44] = C[7] * C[0] * exp(3.08132330e+01 - (1.99777016e+03 / T));
  fwd_rxn_rates[45] = C[4] * C[3] * exp(1.88701627e+01 + 1.2843 * logT - (3.59925720e+04 / T));
  fwd_rxn_rates[46] = C[7] * C[0] * exp(3.15063801e+01 - (4.00057249e+03 / T));
  fwd_rxn_rates[47] = C[6] * C[1] * exp(2.47202181e+01 + 0.74731 * logT - (1.19941692e+04 / T));
  fwd_rxn_rates[48] = C[7] * C[2] * exp(1.60720517e+01 + 2.0 * logT - (1.99777016e+03 / T));
  fwd_rxn_rates[49] = C[3] * C[6] * exp(8.90174786e+00 + 2.6938 * logT - (9.31906943e+03 / T));
  fwd_rxn_rates[50] = C[7] * C[3] * exp(2.81849062e+01 - (1.60022900e+02 / T));
  fwd_rxn_rates[51] = C[6] * C[4] * exp(2.56312037e+01 + 0.42021 * logT - (1.59826645e+04 / T));
  fwd_rxn_rates[52] = C[7] * C[3] * exp(3.19604378e+01 - (3.65838516e+03 / T));
  fwd_rxn_rates[53] = C[6] * C[4] * exp(2.94067528e+01 + 0.42021 * logT - (1.94810268e+04 / T));
} // end eval_rxn_rates

