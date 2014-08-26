#include "header.h"

/** Function converting species mole fractions to mass fractions.
 *
 * \param[in]  X  array of species mole fractions
 * \param[out] Y  array of species mass fractions
 */
void mole2mass (const Real * X, Real * Y) {

  // average molecular weight
  Real mw_avg = 0.0;
  mw_avg += X[0] * 2.01594;
  mw_avg += X[1] * 1.00797;
  mw_avg += X[2] * 15.9994;
  mw_avg += X[3] * 31.9988;
  mw_avg += X[4] * 17.00737;
  mw_avg += X[5] * 18.01534;
  mw_avg += X[6] * 33.00677;
  mw_avg += X[7] * 34.01474;
  mw_avg += X[8] * 12.01115;
  mw_avg += X[9] * 13.01912;
  mw_avg += X[10] * 14.02709;
  mw_avg += X[11] * 14.02709;
  mw_avg += X[12] * 15.03506;
  mw_avg += X[13] * 16.04303;
  mw_avg += X[14] * 28.01055;
  mw_avg += X[15] * 44.00995;
  mw_avg += X[16] * 29.01852;
  mw_avg += X[17] * 30.02649;
  mw_avg += X[18] * 31.03446;
  mw_avg += X[19] * 31.03446;
  mw_avg += X[20] * 32.04243;
  mw_avg += X[21] * 25.03027;
  mw_avg += X[22] * 26.03824;
  mw_avg += X[23] * 27.04621;
  mw_avg += X[24] * 28.05418;
  mw_avg += X[25] * 29.06215;
  mw_avg += X[26] * 30.07012;
  mw_avg += X[27] * 41.02967;
  mw_avg += X[28] * 42.03764;
  mw_avg += X[29] * 42.03764;
  mw_avg += X[30] * 14.0067;
  mw_avg += X[31] * 15.01467;
  mw_avg += X[32] * 16.02264;
  mw_avg += X[33] * 17.03061;
  mw_avg += X[34] * 29.02137;
  mw_avg += X[35] * 30.0061;
  mw_avg += X[36] * 46.0055;
  mw_avg += X[37] * 44.0128;
  mw_avg += X[38] * 31.01407;
  mw_avg += X[39] * 26.01785;
  mw_avg += X[40] * 27.02582;
  mw_avg += X[41] * 28.03379;
  mw_avg += X[42] * 41.03252;
  mw_avg += X[43] * 43.02522;
  mw_avg += X[44] * 43.02522;
  mw_avg += X[45] * 43.02522;
  mw_avg += X[46] * 42.01725;
  mw_avg += X[47] * 28.0134;
  mw_avg += X[48] * 39.948;
  mw_avg += X[49] * 43.08924;
  mw_avg += X[50] * 44.09721;
  mw_avg += X[51] * 43.04561;
  mw_avg += X[52] * 44.05358;

  // calculate mass fractions
  Y[0] = X[0] * 2.01594 / mw_avg;
  Y[1] = X[1] * 1.00797 / mw_avg;
  Y[2] = X[2] * 15.9994 / mw_avg;
  Y[3] = X[3] * 31.9988 / mw_avg;
  Y[4] = X[4] * 17.00737 / mw_avg;
  Y[5] = X[5] * 18.01534 / mw_avg;
  Y[6] = X[6] * 33.00677 / mw_avg;
  Y[7] = X[7] * 34.01474 / mw_avg;
  Y[8] = X[8] * 12.01115 / mw_avg;
  Y[9] = X[9] * 13.01912 / mw_avg;
  Y[10] = X[10] * 14.02709 / mw_avg;
  Y[11] = X[11] * 14.02709 / mw_avg;
  Y[12] = X[12] * 15.03506 / mw_avg;
  Y[13] = X[13] * 16.04303 / mw_avg;
  Y[14] = X[14] * 28.01055 / mw_avg;
  Y[15] = X[15] * 44.00995 / mw_avg;
  Y[16] = X[16] * 29.01852 / mw_avg;
  Y[17] = X[17] * 30.02649 / mw_avg;
  Y[18] = X[18] * 31.03446 / mw_avg;
  Y[19] = X[19] * 31.03446 / mw_avg;
  Y[20] = X[20] * 32.04243 / mw_avg;
  Y[21] = X[21] * 25.03027 / mw_avg;
  Y[22] = X[22] * 26.03824 / mw_avg;
  Y[23] = X[23] * 27.04621 / mw_avg;
  Y[24] = X[24] * 28.05418 / mw_avg;
  Y[25] = X[25] * 29.06215 / mw_avg;
  Y[26] = X[26] * 30.07012 / mw_avg;
  Y[27] = X[27] * 41.02967 / mw_avg;
  Y[28] = X[28] * 42.03764 / mw_avg;
  Y[29] = X[29] * 42.03764 / mw_avg;
  Y[30] = X[30] * 14.0067 / mw_avg;
  Y[31] = X[31] * 15.01467 / mw_avg;
  Y[32] = X[32] * 16.02264 / mw_avg;
  Y[33] = X[33] * 17.03061 / mw_avg;
  Y[34] = X[34] * 29.02137 / mw_avg;
  Y[35] = X[35] * 30.0061 / mw_avg;
  Y[36] = X[36] * 46.0055 / mw_avg;
  Y[37] = X[37] * 44.0128 / mw_avg;
  Y[38] = X[38] * 31.01407 / mw_avg;
  Y[39] = X[39] * 26.01785 / mw_avg;
  Y[40] = X[40] * 27.02582 / mw_avg;
  Y[41] = X[41] * 28.03379 / mw_avg;
  Y[42] = X[42] * 41.03252 / mw_avg;
  Y[43] = X[43] * 43.02522 / mw_avg;
  Y[44] = X[44] * 43.02522 / mw_avg;
  Y[45] = X[45] * 43.02522 / mw_avg;
  Y[46] = X[46] * 42.01725 / mw_avg;
  Y[47] = X[47] * 28.0134 / mw_avg;
  Y[48] = X[48] * 39.948 / mw_avg;
  Y[49] = X[49] * 43.08924 / mw_avg;
  Y[50] = X[50] * 44.09721 / mw_avg;
  Y[51] = X[51] * 43.04561 / mw_avg;
  Y[52] = X[52] * 44.05358 / mw_avg;

} // end mole2mass

/** Function converting species mass fractions to mole fractions.
 *
 * \param[in]  Y  array of species mass fractions
 * \param[out] X  array of species mole fractions
 */
void mass2mole (const Real * Y, Real * X) {

  // average molecular weight
  Real mw_avg = 0.0;
  mw_avg += Y[0] / 2.01594;
  mw_avg += Y[1] / 1.00797;
  mw_avg += Y[2] / 15.9994;
  mw_avg += Y[3] / 31.9988;
  mw_avg += Y[4] / 17.00737;
  mw_avg += Y[5] / 18.01534;
  mw_avg += Y[6] / 33.00677;
  mw_avg += Y[7] / 34.01474;
  mw_avg += Y[8] / 12.01115;
  mw_avg += Y[9] / 13.01912;
  mw_avg += Y[10] / 14.02709;
  mw_avg += Y[11] / 14.02709;
  mw_avg += Y[12] / 15.03506;
  mw_avg += Y[13] / 16.04303;
  mw_avg += Y[14] / 28.01055;
  mw_avg += Y[15] / 44.00995;
  mw_avg += Y[16] / 29.01852;
  mw_avg += Y[17] / 30.02649;
  mw_avg += Y[18] / 31.03446;
  mw_avg += Y[19] / 31.03446;
  mw_avg += Y[20] / 32.04243;
  mw_avg += Y[21] / 25.03027;
  mw_avg += Y[22] / 26.03824;
  mw_avg += Y[23] / 27.04621;
  mw_avg += Y[24] / 28.05418;
  mw_avg += Y[25] / 29.06215;
  mw_avg += Y[26] / 30.07012;
  mw_avg += Y[27] / 41.02967;
  mw_avg += Y[28] / 42.03764;
  mw_avg += Y[29] / 42.03764;
  mw_avg += Y[30] / 14.0067;
  mw_avg += Y[31] / 15.01467;
  mw_avg += Y[32] / 16.02264;
  mw_avg += Y[33] / 17.03061;
  mw_avg += Y[34] / 29.02137;
  mw_avg += Y[35] / 30.0061;
  mw_avg += Y[36] / 46.0055;
  mw_avg += Y[37] / 44.0128;
  mw_avg += Y[38] / 31.01407;
  mw_avg += Y[39] / 26.01785;
  mw_avg += Y[40] / 27.02582;
  mw_avg += Y[41] / 28.03379;
  mw_avg += Y[42] / 41.03252;
  mw_avg += Y[43] / 43.02522;
  mw_avg += Y[44] / 43.02522;
  mw_avg += Y[45] / 43.02522;
  mw_avg += Y[46] / 42.01725;
  mw_avg += Y[47] / 28.0134;
  mw_avg += Y[48] / 39.948;
  mw_avg += Y[49] / 43.08924;
  mw_avg += Y[50] / 44.09721;
  mw_avg += Y[51] / 43.04561;
  mw_avg += Y[52] / 44.05358;
  mw_avg = 1.0 / mw_avg;

  // calculate mass fractions
  X[0] = Y[0] * mw_avg / 2.01594;
  X[1] = Y[1] * mw_avg / 1.00797;
  X[2] = Y[2] * mw_avg / 15.9994;
  X[3] = Y[3] * mw_avg / 31.9988;
  X[4] = Y[4] * mw_avg / 17.00737;
  X[5] = Y[5] * mw_avg / 18.01534;
  X[6] = Y[6] * mw_avg / 33.00677;
  X[7] = Y[7] * mw_avg / 34.01474;
  X[8] = Y[8] * mw_avg / 12.01115;
  X[9] = Y[9] * mw_avg / 13.01912;
  X[10] = Y[10] * mw_avg / 14.02709;
  X[11] = Y[11] * mw_avg / 14.02709;
  X[12] = Y[12] * mw_avg / 15.03506;
  X[13] = Y[13] * mw_avg / 16.04303;
  X[14] = Y[14] * mw_avg / 28.01055;
  X[15] = Y[15] * mw_avg / 44.00995;
  X[16] = Y[16] * mw_avg / 29.01852;
  X[17] = Y[17] * mw_avg / 30.02649;
  X[18] = Y[18] * mw_avg / 31.03446;
  X[19] = Y[19] * mw_avg / 31.03446;
  X[20] = Y[20] * mw_avg / 32.04243;
  X[21] = Y[21] * mw_avg / 25.03027;
  X[22] = Y[22] * mw_avg / 26.03824;
  X[23] = Y[23] * mw_avg / 27.04621;
  X[24] = Y[24] * mw_avg / 28.05418;
  X[25] = Y[25] * mw_avg / 29.06215;
  X[26] = Y[26] * mw_avg / 30.07012;
  X[27] = Y[27] * mw_avg / 41.02967;
  X[28] = Y[28] * mw_avg / 42.03764;
  X[29] = Y[29] * mw_avg / 42.03764;
  X[30] = Y[30] * mw_avg / 14.0067;
  X[31] = Y[31] * mw_avg / 15.01467;
  X[32] = Y[32] * mw_avg / 16.02264;
  X[33] = Y[33] * mw_avg / 17.03061;
  X[34] = Y[34] * mw_avg / 29.02137;
  X[35] = Y[35] * mw_avg / 30.0061;
  X[36] = Y[36] * mw_avg / 46.0055;
  X[37] = Y[37] * mw_avg / 44.0128;
  X[38] = Y[38] * mw_avg / 31.01407;
  X[39] = Y[39] * mw_avg / 26.01785;
  X[40] = Y[40] * mw_avg / 27.02582;
  X[41] = Y[41] * mw_avg / 28.03379;
  X[42] = Y[42] * mw_avg / 41.03252;
  X[43] = Y[43] * mw_avg / 43.02522;
  X[44] = Y[44] * mw_avg / 43.02522;
  X[45] = Y[45] * mw_avg / 43.02522;
  X[46] = Y[46] * mw_avg / 42.01725;
  X[47] = Y[47] * mw_avg / 28.0134;
  X[48] = Y[48] * mw_avg / 39.948;
  X[49] = Y[49] * mw_avg / 43.08924;
  X[50] = Y[50] * mw_avg / 44.09721;
  X[51] = Y[51] * mw_avg / 43.04561;
  X[52] = Y[52] * mw_avg / 44.05358;

} // end mass2mole

/** Function calculating density from mole fractions.
 *
 * \param[in]  temp  temperature
 * \param[in]  pres  pressure
 * \param[in]  X     array of species mole fractions
 * \return     rho  mixture mass density
 */
Real getDensity (const Real temp, const Real pres, const Real * X) {

  // average molecular weight
  Real mw_avg = 0.0;
  mw_avg += X[0] * 2.01594;
  mw_avg += X[1] * 1.00797;
  mw_avg += X[2] * 15.9994;
  mw_avg += X[3] * 31.9988;
  mw_avg += X[4] * 17.00737;
  mw_avg += X[5] * 18.01534;
  mw_avg += X[6] * 33.00677;
  mw_avg += X[7] * 34.01474;
  mw_avg += X[8] * 12.01115;
  mw_avg += X[9] * 13.01912;
  mw_avg += X[10] * 14.02709;
  mw_avg += X[11] * 14.02709;
  mw_avg += X[12] * 15.03506;
  mw_avg += X[13] * 16.04303;
  mw_avg += X[14] * 28.01055;
  mw_avg += X[15] * 44.00995;
  mw_avg += X[16] * 29.01852;
  mw_avg += X[17] * 30.02649;
  mw_avg += X[18] * 31.03446;
  mw_avg += X[19] * 31.03446;
  mw_avg += X[20] * 32.04243;
  mw_avg += X[21] * 25.03027;
  mw_avg += X[22] * 26.03824;
  mw_avg += X[23] * 27.04621;
  mw_avg += X[24] * 28.05418;
  mw_avg += X[25] * 29.06215;
  mw_avg += X[26] * 30.07012;
  mw_avg += X[27] * 41.02967;
  mw_avg += X[28] * 42.03764;
  mw_avg += X[29] * 42.03764;
  mw_avg += X[30] * 14.0067;
  mw_avg += X[31] * 15.01467;
  mw_avg += X[32] * 16.02264;
  mw_avg += X[33] * 17.03061;
  mw_avg += X[34] * 29.02137;
  mw_avg += X[35] * 30.0061;
  mw_avg += X[36] * 46.0055;
  mw_avg += X[37] * 44.0128;
  mw_avg += X[38] * 31.01407;
  mw_avg += X[39] * 26.01785;
  mw_avg += X[40] * 27.02582;
  mw_avg += X[41] * 28.03379;
  mw_avg += X[42] * 41.03252;
  mw_avg += X[43] * 43.02522;
  mw_avg += X[44] * 43.02522;
  mw_avg += X[45] * 43.02522;
  mw_avg += X[46] * 42.01725;
  mw_avg += X[47] * 28.0134;
  mw_avg += X[48] * 39.948;
  mw_avg += X[49] * 43.08924;
  mw_avg += X[50] * 44.09721;
  mw_avg += X[51] * 43.04561;
  mw_avg += X[52] * 44.05358;

  return pres * mw_avg / (8.31451000e+07 * temp);
} // end getDensity

