#include "head.h"

/** Function converting species mole fractions to mass fractions.
 *
 * \param[in]  X  array of species mole fractions
 * \param[out] Y  array of species mass fractions
 */
void mole2mass (const Real * X, Real * Y) {

  // average molecular weight
  Real mw_avg = 0.0;
  mw_avg += X[0] * 1.00797;
  mw_avg += X[1] * 2.01594;
  mw_avg += X[2] * 15.9994;
  mw_avg += X[3] * 17.00737;
  mw_avg += X[4] * 18.01534;
  mw_avg += X[5] * 31.9988;
  mw_avg += X[6] * 33.00677;
  mw_avg += X[7] * 34.01474;
  mw_avg += X[8] * 28.0134;
  mw_avg += X[9] * 39.948;
  mw_avg += X[10] * 4.0026;
  mw_avg += X[11] * 28.01055;
  mw_avg += X[12] * 44.00995;

  // calculate mass fractions
  Y[0] = X[0] * 1.00797 / mw_avg;
  Y[1] = X[1] * 2.01594 / mw_avg;
  Y[2] = X[2] * 15.9994 / mw_avg;
  Y[3] = X[3] * 17.00737 / mw_avg;
  Y[4] = X[4] * 18.01534 / mw_avg;
  Y[5] = X[5] * 31.9988 / mw_avg;
  Y[6] = X[6] * 33.00677 / mw_avg;
  Y[7] = X[7] * 34.01474 / mw_avg;
  Y[8] = X[8] * 28.0134 / mw_avg;
  Y[9] = X[9] * 39.948 / mw_avg;
  Y[10] = X[10] * 4.0026 / mw_avg;
  Y[11] = X[11] * 28.01055 / mw_avg;
  Y[12] = X[12] * 44.00995 / mw_avg;

} // end mole2mass

/** Function converting species mass fractions to mole fractions.
 *
 * \param[in]  Y  array of species mass fractions
 * \param[out] X  array of species mole fractions
 */
void mass2mole (const Real * Y, Real * X) {

  // average molecular weight
  Real mw_avg = 0.0;
  mw_avg += Y[0] / 1.00797;
  mw_avg += Y[1] / 2.01594;
  mw_avg += Y[2] / 15.9994;
  mw_avg += Y[3] / 17.00737;
  mw_avg += Y[4] / 18.01534;
  mw_avg += Y[5] / 31.9988;
  mw_avg += Y[6] / 33.00677;
  mw_avg += Y[7] / 34.01474;
  mw_avg += Y[8] / 28.0134;
  mw_avg += Y[9] / 39.948;
  mw_avg += Y[10] / 4.0026;
  mw_avg += Y[11] / 28.01055;
  mw_avg += Y[12] / 44.00995;
  mw_avg = 1.0 / mw_avg;

  // calculate mass fractions
  X[0] = Y[0] * mw_avg / 1.00797;
  X[1] = Y[1] * mw_avg / 2.01594;
  X[2] = Y[2] * mw_avg / 15.9994;
  X[3] = Y[3] * mw_avg / 17.00737;
  X[4] = Y[4] * mw_avg / 18.01534;
  X[5] = Y[5] * mw_avg / 31.9988;
  X[6] = Y[6] * mw_avg / 33.00677;
  X[7] = Y[7] * mw_avg / 34.01474;
  X[8] = Y[8] * mw_avg / 28.0134;
  X[9] = Y[9] * mw_avg / 39.948;
  X[10] = Y[10] * mw_avg / 4.0026;
  X[11] = Y[11] * mw_avg / 28.01055;
  X[12] = Y[12] * mw_avg / 44.00995;

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
  mw_avg += X[0] * 1.00797;
  mw_avg += X[1] * 2.01594;
  mw_avg += X[2] * 15.9994;
  mw_avg += X[3] * 17.00737;
  mw_avg += X[4] * 18.01534;
  mw_avg += X[5] * 31.9988;
  mw_avg += X[6] * 33.00677;
  mw_avg += X[7] * 34.01474;
  mw_avg += X[8] * 28.0134;
  mw_avg += X[9] * 39.948;
  mw_avg += X[10] * 4.0026;
  mw_avg += X[11] * 28.01055;
  mw_avg += X[12] * 44.00995;

  Real rho = pres * mw_avg / (8.31451000e+07 * temp);
  return rho;

} // end getDensity

