#include "header.h"
#include "chem_utils.cuh"
#include "rates.cuh"

#if defined(CONP)

__device__ void dydt (const Real t, const Real pres, const Real * y, Real * dy) {

  // mass-averaged density
  Real rho;
  rho = (y[1] / 1.00797) + (y[2] / 2.01594) + (y[3] / 15.9994) + (y[4] / 17.00737)
      + (y[5] / 18.01534) + (y[6] / 31.9988) + (y[7] / 33.00677) + (y[8] / 34.01474)
      + (y[9] / 28.0134) + (y[10] / 39.948) + (y[11] / 4.0026) + (y[12] / 28.01055)
      + (y[13] / 44.00995);
  rho = pres / (8.31451000e+07 * y[0] * rho);

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

  // local array holding reaction rates
  Real rates[54];
  eval_rxn_rates (y[0], conc, rates);

  // get pressure modifications to reaction rates
  Real pres_mod[12];
  get_rxn_pres_mod (y[0], pres, conc, pres_mod);

  // evaluate rate of change of species molar concentration
  eval_spec_rates (rates, &dy[1] );

  // local array holding constant pressure specific heat
  Real cp[13];
  eval_cp (y[0], cp);

  // constant pressure mass-average specific heat
  Real cp_avg = (cp[0] * y[1]) + (cp[1] * y[2]) + (cp[2] * y[3]) + (cp[3] * y[4])
              + (cp[4] * y[5]) + (cp[5] * y[6]) + (cp[6] * y[7]) + (cp[7] * y[8])
              + (cp[8] * y[9]) + (cp[9] * y[10]) + (cp[10] * y[11]) + (cp[11] * y[12])
              + (cp[12] * y[13]);

  // local array for species enthalpies
  Real h[13];
  eval_h (y[0], h);

  // rate of change of temperature
  dy[0] = (-1.0 / (rho * cp_avg)) * ( (dy[1] * h[0] * 1.00797) + (dy[2] * h[1] * 2.01594)
        + (dy[3] * h[2] * 15.9994) + (dy[4] * h[3] * 17.00737) + (dy[5] * h[4] * 18.01534)
        + (dy[6] * h[5] * 31.9988) + (dy[7] * h[6] * 33.00677) + (dy[8] * h[7] * 34.01474)
        + (dy[9] * h[8] * 28.0134) + (dy[10] * h[9] * 39.948) + (dy[11] * h[10] * 4.0026)
        + (dy[12] * h[11] * 28.01055) + (dy[13] * h[12] * 44.00995) );

  // calculate rate of change of species mass fractions
  dy[1] *= (1.00797 / rho);
  dy[2] *= (2.01594 / rho);
  dy[3] *= (15.9994 / rho);
  dy[4] *= (17.00737 / rho);
  dy[5] *= (18.01534 / rho);
  dy[6] *= (31.9988 / rho);
  dy[7] *= (33.00677 / rho);
  dy[8] *= (34.01474 / rho);
  dy[9] *= (28.0134 / rho);
  dy[10] *= (39.948 / rho);
  dy[11] *= (4.0026 / rho);
  dy[12] *= (28.01055 / rho);
  dy[13] *= (44.00995 / rho);

} // end dydt

#elif defined(CONV)

__device__ void dydt (const Real t, const Real rho, const Real * y, Real * dy) {

  // pressure
  Real pres;
  pres = (y[1] / 1.00797) + (y[2] / 2.01594) + (y[3] / 15.9994) + (y[4] / 17.00737)
       + (y[5] / 18.01534) + (y[6] / 31.9988) + (y[7] / 33.00677) + (y[8] / 34.01474)
       + (y[9] / 28.0134) + (y[10] / 39.948) + (y[11] / 4.0026) + (y[12] / 28.01055)
       + (y[13] / 44.00995);
  pres = rho * 8.31451000e+07 * y[0] * pres;

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

  // local array holding reaction rates
  Real rates[54];
  eval_rxn_rates (y[0], pres, conc, rates);

  // evaluate rate of change of species molar concentration
  eval_spec_rates (rates, &dy[1]);

  // local array holding constant volume specific heat
  Real cv[13];
  eval_cv (y[0], cv);

  // constant volume mass-average specific heat
  Real cv_avg = (cv[0] * y[1]) + (cv[1] * y[2]) + (cv[2] * y[3]) + (cv[3] * y[4])
              + (cv[4] * y[5]) + (cv[5] * y[6]) + (cv[6] * y[7]) + (cv[7] * y[8])
              + (cv[8] * y[9]) + (cv[9] * y[10]) + (cv[10] * y[11]) + (cv[11] * y[12])
              + (cv[12] * y[13]);

  // local array for species internal energies
  Real u[13];
  eval_u (y[0], u);

  // rate of change of temperature
  dy[0] = (-1.0 / (rho * cv_avg)) * ( (dy[1] * u[0] * 1.00797) + (dy[2] * u[1] * 2.01594)
        + (dy[3] * u[2] * 15.9994) + (dy[4] * u[3] * 17.00737) + (dy[5] * u[4] * 18.01534)
        + (dy[6] * u[5] * 31.9988) + (dy[7] * u[6] * 33.00677) + (dy[8] * u[7] * 34.01474)
        + (dy[9] * u[8] * 28.0134) + (dy[10] * u[9] * 39.948) + (dy[11] * u[10] * 4.0026)
        + (dy[12] * u[11] * 28.01055) + (dy[13] * u[12] * 44.00995) );

  // calculate rate of change of species mass fractions
  dy[1] *= (1.00797 / rho);
  dy[2] *= (2.01594 / rho);
  dy[3] *= (15.9994 / rho);
  dy[4] *= (17.00737 / rho);
  dy[5] *= (18.01534 / rho);
  dy[6] *= (31.9988 / rho);
  dy[7] *= (33.00677 / rho);
  dy[8] *= (34.01474 / rho);
  dy[9] *= (28.0134 / rho);
  dy[10] *= (39.948 / rho);
  dy[11] *= (4.0026 / rho);
  dy[12] *= (28.01055 / rho);
  dy[13] *= (44.00995 / rho);

} // end dydt

#endif
