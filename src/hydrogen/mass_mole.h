#ifndef MASS_MOLE_H
#define MASS_MOLE_H

#ifdef __cplusplus
  extern "C" {
#endif

void mole2mass (const Real*, Real*);
void mass2mole (const Real*, Real*);
Real getDensity (Real, Real, Real*);

#ifdef __cplusplus
  }
#endif
#endif
