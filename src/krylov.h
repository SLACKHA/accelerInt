#ifndef EXP4_KRYLOV_HEAD
#define EXP4_KRYLOV_HEAD

#include "header.h"

void exp4_krylov_int (const Real, const Real, const Real, Real*);

//unit test exposed methods
#ifdef COMPILE_TESTING_METHODS
void matvec_m_by_m_test (const int, const Real *, const Real *, Real *);
void matvec_n_by_m_test (const int, const Real *, const Real *, Real *);
double dotproduct_test(const Real*, const Real*);
Real normalize_test(const Real*, Real*);
void scale_subtract_test(const Real, const Real*, Real*);
Real two_norm_test(const Real*);
void scale_mult_test(const Real, const Real*, Real*);
#endif

#endif