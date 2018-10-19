/**
 * \file
 * \brief Implementation for Jacobian vector multiplication, used in exponential integrators
 *
 */

#include "sparse_multiplier.h"

#ifdef GENERATE_DOCS
//put this in the van der Pol namespace for documentation
namespace van_der_pol {
#endif


/**
 * \brief Implements Jacobian \ vector multiplication in sparse (or unrolled) form
 * \param[in]           A           The (NSP x NSP) Jacobian matrix, see eval_jacob() for details on layout
 * \param[in]           Vm          The (NSP x 1) vector to multiply by
 * \param[out]          w           The (NSP x 1) vector to store the result in, \f$w := A * Vm\f$
 */
void sparse_multiplier(const double * A, const double * Vm, double* w) {
  w[0] =  A[0] * Vm[0] +  A[NSP] * Vm[1];
  w[1] =  A[1] * Vm[0] +  A[NSP + 1] * Vm[1];
}


#ifdef GENERATE_DOCS
}
#endif
