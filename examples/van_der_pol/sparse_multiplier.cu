/**
 * \file
 * \brief Implementation for CUDA Jacobian vector multiplication, used in exponential integrators
 *
 */

#include "sparse_multiplier.cuh"

#ifdef GENERATE_DOCS
//put this in the van der Pol namespace for documentation
namespace van_der_pol_cu {
#endif


/**
 * \brief Implements Jacobian \ vector multiplication in sparse (or unrolled) form
 * \param[in]           A           The (NSP x NSP) Jacobian matrix, see eval_jacob() for details on layout
 * \param[in]           Vm          The (NSP x 1) vector to multiply by
 * \param[out]          w           The (NSP x 1) vector to store the result in, \f$w := A * Vm\f$
 */
__device__
void sparse_multiplier(const double * A, const double * Vm, double* w) {
  w[INDEX(0)] =  A[INDEX(0)] * Vm[INDEX(0)] +  A[INDEX(NSP)] * Vm[INDEX(1)];
  w[INDEX(1)] =  A[INDEX(1)] * Vm[INDEX(0)] +  A[INDEX(NSP + 1)] * Vm[INDEX(1)];
}


#ifdef GENERATE_DOCS
}
#endif
