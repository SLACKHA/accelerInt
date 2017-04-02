/**
 * \file
 * \brief Header definition for CUDA Jacobian vector multiplier, used in exponential integrators
 *
 */

#ifndef SPARSE_HEAD_CUH
#define SPARSE_HEAD_CUH

#include "header.cuh"


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
void sparse_multiplier (const double * A, const double * Vm, double* w);


#ifdef GENERATE_DOCS
}
#endif

#endif
