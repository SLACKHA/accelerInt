/*!
 * \file exp_solver.cpp
 * \brief Implementation of generic exponential solver methods
 *
 * \author Nicholas Curtis
 * \date 10/22/18
 */

#include "exp_solver.hpp"
#ifdef HAVE_SPARSE_MULTIPLIER
extern "C"
{
    void sparse_multiplier(const double * __restrict__ A, const double * __restrict__ Vm, double * __restrict__ w);
}
#else
#include "lapack_dfns.h"
#endif


namespace c_solvers
{
    /** \brief Compute the matrix-vector multiplication A * Vm -> w
     *         potentially using a user-specified sparse matrix multiplier
     */
    void ExponentialIntegrator::gemv(const double * __restrict__ A, const double * __restrict__ Vm, double * __restrict__ w)
    {
        #ifdef HAVE_SPARSE_MULTIPLIER
        sparse_multiplier(A, Vm, w);
        #else
        dgemv_(&TRANS, &_neq, &_neq, &ALPHA, A, &_neq, Vm, &XINC, &BETA, w, &XINC);
        #endif
    }
}
