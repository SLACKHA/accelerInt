/**
 * \file
 * \brief Contains a header definition for the Oregonator Jacobian evaluation
 *
 */

#ifndef JACOB_H
#define JACOB_H

#ifdef GENERATE_DOCS
//put this in the Oregonator namespace for documentation
namespace oregonator {
#endif


/**
 * \brief An implementation of the Oregonator jacobian
 *
 * \param[in]           t               The current system time
 * \param[in]           mu              Dummy parameter, needed for compatibility only
 * \param[in]           y               The state vector at time t
 * \param[out]          jac             The jacobian to populate
 *
 */
void eval_jacob (const double t, const double mu, const double * __restrict__ y, double * __restrict__ jac);

#ifdef GENERATE_DOCS
}

#endif

#endif
