/**
 * \file
 * \brief Contains a header definition for the van der Pol Jacobian evaluation
 *
 */

#ifndef JACOB_H
#define JACOB_H

#ifdef GENERATE_DOCS
//put this in the van der Pol namespace for documentation
namespace van_der_pol {
#endif


/**
 * \brief An implementation of the van der Pol jacobian
 *
 * \param[in]           t               The current system time
 * \param[in]           mu              The van der Pol parameter
 * \param[in]           y               The state vector at time t
 * \param[out]          jac             The jacobian to populate
 *
 */
void eval_jacob (const double t, const double mu, const double * __restrict__ y, double * __restrict__ jac);

#ifdef GENERATE_DOCS
}

#endif

#endif