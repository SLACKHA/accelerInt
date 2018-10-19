/**
 * \file
 * \brief Contains a header definition for the IVP Jacobian function
 *
 */

#ifndef JACOB_H
#define JACOB_H


/**
 * \brief The expected signature of Jacobian function of the IVP
 *
 * \param[in]           t               The current system time
 * \param[in]           param           The van der Pol parameter
 * \param[in]           y               The state vector at time t
 * \param[out]          jac             The jacobian to populate
 * \param[in]           rwk             The working buffer for the source term evaluation of the jacobian
 *
 */
void eval_jacob (const double t, const double param, const double * __restrict__ y, double * __restrict__ jac,
                 double * __restrict__ rwk);

#endif
