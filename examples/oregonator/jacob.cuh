/**
 * \file
 * \brief Contains a header definition for the CUDA van der Pol Jacobian evaluation
 *
 */

#ifdef GENERATE_DOCS
//put this in the van der Pol namespace for documentation
namespace van_der_pol_cu {
#endif


/**
 * \brief An implementation of the van der Pol jacobian
 *
 * \param[in]           t               The current system time
 * \param[in]           mu              The van der Pol parameter
 * \param[in]           y               The state vector at time t
 * \param[out]          jac             The jacobian to populate
 * \param[in]           d_mem           The mechanism_memory struct.  In future versions, this will be used to access the \f$\mu\f$ parameter to have a consistent interface.
 *
 */
__device__
void eval_jacob (const double t, const double mu, const double * __restrict__ y, double * __restrict__ jac, const mechanism_memory * __restrict__ d_mem);

#ifdef GENERATE_DOCS
}
#endif