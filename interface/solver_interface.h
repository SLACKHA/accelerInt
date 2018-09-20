/**
 * \file
 * \brief Interface implementation for CPU solvers to be called as a library
 *
 * \author Nicholas Curtis
 * \date 03/09/2015
 *
 * Contains initialization, integration and cleanup header definitions
 */

#ifndef SOLVER_INTERFACE_H
#define SOLVER_INTERFACE_H

#include "solver.h"
#include "solver_init.h"
#include <float.h>

#define EPS DBL_EPSILON

#ifdef GENERATE_DOCS
namespace generic {
#endif

/**
 * \brief Initializes the solver
 * \param[in]       num_threads         The number of OpenMP threads to use
 *
 */
void accelerInt_initialize(int num_threads);


/**
 * \brief integrate NUM odes from time `t` to time `t_end`, using stepsizes of `t_step`
 *
 * \param[in]           NUM             The number of ODEs to integrate.  This should be the size of the leading dimension of `y_host` and `var_host`.  @see accelerint_indx
 * \param[in]           t               The system time
 * \param[in]           t_end           The end time
 * \param[in]           stepsize        The integration step size.  If `stepsize` < 0, the step size will be set to `t_end - t`
 * \param[in,out]       y_host          The state vectors to integrate.
 * \param[in]           var_host        The parameters to use in dydt() and eval_jacob()
 *
 */
void accelerInt_integrate(const int NUM, const double t, const double t_end, const double stepsize,
                          double * __restrict__ y_host, const double * __restrict__ var_host);

/**
 * \brief Cleans up the solver
 * \param[in]       num_threads         The number of OpenMP threads to use
 *
 */
void accelerInt_cleanup(int num_threads);




#ifdef GENERATE_DOCS
}
#endif


#endif