/**
 * \file
 * \brief Interface implementation for CPU solvers to be called as a library
 *
 * \author Nicholas Curtis
 * \date 03/09/2015
 *
 * Contains initialization, integration and cleanup functions
 */

#include "solver_interface.h"
#include <math.h>

#ifdef GENERATE_DOCS
namespace generic {
#endif

/**
 * \brief Initializes the solver
 * \param[in]       num_threads         The number of OpenMP threads to use
 *
 */
void accelerInt_initialize(int num_threads) {
    initialize_solver(num_threads);
}


/**
 * \brief integrate NUM odes from time `t` to time `t_end`, using stepsizes of `t_step`
 *
 * \param[in]           NUM             The number of ODEs to integrate.  This should be the size of the leading dimension of `y_host` and `var_host`.  @see accelerint_indx
 * \param[in]           t_start         The system time
 * \param[in]           t_end           The end time
 * \param[in]           stepsize        The integration step size.  If `stepsize` < 0, the step size will be set to `t_end - t`
 * \param[in,out]       y_host          The state vectors to integrate.
 * \param[in]           var_host        The parameters to use in dydt() and eval_jacob()
 *
 */
void accelerInt_integrate(const int NUM, const double t_start, const double t_end, const double stepsize,
                          double * __restrict__ y_host, const double * __restrict__ var_host)
{
    double t = t_start;
    double step = stepsize < 0 ? t_end - t : stepsize;
    double t_next = fmin(end_time, t + step);
    int numSteps = 0;

    // time integration loop
    while (t + EPS < t_end)
    {
        numSteps++;
        intDriver(NUM, t, t_end, var_host, y_host);
        t = t_next;
        t_next = fmin(t_end, (numSteps + 1) * step);
    }
}


/**
 * \brief Cleans up the solver
 * \param[in]       num_threads         The number of OpenMP threads to use
 *
 */
void accelerInt_cleanup(int num_threads) {
    cleanup_solver(num_threads);
}




#ifdef GENERATE_DOCS
}
#endif