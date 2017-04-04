/**
 * \file
 * \brief Contains skeleton of all methods that need to be defined on a per solver basis.
 *
 * \author Nicholas Curtis
 * \date 03/10/2015
 *
 *
 */

 #ifndef SOLVER_H
 #define SOLVER_H

 #include "solver_options.h"
 #include "solver_init.h"
 #include "solver_props.h"

#ifdef GENERATE_DOCS
 namespace generic {
#endif

/**
 * \brief Integration driver for the CPU integrators
 * \param[in]       NUM             The (non-padded) number of IVPs to integrate
 * \param[in]       t               The current system time
 * \param[in]       t_end           The IVP integration end time
 * \param[in]       pr_global       The system constant variable (pressures / densities)
 * \param[in,out]   y_global        The system state vectors at time t.
                                    Returns system state vectors at time t_end
 *
 */
 void intDriver (const int NUM, const double t, const double t_end,
                const double* pr_global, double* y_global);

 /**
  * \brief A header definition of the integrate method, that must be implemented by various solvers
  * \param[in]          t_start             the starting IVP integration time
  * \param[in]          t_end               the IVP integration endtime
  * \param[in]          pr                  the IVP constant variable (presssure/density)
  * \param[in,out]      y                   The IVP state vector at time t_start.
                                            At end of this function call, the system state at time t_end is stored here
  */
 int integrate(const double t_start, const double t_end, const double pr, double* y);

#ifdef GENERATE_DOCS
}
#endif

 #endif