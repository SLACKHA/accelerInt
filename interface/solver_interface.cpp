/**
 * \file
 * \brief Interface implementation for CPU solvers to be called as a library
 *
 * \author Nicholas Curtis
 * \date 03/09/2015
 *
 * Contains initialization, integration and cleanup functions
 */

#include "solver_interface.hpp"

#include <cmath>
#include <sstream>

namespace c_solvers
{

    /**
     * \brief Initializes the solver
     * \param[in]       type                The type of solver to use
     * \param[in]       num_threads         The number of OpenMP threads to use
     *
     * \param[out]      solver              The initialized solver
     */
    std::unique_ptr<Integrator> init(IntegratorType type, int num_threads)
    {
        switch(type)
        {
            case IntegratorType::RADAU_II_A:
                return std::unique_ptr<RadauIntegrator>(new RadauIntegrator(num_threads));
            default:
                std::ostringstream ss;
                ss << "Integrator type: " << type << " not implemented for C!" << std::endl;
                throw std::invalid_argument(ss.str());
        }
    }

    /**
     * \brief integrate NUM odes from time `t` to time `t_end`, using stepsizes of `t_step`
     *
     * \param[in]           NUM             The number of ODEs to integrate.
                                            This should be the size of the leading dimension of `y_host` and `var_host`.
                                            @see accelerint_indx
     * \param[in]           t_start         The system time
     * \param[in]           t_end           The end time
     * \param[in]           step            The integration step size.  If `step` < 0, the step size will be set to `t_end - t`
     * \param[in,out]       y_host          The state vectors to integrate.
     * \param[in]           var_host        The parameters to use in dydt() and eval_jacob()
     *
     */
    void integrate(const Integrator& integrator,
                   const int NUM, const double t_start,
                   const double t_end, const double step,
                   double * __restrict__ y_host,
                   const double * __restrict__ var_host)
    {
        double t = t_start;

        if (step < 0)
        {
            // single step
            double t_next = t_end;
            intDriver(integrator, NUM, t_start, t_end, var_host, y_host);
        }

        int numSteps = 0;
        double t_next = t + step;
        // time integration loop
        while (t + EPS < t_end)
        {
            numSteps++;
            intDriver(integrator, NUM, t, t_end, var_host, y_host);
            t = t_next;
            t_next = fmin(t_end, (numSteps + 1) * step);
        }
    }

}
