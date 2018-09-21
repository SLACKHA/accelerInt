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
#include "radau2a_solver.hpp"
#include "rk78_solver.hpp"
#include "rkc_solver.hpp"

#include <cmath>
#include <sstream>
#include <chrono>

namespace c_solvers
{

    /**
     * \brief Initializes the solver
     * \param[in]       type                The type of solver to use
     * \param[in]       neq                 The number of equations to solve
     * \param[in]       numThreads          The number of OpenMP threads to use
     *
     * \param[out]      solver              The initialized solver
     */
    std::unique_ptr<Integrator> init(IntegratorType type, int neq, int numThreads)
    {
        switch(type)
        {
            case IntegratorType::RADAU_II_A:
                return std::unique_ptr<RadauIntegrator>(new RadauIntegrator(neq, numThreads));
            case IntegratorType::RK_78:
                return std::unique_ptr<RK78Integrator>(new RK78Integrator(neq, numThreads));
            case IntegratorType::RKC:
                return std::unique_ptr<RKCIntegrator>(new RKCIntegrator(neq, numThreads));
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
     * \returns             timing          The wall-clock duration spent in integration in milliseconds
     *
     */
    double integrate(Integrator& integrator,
                     const int NUM, const double t_start,
                     const double t_end, const double step,
                     double * __restrict__ y_host,
                     const double * __restrict__ var_host)
    {

        auto t1 = std::chrono::high_resolution_clock::now();
        double t = t_start;
        double stepsize = step;
        if (step < 0)
        {
            stepsize = t_end - t_start;
        }

        int numSteps = 0;
        double t_next = t + step;
        // time integration loop
        while (t + EPS < t_end)
        {
            numSteps++;
            intDriver(integrator, NUM, t, t_end, var_host, y_host);
            t = t_next;
            t_next = fmin(t_end, (numSteps + 1) * stepsize);
        }
        auto t2 = std::chrono::high_resolution_clock::now();

        return std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t2).count();
    }

}
