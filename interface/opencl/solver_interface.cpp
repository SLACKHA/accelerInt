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
#include "rkf45_solver.hpp"

#include <cmath>
#include <sstream>
#include <chrono>

namespace c_solvers
{

    /**
     * \brief Initializes the solver
     * \param[in]       type                The type of solver to use
     * \param[in]       neq                 The number of equations to solve for each IVP
     * \param[in]       numThreads          The number of OpenMP threads to use
     * \param[in]       options             The SolverOptions to use
     * \param[in]       ivp                 The SolverIVP object describing the initial value problem to solve
     * \param[out]      solver              The initialized solver
     */
    std::unique_ptr<IntegratorBase> init(IntegratorType type, int neq, int numThreads,
                                         const IVP& ivp, const SolverOptions& options)
    {
        switch(type)
        {
            case IntegratorType::RKF45:
                return std::unique_ptr<RKF45Integrator>(new RKF45Integrator(
                    neq, numThreads, ivp, std::static_cast<RKF45SolverOptions>(options)));
            default:
                std::ostringstream ss;
                ss << "Integrator type: " << type << " not implemented for OpenCL!" << std::endl;
                throw std::invalid_argument(ss.str());
        }
    }

    /**
     * \brief Initializes the solver
     * \param[in]       type                The type of solver to use
     * \param[in]       neq                 The number of equations to solve for each IVP
     * \param[in]       numThreads          The number of OpenMP threads to use
     * \param[in]       ivp                 The SolverIVP object describing the initial value problem to solve
     * \param[out]      solver              The initialized solver
     */
    std::unique_ptr<IntegratorBase> init(IntegratorType type, int neq, int numThreads, const IVP& ivp)
    {
        switch(type)
        {
            case IntegratorType::RKF45:
                return init(type, neq, numThreads, RKF45SolverOptions());
            default:
                std::ostringstream ss;
                ss << "Integrator type: " << type << " not implemented for OpenCL!" << std::endl;
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
     * \param[in,out]       phi_host        The state vectors to integrate.
     * \param[in]           param_host      The parameters to use in dydt() and eval_jacob()
     * \returns             timing          The wall-clock duration spent in integration in milliseconds
     *
     */
    double integrate(IntegratorBase& integrator,
                     const int NUM, const double* t_start,
                     const double* t_end, const double step,
                     double * __restrict__ phi_host,
                     const double * __restrict__ param_host)
    {
        auto t1 = std::chrono::high_resolution_clock::now();
        double t = t_start;
        double stepsize = step;
        if (step < 0)
        {
            stepsize = t_end - t_start;
        }

        if (integrator.logging())
        {
            integrator.log(NUM, t_start, y_host);
        }

        int numSteps = 0;
        double t_next = t + step;
        // time integration loop
        while (t + EPS < t_end)
        {
            numSteps++;
            integrator.intDriver(NUM, t, t_next, var_host, y_host);
            t = t_next;
            if (integrator.logging())
            {
                integrator.log(NUM, t, y_host);
            }
            t_next = std::fmin(t_end, (numSteps + 1) * stepsize);
        }
        auto t2 = std::chrono::high_resolution_clock::now();

        return std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    }

}
