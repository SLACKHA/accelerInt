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
#include <cfloat>

namespace opencl_solvers
{

    /*! Machine precision constant. */
    #define EPS DBL_EPSILON
    /*! Smallest representable double */
    #define SMALL DBL_MIN

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
                    neq, numThreads, ivp, static_cast<const RKF45SolverOptions&>(options)));
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
                return init(type, neq, numThreads, ivp, RKF45SolverOptions());
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
    double integrate_varying(IntegratorBase& integrator,
                             const int NUM, const double t_start,
                             const double* __restrict__ t_end, const double step,
                             double * __restrict__ phi_host,
                             const double * __restrict__ param_host)
    {
        auto t1 = std::chrono::high_resolution_clock::now();
        double t = t_start;
        double stepsize = step;
        double t_max = *std::max_element(t_end, t_end + NUM);
        if (step < 0)
        {
            stepsize = t_max - t_start;
        }
        if (integrator.logging())
        {
            integrator.log(NUM, t_start, phi_host);
        }

        std::vector<double> times(t_end, t_end + NUM);
        // time integration loop
        while (t + EPS < t_max)
        {
            // update times
            std::for_each(times.begin(), times.end(), [stepsize](double& d) { d+=stepsize;});
            integrator.intDriver(NUM, t, &times[0], param_host, phi_host);
            if (integrator.logging())
            {
                integrator.log(NUM, t, phi_host);
            }
            t = std::fmin(t_max, t + stepsize);
        }
        auto t2 = std::chrono::high_resolution_clock::now();

        return std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    }

}
