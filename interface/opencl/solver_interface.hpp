/**
 * \file
 * \brief Interface implementation for CPU solvers to be called as a library
 *
 * \author Nicholas Curtis
 * \date 10/06/18
 *
 * \brief Interface to create / call OpenCL integrators
 *
 */

#ifndef SOLVER_INTERFACE_H
#define SOLVER_INTERFACE_H

#include <memory>
#include <vector>

#include "solver.hpp"
#include "solver_types.hpp"

namespace opencl_solvers
{

    /**
     * \brief Initializes the solver
     * \param[in]       type                The type of solver to use
     * \param[in]       neq                 The number of equations to solve for each IVP
     * \param[in]       numThreads          The number of OpenMP threads to use
     * \param[in]       options             The SolverOptions to use
     * \param[in]       ivp                 The IVP object describing the initial value problem to solve
     * \param[out]      solver              The initialized solver
     */
    std::unique_ptr<IntegratorBase> init(IntegratorType type, int neq, int numThreads,
                                         const IVP& ivp,
                                         const SolverOptions& options);

    /**
     * \brief integrate NUM odes from time `t` to time `t_end`, using stepsizes of `t_step`
     *
     * \param[in]           integrator      The integator object to use
     * \param[in]           NUM             The number of ODEs to integrate.  This should be the size of the leading dimension of `y_host` and `var_host`.  @see accelerint_indx
     * \param[in]           t               The array of system times
     * \param[in]           t_end           The array end times
     * \param[in]           stepsize        The integration step size.  If `stepsize` < 0, the step size will be set to `t_end - t`
     * \param[in,out]       phi_host        The state vectors to integrate.
     * \param[in]           param_host      The parameters to use in dydt() and eval_jacob()
     * \param[out]          last_stepsize   If supplied, store last step-size taken by the integrator for each IVP. Useful for OpenFOAM / chemistry timescale integration
     * \returns             timing          The wall-clock duration spent in integration in milliseconds
     *
     */
    double integrate_varying(IntegratorBase& integrator,
                             const int NUM, const double t_start,
                             const double* __restrict__  t_end, const double stepsize,
                             double * __restrict__ phi_host,
                             const double * __restrict__ param_host,
                             double* __restrict__ last_stepsize=NULL);

    /**
     * \brief integrate NUM odes from time `t` to time `t_end`, using stepsizes of `t_step`
     *
     * \param[in]           integrator      The integator object to use
     * \param[in]           NUM             The number of ODEs to integrate.  This should be the size of the leading dimension of `y_host` and `var_host`.  @see accelerint_indx
     * \param[in]           t               The system time
     * \param[in]           t_end           The end time
     * \param[in]           stepsize        The integration step size.  If `stepsize` < 0, the step size will be set to `t_end - t`
     * \param[in,out]       phi_host        The state vectors to integrate.
     * \param[in]           param_host      The parameters to use in dydt() and eval_jacob()
     * \param[out]          last_stepsize   If supplied, store last step-size taken by the integrator for each IVP. Useful for OpenFOAM / chemistry timescale integration
     * \returns             timing          The wall-clock duration spent in integration in milliseconds
     *
     */
    double integrate(IntegratorBase& integrator,
                     const int NUM, const double t, const double t_end,
                     const double stepsize, double * __restrict__ phi_host,
                     const double * __restrict__ param_host,
                     double* __restrict__ last_stepsize=NULL)
    {
        std::vector<double> tf(NUM, t_end);
        return integrate_varying(integrator, NUM, t, &tf[0], stepsize, phi_host, param_host, last_stepsize);
    }


}

#endif
