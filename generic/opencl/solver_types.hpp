/**
 * \file
 * \brief A listing of available solver types for OpenCL
 *
 * \author Nicholas Curtis
 * \date 09/27/2018
 *
 *
 */

#ifndef SOLVER_TYPE_HPP
#define SOLVER_TYPE_HPP

namespace opencl_solvers
{
    enum IntegratorType : int
    {
        //! 4th order linearly-implicit Rosenbrock integrator
        ROSENBROCK = 0,
        //! 4th order Runge-Kutta-Feldberg
        RKF45 = 1
    };
}

#endif
