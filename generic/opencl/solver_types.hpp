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
        //! 4th order Runge-Kutta-Feldberg
        RKF45 = 0,
        //! 3rd order linearly-implicit Rosenbrock integrator
        ROS3 = 1,
        //! 3rd order linearly-implicit RODAS integrator
        RODAS3 = 2,
        //! 4th order linearly-implicit Rosenbrock integrator
        ROS4 = 3,
        //! 4th order linearly-implicit RODAS integrator
        RODAS4 = 4
    };
}

#endif
