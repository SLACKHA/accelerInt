/**
 * \file
 * \brief A listing of error-codes that may be returned by the solver
 *
 * \author Nicholas Curtis
 * \date 03/10/2015
 *
 *
 */

#ifndef ERROR_CODES_HPP
#define ERROR_CODES_HPP

namespace c_solvers
{
    enum ErrorCode : int
    {
        //! Successful integration step
        SUCCESS = 0,
        //! Maximum consecutive errors on internal integration steps reached
        MAX_CONSECUTIVE_ERRORS_EXCEEDED = 1,
        //! Maximum number of internal integration steps reached
        MAX_STEPS_EXCEEDED = 2,
        //! Timestep reduced such that update would have no effect on simulation time
        H_PLUS_T_EQUALS_H = 3,
        //! Maximum allowed Newton Iteration steps exceeded @see #NewtonMaxit
        MAX_NEWTON_ITER_EXCEEDED = 4
    };
}

#endif
