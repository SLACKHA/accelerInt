/**
 * \file
 * \brief A listing of error-codes that may be returned by the OpenCL solvers
 *
 * \author Nicholas Curtis
 * \date 10/06/18
 *
 *
 */

#ifndef ERROR_CODES_HPP
#define ERROR_CODES_HPP

namespace opencl_solvers
{
    enum ErrorCode : int
    {
        //! Successful integration step
        SUCCESS = 0,
        //! Maximum consecutive errors on internal integration steps reached
        TOO_MUCH_WORK = -1,
        //! Integration timestep smaller than roundoff
        TDIST_TOO_SMALL = -2,
        //! Maximum number of internal integration steps reached
        MAX_STEPS_EXCEEDED = -3
    };
}

#endif
