/**
 * \file
 * \brief A listing of the time-step adjustment methods that may be used by the
 *        integrators
 *
 * \author Nicholas Curtis
 * \date 10/24/18
 *
 *
 */

#ifndef STEPPER_TYPES_HPP
#define STEPPER_TYPES_HPP

namespace opencl_solvers
{
    enum StepperType : int
    {
        //! Adaptive time-stepping
        ADAPTIVE = 0,
        //! Constant time-stepping
        CONSTANT = 1
    };
}

#endif
