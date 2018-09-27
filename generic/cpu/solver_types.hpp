/**
 * \file
 * \brief A listing of available solver types
 *
 * \author Nicholas Curtis
 * \date 03/10/2015
 *
 *
 */

#ifndef SOLVER_TYPE_HPP
#define SOLVER_TYPE_HPP

namespace c_solvers
{
    enum IntegratorType : int
    {
        //! RadauII-A
        RADAU_II_A = 0,
        //! exp4
        EXP4 = 1,
        //! exprb43
        EXPRB43 = 2,
        //! RK-78
        RK_78 = 3,
        //! RKC
        RKC = 4,
        //! CVODES
        CVODES = 5
    };
}

#endif
