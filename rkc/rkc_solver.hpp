/**
 * \file
 * \brief Definition of the RKC CPU solver
 * \author Nicholas Curtis, Kyle Niemeyer
 * \date 09/19/2019
 */

#ifndef RADAU2A_SOLVER_H
#define RADAU2A_SOLVER_H

#include "header.h"
#include "solver.hpp"
namespace c_solvers
{

    #define ZERO 0.0
    #define ONE 1.0
    #define TWO 2.0
    #define THREE 3.0
    #define FOUR 4.0

    #define TEN 10.0
    #define ONEP1 1.1
    #define ONEP2 1.2
    #define ONEP54 1.54
    #define P8 0.8
    #define P4 0.4
    #define P1 0.1
    #define P01 0.01
    #define ONE3RD (1.0 / 3.0)
    #define TWO3RD (2.0 / 3.0)
    #define UROUND (2.22e-16)

    class RKCIntegrator : public Integrator
    {
        public:

            RKCIntegrator(int numThreads) : Integrator(numThreads)
            {
            }

            /*!
               \fn char* solverName()
               \brief Returns a descriptive solver name
            */
            const char* solverName() const {
                const char* name = "rkc-int";
                return name;
            }

            void clean() {
                // pass
            }
            void reinitialize(int numThreads) {
                // pass
            }

            void initSolverLog() {
                // pass
            }

            void solverLog() {
                // pass
            }

            std::size_t requiredSolverMemorySize() const
            {
                // C-solvers don't require pre-allocation
                return 0;
            }

            /**
             * \brief Driver function for RKC integrator.
             *
             * \param[in,out] t     The time (starting, then ending).
             * \param[in] tEnd      The desired end time.
             * \param[in] pr        A parameter used for pressure or density to pass to the derivative function.
             * \param[in,out] y     Dependent variable array, integrated values replace initial conditions.
             */
            ErrorCode integrate (
                const double t_start, const double t_end, const double pr, double* y) const;

        };

}
