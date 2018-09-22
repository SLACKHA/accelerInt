/**
 * \file
 * \brief Definition of the RKC CPU solver
 * \author Nicholas Curtis, Kyle Niemeyer
 * \date 09/19/2019
 */

#ifndef RKC_SOLVER_H
#define RKC_SOLVER_H

#include "dydt.h"
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
    protected:
        //! offsets

        std::size_t _work;
        std::size_t _y_n;
        std::size_t _F_n;
        std::size_t _temp_arr;
        std::size_t _temp_arr2;
        std::size_t _y_jm1;
        std::size_t _y_jm2;

        /**
         * \brief Function to estimate spectral radius.
         *
         * \param[in] t     the time.
         * \param[in] pr    A parameter used for pressure or density to pass to the derivative function.
         * \param[in] hmax  Max time step size.
         * \param[in] y     Array of dependent variable.
         * \param[in] F     Derivative evaluated at current state
         * \param[in,out] v Array for eigenvectors
         * \param[out] Fv   Array for derivative evaluations
         */
        double rkc_spec_rad (const double t, const double pr, const double hmax,
                             const double* __restrict__ y, const double* __restrict__ F,
                             double* __restrict__ v, double* __restrict__ Fv);

        /**
         * \brief Function to take a single RKC integration step
         *
         * \param[in] t    the starting time.
         * \param[in] pr   A parameter used for pressure or density to pass to the derivative function.
         * \param[in] h    Time-step size.
         * \param[in] y_0  Initial conditions.
         * \param[in] F_0  Derivative function at initial conditions.
         * \param[in] s    number of steps.
         * \param[out] y_j  Integrated variables.
         */
        void rkc_step (const double t, const double pr, const double h,
                       const double* y_0, const double* F_0, const int s,
                       double* __restrict__ y_j, double* __restrict__ y_jm1,
                       double* __restrict__ y_jm2);

    public:

        RKCIntegrator(int neq, int numThreads, double atol=1e-10, double rtol=1e-6) :
            Integrator(neq, numThreads, atol, rtol)
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

        std::size_t requiredSolverMemorySize()
        {
            std::size_t working = Integrator::requiredSolverMemorySize();
            // work
            _work = working;
            working += (4 + _neq) * sizeof(double);
            // yn
            _y_n = working;
            working += _neq * sizeof(double);
            // fn
            _F_n = working;
            working += _neq * sizeof(double);
            // temp1
            _temp_arr = working;
            working += _neq * sizeof(double);
            // temp2
            _temp_arr2 = working;
            working += _neq * sizeof(double);
            // y_jm1
            _y_jm1 = working;
            working += _neq * sizeof(double);
            // y_jm2
            _y_jm2 = working;
            working += _neq * sizeof(double);

            return working;
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
            const double t_start, const double t_end, const double pr, double* y);

    };
}

#endif
