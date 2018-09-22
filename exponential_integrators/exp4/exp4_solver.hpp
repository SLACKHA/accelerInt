/**
 * \file
 * \brief Definition of the RKC CPU solver
 * \author Nicholas Curtis, Kyle Niemeyer
 * \date 09/19/2019
 */

#ifndef EXP4_SOLVER_HPP
#define EXP4_SOLVER_HPP

#include "exp_solver.hpp"

namespace c_solvers
{

    //! max order of the phi functions (for error estimation)
    #define P 1
    //! order of embedded methods
    #define ORD 3.0
    //! Maximum allowed internal timesteps per integration step
    #define MAX_STEPS (100000)
    //! Number of consecutive errors on internal integration steps allowed before exit
    #define MAX_CONSECUTIVE_ERRORS (5)


    class EXP4Integrator : public ExponentialIntegrator
    {
    protected:
        // log format t, h, err, m, m1, m2
        std::vector<std::tuple<double, double, double, int, int, int>> subspaceLog;

        //! offsets

        //! offset for scaling vector
        std::size_t _sc;
        // source vector offset
        std::size_t _fy;
        // Jacobian matrix offset
        std::size_t _A;
        // temporary array offsets
        std::size_t _temp;
        std::size_t _ftemp;
        std::size_t _y1;
        std::size_t _Hm;
        std::size_t _Vm;
        std::size_t _phiHm;

        // i-vectors
        std::size_t _k1;
        std::size_t _k2;
        std::size_t _k3;
        std::size_t _k4;
        std::size_t _k5;
        std::size_t _k6;
        std::size_t _k7;

    public:

        EXP4Integrator(int neq, int numThreads,
                       double atol=1e-10, double rtol=1e-6,
                       int N_RA=10, int M_MAX=-1) :
            ExponentialIntegrator(neq, numThreads, M_MAX + P, atol, rtol, N_RA, M_MAX)
        {

        }

        /*!
           \fn char* solverName()
           \brief Returns a descriptive solver name
        */
        const char* solverName() const {
            const char* name = "exp4-int";
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
            std::size_t working = ExponentialIntegrator::requiredSolverMemorySize();
            // sc
            _sc = working;
            working += _neq * sizeof(double);
            // fy
            _fy = working;
            working += _neq * sizeof(double);
            // A
            _A = working;
            working += _neq * _neq * sizeof(double);
            // temp
            _temp = working;
            working += _neq * sizeof(double);
            // _ftemp
            _ftemp = working;
            working += _neq * sizeof(double);
            // y1
            _y1 = working;
            working += _neq * sizeof(double);
            // _Hm
            _Hm = working;
            working += STRIDE * STRIDE * sizeof(double);
            // _Vm
            _Vm = working;
            working += _neq * STRIDE * sizeof(double);
            // _phiHm
            _phiHm = working;
            working += STRIDE * STRIDE * sizeof(double);
            // k1
            _k1 = working;
            working += _neq * sizeof(double);
            // k2
            _k2 = working;
            working += _neq * sizeof(double);
            // k3
            _k3 = working;
            working += _neq * sizeof(double);
            // k4
            _k4 = working;
            working += _neq * sizeof(double);
            // k5
            _k5 = working;
            working += _neq * sizeof(double);
            // k6
            _k6 = working;
            working += _neq * sizeof(double);
            // k7
            _k7 = working;
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
