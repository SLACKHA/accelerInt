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

    class EXP4Integrator : public ExponentialIntegrator
    {
    private:
        //! The required memory size of this integrator in bytes.
        //! This is cummulative with any base classes
        std::size_t _ourMemSize;

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

        /** \brief Compute the correct order Phi (exponential) matrix function.
         *         This is dependent on the exponential solver type, and hence must be
         *         overridden in the subclasses.
         *
         *  \param[in]      m       The Hessenberg matrix size (mxm)
         *  \param[in]      A       The input Hessenberg matrix
         *  \param[in]      c       The scaling factor
         *  \param[out]     phiA    The resulting exponential matrix
         */
        inline int exponential(const int m, const double* A, const double c, double* phiA)
        {
            return phiAc_variable (m, A, c, phiA);
        }

        std::size_t setOffsets()
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

        /*
         * \brief Return the required memory size (per-thread) in bytes
         */
        virtual std::size_t requiredSolverMemorySize()
        {
            return _ourMemSize;
        }

    public:

        //! max order of the phi functions (for error estimation)
        static constexpr int P = 1;
        //! order of embedded methods
        static constexpr double ORD = 3.0;
        //! Maximum allowed internal timesteps per integration step
        static constexpr int MAX_STEPS = 100000;
        //! Number of consecutive errors on internal integration steps allowed before exit
        static constexpr int MAX_CONSECUTIVE_ERRORS = 5;


        EXP4Integrator(int neq, int numThreads, const EXPSolverOptions& options) :
            ExponentialIntegrator(neq, numThreads, P, options)
        {
            _ourMemSize = this->setOffsets();
            this->reinitialize(numThreads);
        }

        /*!
           \fn char* solverName()
           \brief Returns a descriptive solver name
        */
        const char* solverName() const {
            const char* name = "exp4-int";
            return name;
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
