/**
 * \file
 * \brief Definition of the RadauIIA CPU solver
 * \author Nicholas Curtis
 * \date 09/19/2019
 */

#ifndef RK78_SOLVER_H
#define RK78_SOLVER_H

#include "dydt.h"
#include "solver.hpp"

#include <cstdio>
#include <cstdlib>

#include <boost/numeric/odeint.hpp>
using namespace boost::numeric::odeint;

//! boost stepper type
typedef runge_kutta_fehlberg78<double* , double> stepper;

//! boost step size controller
typedef controlled_runge_kutta<stepper> controller;

namespace c_solvers
{

    /**
       \brief A wrapper class to evaluate the rhs function y' = f(y)
       stores the state variable, and provides to dydt
    */
    class RK78 {
        double m_statevar;
    public:
        RK78() {
            this->m_statevar = -1;
        }

        void set_state_var(const double state_var)
        {
            this->m_statevar = state_var;
        }

        //wrapper for the pyJac RHS fn
        void operator() (const double* y , double* fy , const double t) const
        {
            dydt(t, this->m_statevar, &y[0], &fy[0]);
        }
    };

    class RK78Integrator : public Integrator
    {
        protected:
            RK78 wrapper;
            std::vector<std::unique_ptr<RK78>> evaluators;
            std::vector<std::unique_ptr<stepper>> steppers;
            std::vector<controller> controllers;

        public:

            RK78Integrator(int neq, int numThreads, double atol=1e-10, double rtol=1e-6) :
                Integrator(neq, numThreads, atol, rtol)
            {
                this->reinitialize(numThreads);
            }

            ~RK78Integrator()
            {
                this->clean();
            }

            /*!
               \fn char* solver_name()
               \brief Returns a descriptive solver name
            */
            const char* solverName() const {
                const char* name = "rk78-int";
                return name;
            }

            void clean() {
                evaluators.clear();
                steppers.clear();
                controllers.clear();
            }
            void reinitialize(int numThreads) {
                for (int i = 0; i < numThreads; ++i)
                {
                    // initialize boost interface
                    evaluators.push_back(std::unique_ptr<RK78>(new RK78()));
                    steppers.push_back(std::unique_ptr<stepper>(new stepper()));
                    controllers.push_back(make_controlled<stepper>(atol(), rtol(), *steppers[i]));
                }
            }

            void initSolverLog() {
                // pass
            }

            void solverLog() {
                // pass
            }

            std::size_t requiredSolverMemorySize()
            {
                // C-solvers don't require pre-allocation
                return 0;
            }

            /**
             *  \brief 5th-order Radau2A CPU implementation
             *
             *  \param[in]          t_start             The starting time
             *  \param[in]          t_end               The end integration time
             *  \param[in]          pr                  The system constant variable (pressure/density)
             *  \param[in,out]      y                   The system state vector at time `t_start`.
                                                        Overwritten with the system state at time `t_end`
             *  \returns Return code, @see RK_ErrCodes
             */
            ErrorCode integrate (
                const double t_start, const double t_end, const double pr, double* y);

        };
}

#endif
