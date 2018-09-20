/**
 * \file
 * \brief Contains skeleton of all methods that need to be defined on a per solver basis.
 *
 * \author Nicholas Curtis
 * \date 03/10/2015
 *
 *
 */

 #ifndef SOLVER_H
 #define SOLVER_H

#include <float.h>
#include "error_codes.hpp"

namespace c_solvers {

    // something for solvers to inherit from
    // contains memory & properties for each individual solver
    class Solver
    {
        #ifndef RTOL
            //! default relative tolerance
            #define RTOL (1e-6)
        #endif
        static constexpr double rtol = RTOL;
        #ifndef ATOL
            //! default absolute tolerance
            #define ATOL (1e-10)
        #endif
        static constexpr double atol = ATOL;

        /*! Machine precision constant. */
        #define EPS DBL_EPSILON
        static constexpr double eps = EPS;
        /*! Smallest representable double */
        #define SMALL DBL_MIN
        static constexpr double small = SMALL;
    };

    // skeleton of the c-solver
    class Integrator
    {
        public:
            Integrator(int numThreads) : numThreads(numThreads)
            {

            }
            virtual ~Integrator();

            virtual void initSolverLog() = 0;
            virtual void log() = 0;

            virtual void reinitialize(int numThreads) = 0;
            virtual void clean() = 0;

            virtual const char* solverName() const = 0;
            virtual void checkError(int, ErrorCode) const = 0;

            virtual size_t requiredSolverMemorySize() const = 0;

            /**
            * \brief A header definition of the integrate method, that must be implemented by various solvers
            * \param[in]          t_start             the starting IVP integration time
            * \param[in]          t_end               the IVP integration endtime
            * \param[in]          pr                  the IVP constant variable (presssure/density)
            * \param[in,out]      y                   The IVP state vector at time t_start.
                                                      At end of this function call, the system state at time t_end is stored here
            */
            ErrorCode integrate(const double t_start, const double t_end,
                                const double pr, double* y) const;

            Solver solver_type;

        protected:
            int numThreads;
    };

    /**
    * \brief Integration driver for the CPU integrators
    * \param[in]       int             An instance of an integrator class to use
    * \param[in]       NUM             The (non-padded) number of IVPs to integrate
    * \param[in]       t               The current system time
    * \param[in]       t_end           The IVP integration end time
    * \param[in]       pr_global       The system constant variable (pressures / densities)
    * \param[in,out]   y_global        The system state vectors at time t.
                                      Returns system state vectors at time t_end
    *
    */
    void intDriver (const Integrator& integrator, const int NUM, const double t,
                    const double t_end, const double* pr_global, double* y_global);


}

 #endif
