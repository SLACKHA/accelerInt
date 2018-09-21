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

#include <cfloat>
#include <iostream>
#include <memory>
#include <complex>
#include "error_codes.hpp"

//include OpenMP if available
#ifdef _OPENMP
 #include <omp.h>
#else
 #warning 'OpenMP not found, only single-threaded execution will be available for C-solvers.'
 #define omp_get_max_threads() (1)
 #define omp_get_num_threads() (1)
 #define omp_get_thread_num() (0)
#endif


namespace c_solvers {

    /*! Machine precision constant. */
    #define EPS DBL_EPSILON
    /*! Smallest representable double */
    #define SMALL DBL_MIN

    //! Options for this solver
    class SolverOptions
    {
        public:
            //! relative tolerance
            const double rtol;
            //! absolute tolerance
            const double atol;

            SolverOptions(double atol=1e-10, double rtol=1e-6):
                atol(atol),
                rtol(rtol)
            {

            }

            static constexpr double eps = EPS;
            static constexpr double small = SMALL;
    };

    // skeleton of the c-solver
    class Integrator
    {
        public:
            Integrator(int neq, int numThreads, double atol=1e-10, double rtol=1e-6) :
                _numThreads(numThreads),
                _neq(neq),
                options(atol, atol),
                memSize(requiredSolverMemorySize())
            {
                working_buffer = std::unique_ptr<char>(new char[memSize * _numThreads]);
            }

            ~Integrator()
            {

            }

            virtual void initSolverLog() = 0;
            virtual void solverLog() = 0;

            virtual void reinitialize(int numThreads) = 0;
            virtual void clean() = 0;

            virtual const char* solverName() const = 0;

            /*! checkError
                \brief Checks the return code of the given thread (IVP) for an error, and exits if found
                \param tid The thread (IVP) index
                \param code The return code of the thread
                @see ErrorCodes
            */
            void checkError(int tid, ErrorCode code) const
            {
                switch(code)
                {
                    case ErrorCode::MAX_CONSECUTIVE_ERRORS_EXCEEDED:
                        std::cerr << "During integration of ODE#" << tid <<
                            ", an error occured on too many consecutive integration steps,"
                            "exiting..." << std::endl;
                        exit(code);
                    case ErrorCode::MAX_STEPS_EXCEEDED:
                        std::cerr << "During integration of ODE#" << tid <<
                            "the allowed number of integration steps was exceeded,"
                            "exiting..." << std::endl;
                        exit(code);
                    case ErrorCode::H_PLUS_T_EQUALS_H:
                        std::cerr << "During integration of ODE#" << tid <<
                        "the stepsize 'h' was decreased such that h = t + h,"
                        "exiting..." << std::endl;
                        exit(code);
                    case ErrorCode::MAX_NEWTON_ITER_EXCEEDED:
                        std::cerr << "During integration of ODE#" << tid <<
                        "the allowed number of newton iteration steps was exceeded,"
                        "exiting..." << std::endl;
                        exit(code);
                }
            }

            /*
             * \brief Return the required memory size (per-thread) in bytes
             */
            std::size_t requiredSolverMemorySize()
            {
                // phi local
                return _neq * sizeof(double);
            }

            /**
            * \brief A header definition of the integrate method, that must be implemented by various solvers
            * \param[in]          t_start             the starting IVP integration time
            * \param[in]          t_end               the IVP integration endtime
            * \param[in]          pr                  the IVP constant variable (presssure/density)
            * \param[in,out]      y                   The IVP state vector at time t_start.
                                                      At end of this function call, the system state at time t_end is stored here
            */
            virtual ErrorCode integrate(const double t_start, const double t_end,
                                        const double pr, double* y) = 0;

            //! return reference to the beginning of the working memory
            //! for this thread `tid`
            virtual double* phi(int tid) final;

            //! return the absolute tolerance
            inline const double atol() const
            {
                return options.atol;
            }

            //! return the relative tolerance
            inline const double rtol() const
            {
                return options.rtol;
            }

            //! return the number of equations to solve
            inline const int neq() const
            {
                return _neq;
            }

            //! return the number of equations to solve
            inline const int numThreads() const
            {
                return _numThreads;
            }

        protected:
            //! the required memory size (in bytes) for this solver
            const std::size_t memSize;

            //! the number of OpenMP threads to use
            const int _numThreads;

            //! the number of equations to solver per-IVP
            const int _neq;

            //! the options for this solver
            SolverOptions options;

            //! working memory for this integrator
            std::unique_ptr<char> working_buffer;

            //! Return unique memory access
            template <typename T> T* _unique(int tid, std::size_t offset)
            {
                return (T*)(&working_buffer.get()[tid * memSize + offset]);
            }
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
    void intDriver (Integrator& integrator, const int NUM, const double t,
                    const double t_end, const double* pr_global, double* y_global);


}

 #endif
