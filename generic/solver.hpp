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
#include <vector>
#include <memory>
#include <complex>
#include <cstring>
#include "error_codes.hpp"

extern "C"
{
    #include "jacob.h"
    #include "dydt.h"
}

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

    class SolverOptions
    {
    public:
        SolverOptions(double atol=1e-10, double rtol=1e-6, bool logging=false):
            _atol(atol),
            _rtol(rtol),
            _logging(logging)
        {

        }

        inline double atol() const
        {
            return _atol;
        }

        inline double rtol() const
        {
            return _rtol;
        }

        inline bool logging() const
        {
            return _logging;
        }

    protected:
        //! the absolute tolerance for this integrator
        const double _atol;
        //! the relative tolerance for this integrator
        const double _rtol;
        //! whether logging is enabled or not
        bool _logging;
    };

    // skeleton of the c-solver
    class Integrator
    {
    public:

        static constexpr double eps = EPS;
        static constexpr double small = SMALL;

        Integrator(int neq, int numThreads,
                   const SolverOptions& options) :
            _numThreads(numThreads),
            _neq(neq),
            _log(),
            _options(options),
            _ourMemSize(_neq * sizeof(double))
        {

        }

        ~Integrator()
        {
            this->clean();
        }

        void log(const int NUM, const double t, double const * __restrict__ phi)
        {
            // allocate new memory
            _log.emplace_back(std::move(std::unique_ptr<double[]>(new double[1 + NUM * _neq])));
            double* __restrict__ set = _log.back().get();
            // and set
            set[0] = t;
            std::memcpy(&set[1], phi, NUM * _neq * sizeof(double));
        }

        /*! \brief Copy the current log of data to the given array */
        void getLog(const int NUM, double* __restrict__ phi) const
        {
            std::size_t offset = 0;
            for (const auto& log_entry : _log)
            {
                std::memcpy(&phi[offset], log_entry.get(), (1 + NUM * _neq) * sizeof(double));
                offset += (1 + NUM * _neq);
            }
        }

        void reinitialize(int numThreads)
        {
            this->clean();
            _numThreads = numThreads;
            std::size_t _memSize = requiredSolverMemorySize();
            working_buffer = std::move(std::unique_ptr<char[]>(new char[_memSize * _numThreads]));
            std::memset(working_buffer.get(), 0, _memSize * _numThreads);
        }

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
                default:
                    return;
            }
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

        //! return the absolute tolerance
        inline const double atol() const
        {
            return _options.atol();
        }

        //! return the relative tolerance
        inline const double rtol() const
        {
            return _options.rtol();
        }

        inline bool logging() const
        {
            return _options.logging();
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

       /**
        * \brief Integration driver for the CPU integrators
        * \param[in]       NUM             The (non-padded) number of IVPs to integrate
        * \param[in]       t               The current system time
        * \param[in]       t_end           The IVP integration end time
        * \param[in]       pr_global       The system constant variable (pressures / densities)
        * \param[in,out]   y_global        The system state vectors at time t.
                                          Returns system state vectors at time t_end
        *
        */
        void intDriver (const int NUM, const double t,
                        const double t_end, const double* __restrict__ pr_global,
                        double* __restrict__ y_global);

    protected:
        //! return reference to the beginning of the working memory
        //! for this thread `tid`
        double* phi(int tid);

        //! the number of OpenMP threads to use
        int _numThreads;
        //! the number of equations to solver per-IVP
        const int _neq;
        //! working memory for this integrator
        std::unique_ptr<char[]> working_buffer;
        //! log of state vectors / times
        std::vector<std::unique_ptr<double[]>> _log;
        //! solver options
        const SolverOptions& _options;


        //! Return unique memory access
        template <typename T> T* _unique(int tid, std::size_t offset)
        {
            return (T*)(&working_buffer.get()[tid * requiredSolverMemorySize() + offset]);
        }

        /*
         * \brief Return the required memory size (per-thread) in bytes
         */
        virtual std::size_t requiredSolverMemorySize()
        {
            return _ourMemSize;
        }

    private:
        void clean()
        {
            _log.clear();
        }

        //! The required memory size of this integrator in bytes.
        std::size_t _ourMemSize;

    };

}

 #endif
