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

    //! \brief Implementation of a initial value problem
    class IVP
    {
    public:
        IVP(std::size_t requiredMemorySize,
            bool pass_entire_working=false):
            _requiredMemorySize(requiredMemorySize),
            _pass_entire_working(pass_entire_working)
        {

        }

        //! \brief Return the required amount of double-precision working memory
        //         required by the IVP functions [in bytes]
        std::size_t requiredMemorySize() const
        {
            return _requiredMemorySize;
        }

        //! \brief If true, pass the entire working buffer allocated for the IVP to
        //         the source term and jacobian evaluation functions.  Note that this
        //         requires the IVP functions to implement the correct indexing of
        //         the working buffer.
        bool passEntireWorkingBufferToIVP() const
        {
            return _pass_entire_working;
        }


    protected:
        std::size_t _requiredMemorySize;
        bool _pass_entire_working;

    };

    class SolverOptions
    {
    public:
        SolverOptions(double atol=1e-10, double rtol=1e-6, bool logging=false,
                      size_t minIters = 1, size_t maxIters = 1000):
            _atol(atol),
            _rtol(rtol),
            _logging(logging),
            _minIters(minIters),
            _maxIters(maxIters)
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

        inline std::size_t minIters() const
        {
            return _minIters;
        }

        inline std::size_t maxIters() const
        {
            return _maxIters;
        }

    protected:
        //! the absolute tolerance for this integrator
        const double _atol;
        //! the relative tolerance for this integrator
        const double _rtol;
        //! whether logging is enabled or not
        bool _logging;
        //! The minimum allowed internal integration steps
        const size_t _minIters;
        //! The maxiumum allowed internal integration steps
        const size_t _maxIters;
    };

    // skeleton of the c-solver
    class Integrator
    {
    public:

        static constexpr double eps = EPS;
        static constexpr double small = SMALL;

        Integrator(int neq, int numThreads,
                   const IVP& ivp,
                   const SolverOptions& options) :
            _numThreads(numThreads),
            _neq(neq),
            _log(),
            _ivp(ivp),
            _options(options),
            _ourMemSize(setOffsets())
        {
            setOffsets();
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
        void getLog(const int NUM, double* __restrict__ times, double* __restrict__ phi) const
        {
            for (std::size_t index = 0; index < _log.size(); ++index)
            {
                const double* __restrict__ log_entry = _log[index].get();
                times[index] = log_entry[0];
                std::memcpy(&phi[index * NUM * _neq], &log_entry[1], NUM * _neq * sizeof(double));
            }
        }

        /*! \brief Return the number of integration steps completed by this Integrator */
        std::size_t numSteps() const
        {
            return _log.size();
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
                        ", the allowed number of integration steps was exceeded,"
                        "exiting..." << std::endl;
                    exit(code);
                case ErrorCode::H_PLUS_T_EQUALS_H:
                    std::cerr << "During integration of ODE#" << tid <<
                    ", the stepsize 'h' was decreased such that h = t + h,"
                    "exiting..." << std::endl;
                    exit(code);
                case ErrorCode::MAX_NEWTON_ITER_EXCEEDED:
                    std::cerr << "During integration of ODE#" << tid <<
                    ", the allowed number of newton iteration steps was exceeded,"
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

        inline std::size_t minIters() const
        {
            return _options.minIters();
        }

        inline std::size_t maxIters() const
        {
            return _options.maxIters();
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
        //! return reference to the working memory allocated for the jacobian and source
        //! rates
        double* rwk(int tid);

        //! the number of OpenMP threads to use
        int _numThreads;
        //! the number of equations to solver per-IVP
        const int _neq;
        //! working memory for this integrator
        std::unique_ptr<char[]> working_buffer;
        //! log of state vectors / times
        std::vector<std::unique_ptr<double[]>> _log;
        //! IVP implementation
        const IVP& _ivp;
        //! solver options
        const SolverOptions& _options;

        //! \brief the offset to the memory allocated for the IVP in the working buffer
        std::size_t _ivp_working;
        //! \brief the offset to the memory allocated for a local copy of the state vector
        std::size_t _phi;
        //! \brief The offset to the working memory dy-vector for the initial timestep estimation
        std::size_t _ydot_hin;
        //! \brief The offset to the working memory y-perturbation vector for the initial timestep estimation
        std::size_t _y1_hin;
        //! \brief The offset to the working memory dy-vector for the initial timestep estimation
        std::size_t _ydot1_hin;


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

        /*
         * \brief Estimate the inital time-step
         *
         * \param[in]       t           The starting integration time
         * \param[in]       t_end       the end integration time
         * \param[in]       y           The initial state vector
         * \param[in]       user_data   The user parameter
         * \param[in,out]   h0          The calculated initial time-step
         */
        ErrorCode hinit(const double t, const double t_end,
                        const double* __restrict__ y,
                        const double user_data,
                        double* __restrict__ h0);

        /* \brief Return the weighted error norm (WRMS) of the given inputs
         *
         * \param[in]       x           The potential new vector
         * \param[in]       y           The reference vector
         */
        inline double get_wnorm(const double* __restrict__ x, const double* __restrict__ y)
        {
            double sum = 0;
            for (int k = 0; k < _neq; k++)
            {
                double prod = x[k] / (rtol() * std::fabs(y[k]) + atol());
                sum += (prod*prod);
            }
            return std::sqrt(sum / (double)_neq);
        }

    private:
        void clean()
        {
            _log.clear();
        }

        //! The required memory size of this integrator in bytes.
        std::size_t _ourMemSize;

        /**
         * \brief Determines offsets for memory access from #working_buffer in
         *        the integrator
         */
        std::size_t setOffsets()
        {
            std::size_t offset = 0;
            _ivp_working = offset;
            offset += _ivp.requiredMemorySize();
            _phi = offset;
            offset += _neq * sizeof(double);
            _ydot_hin = offset;
            offset += _neq * sizeof(double);
            _y1_hin = offset;
            offset += _neq * sizeof(double);
            _ydot1_hin = offset;
            offset += _neq * sizeof(double);
            return offset;
        }

    };

}

 #endif
