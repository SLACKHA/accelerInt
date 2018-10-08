#ifndef RKF45_HPPrkf45_types
#define RKF45_HPP

#include "error_codes.h"
#include "solver.hpp"
#include "rkf45_types.h"


namespace opencl_solvers
{
    class RKF45SolverOptions : public SolverOptions
    {
    public:
        RKF45SolverOptions(std::size_t vectorSize=1, std::size_t blockSize=1,
                           double atol=1e-10, double rtol=1e-6,
                           bool logging=false, double h_init=1e-6,
                           bool use_queue=true, std::string order="C",
                           std::string platform = "", DeviceType deviceType=DeviceType::DEFAULT,
                           size_t minIters = 1, size_t maxIters = 1000):
            SolverOptions(vectorSize, blockSize, atol, rtol,
                          logging, h_init, use_queue, order, platform, deviceType),
                _minIters(minIters),
                _maxIters(maxIters),
                _adaptionLimit(4)
        {

        }

        //! \brief The minimum number of iterations allowed
        size_t minIters() const
        {
            return _minIters;
        }

        //! \brief The maximum number of iterations allowed
        size_t maxIters() const
        {
            return _maxIters;
        }

        //! \brief The ODE oder
        size_t adaptionLimit() const
        {
            return _adaptionLimit;
        }

    protected:
        size_t _minIters;
        size_t _maxIters;
        size_t _adaptionLimit;
    };


    class RKF45Integrator : public Integrator<rk_t, rk_counters_t>
    {
private:
        rk_t rk_vals;
        std::vector<std::string> _files;
        std::vector<std::string> _paths;

public:
        RKF45Integrator(int neq, std::size_t numWorkGroups, const IVP& ivp, const RKF45SolverOptions& options) :
            Integrator(neq, numWorkGroups, ivp, options),
            rk_vals(),
            _files({//file_relative_to_me(__FILE__, "rk_types.h"),
                    file_relative_to_me(__FILE__, "rkf45.cl")}),
            _paths({path_of(__FILE__)})
        {
            // ensure our internal error code match the enum-types
            static_assert(ErrorCode::SUCCESS == RK_SUCCESS, "Enum mismatch");
            static_assert(ErrorCode::TOO_MUCH_WORK == RK_TOO_MUCH_WORK, "Enum mismatch");
            static_assert(ErrorCode::TDIST_TOO_SMALL == RK_TDIST_TOO_SMALL, "Enum mismatch");
            static_assert(ErrorCode::MAX_STEPS_EXCEEDED == RK_HIN_MAX_ITERS, "Enum mismatch");

            // init the rk struct
            rk_vals.max_iters = options.maxIters();
            rk_vals.min_iters = options.minIters();
            rk_vals.adaption_limit = options.adaptionLimit();
            rk_vals.s_rtol = options.rtol();
            rk_vals.s_atol = options.atol();
        }

protected:
        std::string solverName() const
        {
            return "rkf45";
        }

        const rk_t& getSolverStruct() const
        {
            return rk_vals;
        }

        //! \brief The requird size, in bytes of the RKF45 solver (per-IVP)
        std::size_t requiredSolverMemorySize()
        {
            // 1 for parameter, 8 working vectors
            return (1 + 8 * _neq) * sizeof(double);
        }

        //! \brief return the list of files for this solver
        const std::vector<std::string>& solverFiles() const
        {
            return _files;
        }

        //! \brief return the list of include paths for this solver
        const std::vector<std::string>& solverIncludePaths() const
        {
            return _paths;
        }


    };
}

#endif
