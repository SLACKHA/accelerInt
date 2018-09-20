/**
 * \file
 * \brief the generic main file for all GPU solvers
 *
 * \author Nicholas Curtis
 * \date 03/10/2015
 *
 * Contains skeleton of all methods that need to be defined on a per solver basis.
 */

#ifndef SOLVER_CUH
#define SOLVER_CUH

 #include "error_codes.hpp"

namespace cuda_solvers {

     __global__
    void intDriver (const int NUM,
                    const double t,
                    const double t_end,
                    const double * __restrict__ pr_global,
                    double * __restrict__ y_global,
                    const mechanism_memory * __restrict__ d_mem,
                    const solver_memory * __restrict__ s_mem);

    __device__ void integrate (const double,
                               const double,
                               const double,
                               double * const __restrict__,
                               mechanism_memory const * const __restrict__,
                               solver_memory const * const __restrict__);

    // something for solvers to inherit from
    __host__ __device__
    class SolverProperties
    {

    };

    // something for solvers to inherit from
    __host__ __device__
    class SolverMemory
    {

    };


    // skeleton of the cuda-solver
    __host__
    class Integrator
    {
        __host__
        virtual void Integrator(const int, solver_memory**, solver_memory**) = 0;
        __host__
        virtual void ~Integrator(solver_memory**, solver_memory**) = 0;
        __host__
        virtual size_t requiredSolverMemorySize() = 0;
        __host__
        virtual void initSolverLog() = 0;
        __host__
        virtual void solverLog() = 0;
        __host__
        virtual void init(const int, solver_memory**, solver_memory**) = 0;
        __host__
        virtual void cleanup(solver_memory**, solver_memory**) = 0;
        __host__
        virtual const char* solverName() = 0;
        __host__
        virtual void checkError(const int, const ErrorCode*) = 0;
    };

}

#endif
