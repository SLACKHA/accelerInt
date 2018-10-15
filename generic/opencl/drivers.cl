#include "solver.h"
#include "error_codes.h"

// check for required defines
#ifndef neq
#pragma error "Number of equations not defined!"
#endif
#ifndef rwk_lensrc
#pragma error "Length of source-rate evaluation working buffer not defined"
#endif
#ifndef __ValueType
#pragma error "Value type not defined!"
#endif
#ifndef counter_type
#pragma error "Counter type not defined!"
#endif
#ifndef counter_type_vec
#pragma error "Vector counter type not defined!"
#endif
#ifndef solver_type
#pragma error "Solver type not defined!"
#endif
#ifndef solver_function
#pragma error "Solver function not defined!"
#endif


// \brief Indexing macro for _all_ global arrays in drivers (all state vectors are of size neq)
#define __getGlobalIndex(pid, idx) (__globalIndex1D(numProblems, neq, pid, idx))
// \brief Indexing macro for state vector driver
#define __getIndex(idx) (__getIndex1D(neq, idx))



#ifdef __EnableQueue
#warning 'Skipping driver kernel'
#else

void __kernel
__attribute__((reqd_work_group_size(__blockSize, 1, 1)))
driver (__global const double * __restrict__ param,
        const double t_start,
        __global const double * __restrict__ t_end,
        __global double * __restrict__ phi,
        __global const solver_type * __restrict__ solver,
        __global __ValueType * __restrict__ rwk,
        __global counter_type * __restrict__ counters,
        const int numProblems)
{
    // Thread-local pointers ...
    // Ordering is phi / param_working, solver working, RHS working
    // such that we can 'peel' off working data easily in subcalls
    __global __ValueType * __restrict__ my_param = rwk + __getOffset1D(neq);
    __global __ValueType *__restrict__ rwk_solver = rwk + __getOffset1D(1 + neq);
    __private counter_type_vec my_counters;
    __private __ValueType tf;

    for (int i = __ValueSize * get_global_id(0); i < numProblems; i += __ValueSize * get_global_size(0))
    {
        #if __ValueSize > 1
        for (int k = 0; k < neq; ++k)
        {
            for (int lane = 0; lane < __ValueSize; ++lane)
            {
                const int problem_id = min(i + lane, numProblems-1);
                __write_to(phi[__getGlobalIndex(problem_id, k)], lane, rwk[__getIndex(k)]);
                __write_to(param[problem_id], lane, my_param[__getIndex1D(1, 0)]);
                __write_to(t_end[problem_id], lane, tf);
            }
        }
        #else
        for (int k = 0; k < neq; ++k)
            rwk[__getIndex(k)] = phi[__getGlobalIndex(i, k)];
        my_param[__getIndex1D(1, 0)] = param[i];
        tf = t_end[i];
        #endif

        __IntType err = solver_function(
                solver, t_start, tf, 0, &my_counters, rwk, rwk_solver, my_param);

        #if __ValueSize > 1
        for (int lane = 0; lane < __ValueSize; ++lane)
        {
            const int problem_id = i + lane;
            if (problem_id < numProblems)
            {
                for (int k = 0; k < neq; ++k)
                    __read_from(rwk[__getIndex(k)], lane, phi[__getGlobalIndex(problem_id, k)]);

                __read_from(my_counters.nsteps, lane, counters[problem_id].nsteps);
                // Each lane has the same value ...
                counters[problem_id].niters = my_counters.niters;
                if (__any(err != SUCCESS))
                {
                    __read_from(err, lane, counters[problem_id].niters);
                }
            }
        }
        #else
        for (int k = 0; k < neq; ++k)
        {
            phi[__getGlobalIndex(i, k)] = rwk[__getIndex(k)];
        }
        counters[i].niters = my_counters.niters;
        counters[i].nsteps = my_counters.nsteps;
        if (err != SUCCESS)
            counters[i].niters = err;
        #endif
    }
}
#endif

#ifndef __EnableQueue
#warning 'Skipping driver_queue kernel'
#else

void __kernel
__attribute__((reqd_work_group_size(__blockSize, 1, 1)))
driver_queue (__global const double * __restrict__ param,
              const double t_start,
              __global const double * __restrict__ t_end,
              __global double * __restrict__ phi,
              __global const solver_type * __restrict__ solver,
              __global __ValueType * __restrict__ rwk,
              __global counter_type * __restrict__ counters,
              const int numProblems,
              volatile __global int *problemCounter)
{
    const int tid = get_global_id(0);

    // Thread-local pointers ...
    // Ordering is phi_woring, solver working, RHS working
    // such that we can 'peel' off working data easily in subcalls
    __global __ValueType * __restrict__ my_param = rwk + __getOffset1D(neq);
    __global __ValueType *__restrict__ rwk_solver = rwk + __getOffset1D(1 + neq);
    __private counter_type_vec my_counters;
    __private __ValueType tf;
    __private int problem_idx;

    // Initial problem set and global counter.
    #if __ValueSize > 1
        problem_idx = get_global_id(0) * __ValueSize;
    #else
        problem_idx = get_global_id(0);
    #endif

    //while ((problem_idx = atomic_inc(problemCounter)) < numProblems)
    while (problem_idx < numProblems)
    {
        #if __ValueSize > 1
        for (int k = 0; k < neq; ++k)
        {
            for (int lane = 0; lane < __ValueSize; ++lane)
            {
                const int problem_id = min(problem_idx + lane, numProblems-1);
                __write_to(phi[__getGlobalIndex(problem_id, k)], lane, rwk[__getIndex(k)]);
                __write_to(param[problem_id], lane, my_param[__getIndex1D(1, 0)]);
                __write_to(t_end[problem_id], lane, tf);
            }
        }
        #else
        for (int k = 0; k < neq; ++k)
        {
            rwk[__getIndex(k)] = phi[__getGlobalIndex(problem_idx, k)];
        }
        my_param[__getIndex1D(1, 0)] = param[problem_idx];
        tf = t_end[problem_idx];
        #endif

        // determine maximum / minumum time steps for this set of problems
        __IntType err = solver_function(
                solver, t_start, tf, 0, &my_counters, rwk, rwk_solver, my_param);

        #if __ValueSize > 1
        for (int lane = 0; lane < __ValueSize; ++lane)
        {
            const int problem_id = problem_idx + lane;
            if (problem_id < numProblems)
            {
                for (int k = 0; k < neq; ++k)
                    __read_from(rwk[__getIndex(k)], lane, phi[__getGlobalIndex(problem_id, k)]);

                __read_from(my_counters.nsteps, lane, counters[problem_id].nsteps);
                // Each lane has the same value ...
                // Each lane has the same value ...
                counters[problem_id].niters = my_counters.niters;
                if (__any(err != SUCCESS))
                {
                    __read_from(err, lane, counters[problem_id].niters);
                }
            }
        }
        #else
        for (int k = 0; k < neq; ++k)
        {
            phi[__getGlobalIndex(problem_idx, k)] = rwk[__getIndex(k)];
        }
        counters[problem_idx].niters = my_counters.niters;
        counters[problem_idx].nsteps = my_counters.nsteps;
        if (err != SUCCESS)
            counters[problem_idx].niters = err;
        #endif
        // Get a new problem atomically.
        #if __ValueSize > 1
            // add a vector's worth of problem id's
            problem_idx = atomic_add(problemCounter, __ValueSize);
        #else
            // add a single problem id
            problem_idx = atomic_inc(problemCounter);
        #endif
    }
}
#endif


#undef __getGlobalIndex
#undef __getIndex
