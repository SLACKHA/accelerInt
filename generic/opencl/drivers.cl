#include "solver.h"
#include "error_codes.h"
#include "dydt.h"

// check for required defines
#ifndef neq
#pragma error "Number of equations not defined!"
#endif
#ifndef rwk_lensrc
#pragma error "Length of source-rate evaluation working buffer not defined"
#endif
#ifndef rwk_lensol
#pragma error "Length of solution working buffer not defined."
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


__ValueType get_wnorm (__global const solver_type* __restrict__ solver, __global const __ValueType* __restrict__ x,
                       __global const __ValueType* __restrict__ y)
{
    __ValueType sum = 0;
    for (int k = 0; k < neq; k++)
    {
        __ValueType ewt = (solver->s_rtol * fabs(y[__getIndex(k)])) + solver->s_atol;
        __ValueType prod = x[__getIndex(k)] / ewt;
        sum += (prod*prod);
    }

    return sqrt(sum / (__ValueType)neq);
}


__IntType get_hin (__global const solver_type *solver, const __ValueType t, const __ValueType t_end,
                   __global __ValueType* __restrict__ h0, __global __ValueType* __restrict__ y,
                   __global __ValueType * rwk,
                   __global __ValueType const * __restrict__ user_data,
                   const int offset)
{
    #define t_round ((t_end - t) * DBL_EPSILON)
    #define h_min (t_round * 100)
    #define h_max ((t_end - t) / solver->min_iters)

    if (__any((t_end - t) < 2 * t_round))
    {
        // requested time-step is smaller than roundoff
        return OCL_TDIST_TOO_SMALL;
    }

    __global __ValueType * __restrict__ ydot  = rwk + __getOffset1D(offset);
    __global __ValueType * __restrict__ y1    = rwk + __getOffset1D(offset + neq);
    __global __ValueType * __restrict__ ydot1 = rwk + __getOffset1D(offset + 2*neq);
    // the portion of the rwk vector that's allocated for the source rate evaluation
    __global __ValueType* rwk_dydt = rwk + __getOffset1D(rwk_lensol);

    __ValueType hlb = h_min;
    __ValueType hub = h_max;
    //double hlb = h_min;
    //double hub = h_max;

    // Already done ...
    __MaskType done = isgreaterequal(*h0, h_min);
    __ValueType hg = sqrt(hlb*hub);

    if (__all(hub < hlb))
    {
        *h0 = __select(hg, *h0, done);
        return OCL_SUCCESS;
    }

    // Start iteration to find solution to ... {WRMS norm of (h0^2 y'' / 2)} = 1

    __MaskType hnew_is_ok = 0;
    __ValueType hnew = hg;
    int iter = 0;
    int ierr = OCL_SUCCESS;

    // compute ydot at t=t0
    dydt(t, user_data, y, ydot, rwk_dydt);

    // maximum of 2 iterations
    #define MAX_HINIT_ITERS (1)
    for(; iter <= MAX_HINIT_ITERS; ++iter)
    {
        // Estimate y'' with finite-difference ...
        //double t1 = hg;
        for (int k = 0; k < neq; k++)
        {
            y1[__getIndex(k)] = y[__getIndex(k)] + hg * ydot[__getIndex(k)];
        }

        // compute y' at t1
        dydt(t, user_data, y1, ydot1, rwk_dydt);

        // Compute WRMS norm of y''
        for (int k = 0; k < neq; k++)
            y1[__getIndex(k)] = (ydot1[__getIndex(k)] - ydot[__getIndex(k)]) / hg;

        __ValueType yddnrm = get_wnorm(solver, y1, y);

        // should we accept this?
        hnew = __select(hnew, hg, hnew_is_ok | (iter == MAX_HINIT_ITERS));
        if (__all(hnew_is_ok) || (iter == MAX_HINIT_ITERS))
            break;

        // Get the new value of h ...
        __MaskType test = isgreater(yddnrm*hub*hub, 2.0);
        hnew = __select(sqrt(hg * hub), sqrt(2.0 / yddnrm), test);
        // test the stopping conditions.
        __ValueType hrat = hnew / hg;

        // Accept this value ... the bias factor should bring it within range.
        hnew_is_ok = isgreater(hrat, 0.5) & isless(hrat, 2.0);

        // If y'' is still bad after a few iterations, just accept h and give up.
        if (iter >= MAX_HINIT_ITERS)
        {
            hnew_is_ok = isgreater(hrat, 2.0);
            hnew = __select (hnew, hg, hnew_is_ok);
        }

        hg = hnew;
    }

    // bound and bias estimate
    *h0 = hnew * 0.5;
    *h0 = fmax(*h0, hlb);
    *h0 = fmin(*h0, hub);

    #undef t_round
    #undef h_min
    #undef h_max

    return ierr;
}


// the size of working memory required for the drivers
#define driver_offset (2 + neq)

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
        __global __IntType* __restrict__ iwk,
        __global counter_type * __restrict__ counters,
        const int numProblems
#ifdef __EstimateChemistryTime
        , __global double* __restrict__ last_timestep
#endif
        )
{
    // Thread-local pointers ...
    // Ordering is phi / param_working, solver working, RHS working
    // such that we can 'peel' off working data easily in subcalls
    __global __ValueType * __restrict__ my_param = rwk + __getOffset1D(neq);
    __global __ValueType * __restrict__ h = rwk + __getOffset1D(1 + neq);
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
                solver, t_start, tf, h, &my_counters, rwk, rwk, iwk, my_param,
                driver_offset);

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
                if (__any(err != OCL_SUCCESS))
                {
                    __read_from(err, lane, counters[problem_id].niters);
                }
                #ifdef __EstimateChemistryTime
                __read_from(h[__getIndex1D(1, 0)], lane, last_timestep[problem_id]);
                #endif
            }
        }
        #else
        for (int k = 0; k < neq; ++k)
        {
            phi[__getGlobalIndex(i, k)] = rwk[__getIndex(k)];
        }
        counters[i].niters = my_counters.niters;
        counters[i].nsteps = my_counters.nsteps;
        if (err != OCL_SUCCESS)
            counters[i].niters = err;
        #endif
        #ifdef __EstimateChemistryTime
        last_timestep[i] = h;
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
              __global __IntType* __restrict__ iwk,
              __global counter_type * __restrict__ counters,
              const int numProblems,
              volatile __global int *problemCounter
#ifdef __EstimateChemistryTime
              , __global double* __restrict__ last_timestep
#endif
)
{
    // Thread-local pointers ...
    // Ordering is phi_woring, solver working, RHS working
    // such that we can 'peel' off working data easily in subcalls
    __global __ValueType * __restrict__ my_param = rwk + __getOffset1D(neq);
    __global __ValueType * __restrict__ h = rwk + __getOffset1D(1 + neq);
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
                solver, t_start, tf, h, &my_counters, rwk, rwk, iwk, my_param,
                driver_offset);

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
                if (__any(err != OCL_SUCCESS))
                {
                    __read_from(err, lane, counters[problem_id].niters);
                }
                #ifdef __EstimateChemistryTime
                __read_from(h[__getIndex1D(1, 0)], lane, last_timestep[problem_id]);
                #endif
            }
        }
        #else
        for (int k = 0; k < neq; ++k)
        {
            phi[__getGlobalIndex(problem_idx, k)] = rwk[__getIndex(k)];
        }
        counters[problem_idx].niters = my_counters.niters;
        counters[problem_idx].nsteps = my_counters.nsteps;
        if (err != OCL_SUCCESS)
            counters[problem_idx].niters = err;
        #ifdef __EstimateChemistryTime
        last_timestep[problem_idx] = h;
        #endif
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

__ValueType update_timestep(__private __ValueType const t_start,
                            __private __ValueType const t_end,
                            __private __ValueType const t,
                            __global __ValueType * __restrict__ hcur,
                            __private __ValueType h_old,
                            __private __ValueType const herr,
                            __private __MaskType const accept,
                            __private __MaskType const done,
                            __private const int niters,
                            __private const int ELO,
                            __private const int adaption_limit,
                            __private const int min_iters)
{
    // the update scheme here is:
    // #ifndef __EstimateChemistryTime
    //      -> estimate at usual
    // #ifdef __EstimateChemistryTime
    //      -> if done
    //          -> if iters > 0
    //              use last time-step size
    //          -> else, use unrestricted estimated time-step size
    //      -> estimate as usual


    #define t_round ((t_end - t_start) * DBL_EPSILON)
    #define h_min (t_round * 100)
    #define h_max ((t_end - t_start) / min_iters)
    #define h (*(hcur))

    __private __ValueType h_old2 = h;

    #ifndef CONSTANT_TIMESTEP
    __ValueType fact = 0.9 * pow( 1.0 / herr, (1.0 / ELO));

    // Restrict the rate of change in dt
    fact = fmax(fact, 1.0 / adaption_limit);
    fact = fmin(fact, (double)adaption_limit);

    // Apply grow/shrink factor for next step.
    h *= fact;

    h_old2 = h;

    // Limit based on the upper/lower bounds
    h = fmin(h, h_max);
    h = fmax(h, h_min);
    #endif

    #if defined(__EstimateChemistryTime) && !defined(CONSTANT_TIMESTEP)

    // use last time-step size
    __MaskType our_done = (t + h - t_end) * (t + h - t_start) >= 0;
    h_old = __select(h, h_old, (our_done) & (niters > 0));

    // use unrestricted estimated time-step size
    h_old = __select(h, h_old2, (our_done) & __not(niters));

    // Stretch the final step if we're really close and we didn't just fail ...
    h = __select(h, t_end - t, accept & isless(fabs((t + h) - t_end), h_min));

    // Don't overshoot the final time ...
    h = __select(h, t_end - t, __not(done) & isgreater((t + h),  t_end));

    #elif !defined(CONSTANT_TIMESTEP)

    // Stretch the final step if we're really close and we didn't just fail ...
    h = __select(h, t_end - t, accept & isless(fabs((t + h) - t_end), h_min));

    // Don't overshoot the final time ...
    h = __select(h, t_end - t, __not(done) & isgreater((t + h),  t_end));

    #endif

    #undef t_round
    #undef h_min
    #undef h_max
    #undef h

    return h_old;
}


#undef driver_offset
#undef __getGlobalIndex
#undef __getIndex
