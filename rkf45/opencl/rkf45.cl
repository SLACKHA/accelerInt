
// include error code definitions
#include "error_codes.h"
// and macros / solver definitions
#include "solver.h"
// include the source rate evaluation header
#include "dydt.h"

// check for required defines
#ifndef neq
#pragma error "Number of equations not defined!"
#endif
#ifndef rk_lensrc
#pragma error "Length of source-rate evaluation working buffer not defined"
#endif
// \brief required work-space for RKF45 internals
#define lenrwk_rk (8 * neq)

// \brief Indexing macro for _all_ arrays in RKF45 solver (all arrays are size neq)
#define __getIndex(idx) (__getIndex1D(neq, idx))


// Single-step function
int FUNC_TYPE(rkf45) (const __ValueType h, const __ValueType t,
                      __global const __ValueType* __restrict__ y,
                      __global __ValueType* __restrict__ y_out,
                      __global __ValueType* __restrict__ rwk,
                      __global __ValueType* __restrict__ user_data)
{
    #define c20 ( 0.25)
    #define c21 ( 0.25)
    #define c30 ( 0.375)
    #define c31 ( 0.09375)
    #define c32 ( 0.28125)
    #define c40 ( 0.92307692307692)
    #define c41 ( 0.87938097405553)
    #define c42 (-3.2771961766045)
    #define c43 ( 3.3208921256258)
    #define c51 ( 2.0324074074074)
    #define c52 (-8.0)
    #define c53 ( 7.1734892787524)
    #define c54 (-0.20589668615984)
    #define c60 ( 0.5)
    #define c61 (-0.2962962962963)
    #define c62 ( 2.0)
    #define c63 (-1.3816764132554)
    #define c64 ( 0.45297270955166)
    #define c65 (-0.275)
    #define a1 ( 0.11574074074074)
    #define a2 ( 0.0)
    #define a3 ( 0.54892787524366)
    #define a4 ( 0.5353313840156)
    #define a5 (-0.2)
    #define b1 ( 0.11851851851852)
    #define b2 ( 0.0)
    #define b3 ( 0.51898635477583)
    #define b4 ( 0.50613149034201)
    #define b5 (-0.18)
    #define b6 ( 0.036363636363636)

    #define h1 (0)
    #define h2 (0.25)
    #define h3 (0.375)
    #define h4 (0.92307692307692307) // 12 / 13
    #define h5 (1.0)
    #define h6 (0.5)
    #define
    /*   const double c20 = 0.25,
                            c21 = 0.25,
                            c30 = 0.375,
                            c31 = 0.09375,
                            c32 = 0.28125,
                            c40 = 0.92307692307692,
                            c41 = 0.87938097405553,
                            c42 =-3.2771961766045,
                            c43 = 3.3208921256258,
                            c51 = 2.0324074074074,
                            c52 =-8.0,
                            c53 = 7.1734892787524,
                            c54 =-0.20589668615984,
                            c60 = 0.5,
                            c61 =-0.2962962962963,
                            c62 = 2.0,
                            c63 =-1.3816764132554,
                            c64 = 0.45297270955166,
                            c65 =-0.275,
                            a1 = 0.11574074074074,
                            a2 = 0.0,
                            a3 = 0.54892787524366,
                            a4 = 0.5353313840156,
                            a5 =-0.2,
                            b1 = 0.11851851851852,
                            b2 = 0.0,
                            b3 = 0.51898635477583,
                            b4 = 0.50613149034201,
                            b5 =-0.18,
                            b6 = 0.036363636363636;*/

    // local dependent variables (5 total)
    __global __ValueType* f1   = rwk ;
    __global __ValueType* f2   = rwk + __getOffset1D(neq);
    __global __ValueType* f3   = rwk + __getOffset1D(2*neq);
    __global __ValueType* f4   = rwk + __getOffset1D(3*neq);
    __global __ValueType* f5   = rwk + __getOffset1D(4*neq);
    __global __ValueType* f6   = rwk + __getOffset1D(5*neq);
    __global __ValueType* ytmp = rwk + __getOffset1D(6*neq);
    // the portion of the rwk vector that's allocated for the source rate evaluation
    // y_out is at 7 * neq, hence we go to 8 for the total offset
    __global __ValueType* rwk_dydt = rwk + __getOffset1D(8*neq);

    // 1)
    dydt(t, user_data, y, f1, rwk_dydt);

    for (int k = 0; k < neq; k++)
    {
        //f1[k] = h * ydot[k];
        f1[k] *= h;
        ytmp[__getIndex(k)] = y[__getIndex(k)] + c21 * f1[__getIndex(k)];
    }

    // 2)
    dydt(t + h2 * h, user_data, ytmp, f2, rwk_dydt);

    for (int k = 0; k < neq; k++)
    {
        //f2[k] = h * ydot[k];
        f2[__getIndex(k)] *= h;
        ytmp[__getIndex(k)] = y[__getIndex(k)] + c31 * f1[__getIndex(k)] + c32 * f2[__getIndex(k)];
    }

    // 3)
    dydt(t + h3 * h, user_data, ytmp, f3, rwk_dydt);

    for (int k = 0; k < neq; k++) {
        //f3[k] = h * ydot[k];
        f3[__getIndex(k)] *= h;
        ytmp[__getIndex(k)] = y[__getIndex(k)] + c41 * f1[__getIndex(k)] + c42 * f2[__getIndex(k)] + c43 * f3[__getIndex(k)];
    }

    // 4)
    dydt(t + h4 * h, user_data, ytmp, f4, rwk_dydt);

    for (int k = 0; k < neq; k++) {
        //f4[k] = h * ydot[k];
        f4[__getIndex(k)] *= h;
        ytmp[__getIndex(k)] = y[__getIndex(k)] + c51 * f1[__getIndex(k)] + c52 * f2[__getIndex(k)] + c53 * f3[__getIndex(k)] + c54 * f4[__getIndex(k)];
    }

    // 5)
    dydt(t + h5 * h, user_data, ytmp, f5);

    for (int k = 0; k < neq; k++) {
        //f5[k] = h * ydot[k];
        f5[__getIndex(k)] *= h;
        ytmp[__getIndex(k)] = y[__getIndex(k)] + c61*f1[__getIndex(k)] + c62*f2[__getIndex(k)] + c63*f3[__getIndex(k)] + c64*f4[__getIndex(k)] + c65*f5[__getIndex(k)];
    }

    // 6)
    dydt(t + h6 * h, user_data, ytmp, f6, rwk_dydt);

    for (int k = 0; k < neq; k++)
    {
        //const T f6 = h * ydot[k];
        f6[__getIndex(k)] *= h;

        // 5th-order RK value.
        const __ValueType r5 = b1*f1[__getIndex(k)] + b3*f3[__getIndex(k)] + b4*f4[__getIndex(k)] + b5*f5[__getIndex(k)] + b6*f6[__getIndex(k)];

        // 4th-order RK residual.
        const __ValueType r4 = a1*f1[__getIndex(k)] + a3*f3[__getIndex(k)] + a4*f4[__getIndex(k)] + a5*f5[__getIndex(k)];

        // Trucation error: difference between 4th and 5th-order RK values.
        rwk[__getIndex(k)] = fabs(r5 - r4);

        // Update solution.
        y_out[__getIndex(k)] = y[__getIndex(k)] + r5; // Local extrapolation
    }

    return RK_SUCCESS;

    #undef c20
    #undef c21
    #undef c30
    #undef c31
    #undef c32
    #undef c40
    #undef c41
    #undef c42
    #undef c43
    #undef c51
    #undef c52
    #undef c53
    #undef c54
    #undef c60
    #undef c61
    #undef c62
    #undef c63
    #undef c64
    #undef c65
    #undef a1
    #undef a2
    #undef a3
    #undef a4
    #undef a5
    #undef b1
    #undef b2
    #undef b3
    #undef b4
    #undef b5
    #undef b6
}

__ValueType FUNC_TYPE(rk_wnorm) (__global const rk_t *rk, __global const __ValueType *x, __global const __ValueType *y)
{
    __ValueType sum = 0;
    for (int k = 0; k < neq; k++)
    {
        __ValueType ewt = (rk->s_rtol * fabs(y[__getIndex(k)])) + rk->s_atol;
        __ValueType prod = x[__getIndex(k)] / ewt;
        sum += (prod*prod);
    }

    return sqrt(sum / (__ValueType)neq);
}

int FUNC_TYPE(rk_hin) (__global const rk_t *rk, const __ValueType t, const __ValueType t_end,
                                             __ValueType* __restrict__ h0, __global __ValueType* __restrict__ y,
                                             __global __ValueType * __restrict__ rwk,
                                             __global __ValueType const * __restrict__ user_data)
{
    #define t_round ((t_end - t) * DBL_EPSILON)
    #define h_min (t_round * 100)
    #define h_max ((tf - t) / rk->min_iters)

    __global __ValueType * __restrict__ ydot  = rwk;
    __global __ValueType * __restrict__ y1    = ydot + __getOffset1D(neq);
    __global __ValueType * __restrict__ ydot1 = y1 + __getOffset1D(neq);
    // portion of the rwk vector used by source rate evaluation
    __global __ValueType * __restrict__ rwk_dydt = rwk + __getOffset1D(8 * neq);

    double hlb = h_min;
    double hub = h_max;
    //double hlb = h_min;
    //double hub = h_max;

    // Alread done ...
    __MaskType done = isgreaterequal(*h0, h_min);

    __ValueType hg = sqrt(hlb*hub);

    if (hub < hlb)
    {
        *h0 = __select(hg, *h0, done);

        return RK_SUCCESS;
    }

    // Start iteration to find solution to ... {WRMS norm of (h0^2 y'' / 2)} = 1

    __MaskType hnew_is_ok = 0;
    __ValueType hnew = hg;
    const int miters = 10;
    int iter = 0;
    int ierr = RK_SUCCESS;

    // compute ydot at t=t0
    dydt(t, user_data, y, ydot, rwk_dydt);

    while(1)
    {
        // Estimate y'' with finite-difference ...
        //double t1 = hg;

        #ifdef __INTEL_COMPILER
        #pragma ivdep
        #endif
        for (int k = 0; k < neq; k++)
             y1[__getIndex(k)] = y[__getIndex(k)] + hg * ydot[__getIndex(k)];

        // compute y' at t1
        dydt(t, user_data, y1, ydot1, rwk_dydt);

        // Compute WRMS norm of y''
        #ifdef __INTEL_COMPILER
        #pragma ivdep
        #endif
        for (int k = 0; k < neq; k++)
            y1[__getIndex(k)] = (ydot1[__getIndex(k)] - ydot[__getIndex(k)]) / hg;

        __ValueType yddnrm = FUNC_TYPE(rk_wnorm) (rk, y1, y);

        // should we accept this?
        hnew = __select(hnew, hg, hnew_is_ok | (iter == miters));
        if (__all(hnew_is_ok) || (iter == miters))
            break;

        // Get the new value of h ...
        //hnew = (yddnrm*hub*hub > 2.0) ? sqrt(2.0 / yddnrm) : sqrt(hg * hub);
        {
            __MaskType test = isgreater( yddnrm*hub*hub, 2.0);
            hnew = __select ( sqrt(hg * hub), sqrt(2.0 / yddnrm), test);
        }

        // test the stopping conditions.
        __ValueType hrat = hnew / hg;

        // Accept this value ... the bias factor should bring it within range.
        hnew_is_ok = isgreater(hrat, 0.5) & isless(hrat, 2.0);

        // If y'' is still bad after a few iterations, just accept h and give up.
        if (iter > 1)
        {
            hnew_is_ok = isgreater(hrat, 2.0);
            hnew = __select (hnew, hg, hnew_is_ok);
        }

        hg = hnew;
        iter ++;
    }

    // bound and bias estimate
    *h0 = hnew * 0.5;
    *h0 = fmax(*h0, hlb);
    *h0 = fmin(*h0, hub);

    //printf("h0=%e, hlb=%e, hub=%e\n", h0, hlb, hub);

    #undef t_round
    #undef h_min
    #undef h_max

    return ierr;
}

//! \brief struct containing information on steps / iterations
typedef struct
{
    int niters;
    __MaskType nsteps;
}
FUNC_SIZE(rk_counters_t);


__IntType FUNC_TYPE(rk_solve) (
                             __global const rk_t * __restrict__ rk,
                             __global __ValueType * const __restrict__ t_start,
                             __global __ValueType * const __restrict__ t_end,
                             __private __ValueType * const __restrict__ hcur,
                             __private FUNC_SIZE(rk_counters_t) * __restrict__ counters,
                             __global __ValueType* __restrict__ y,
                             __global __ValueType* __restrict__ rwk,
                             __global __ValueType* __restrict__ user_data)
{
     int ierr = RK_SUCCESS;
     // Estimate the initial step size ...
     {
        __MaskType test = isless(*hcur, rk->h_min);
        if (__any(test))
        {
            ierr = FUNC_TYPE(rk_hin) (rk, *tcur, *t_end, hcur, y, rwk, user_data);
        }
     }

     #define t (*t_start)
     #define tf (*t_end)
     #define t_round ((tf - t) * DBL_EPSILON)
     #define h (*hcur)
     #define h_min (t_round * 100)
     #define h_max ((tf - t) / rk->min_iters)
     #define iter (counters->niters)
     #define nst (counters->nsteps)

     nst = 0;
     iter = 0;

     __MaskType done = isless(fabs(t - tf), fabs(t_round));

    while (__any(__not(done)))
    {
        __global __ValueType *ytmp = rwk + __getOffset1D(neq*7);

        // Take a trial step over h_cur ...
        FUNC_TYPE(rkf45) (neq, h, t, y, ytmp, rwk, user_data);

        __ValueType herr = fmax(1.0e-20, FUNC_TYPE(rk_wnorm) (rk, rwk, y));

        // Is there error acceptable?
        __MaskType accept = islessequal(herr, 1.0);
        accept |= islessequal(h, h_min);
        accept &= __not(done);

        // update solution ...
        if (__any(accept))
        {
            t   = __select (t,   t + h  , accept);
            nst = __select (nst, nst + 1, accept);

            for (int k = 0; k < neq; k++)
                y[__getIndex(k)] = __select(y[__getIndex(k)], ytmp[__getIndex(k)], accept);

            done = isless( fabs(t - tf), fabs(t_round));
        }

        __ValueType fact = sqrt( sqrt(1.0 / herr) ) * (0.840896415);

        // Restrict the rate of change in dt
        fact = fmax(fact, 1.0 / rk->adaption_limit);
        fact = fmin(fact,       rk->adaption_limit);

        // Apply grow/shrink factor for next step.
        h = __select(h * fact, h, done);

        // Limit based on the upper/lower bounds
        h = fmin(h, h_max);
        h = fmax(h, h_min);

        // Stretch the final step if we're really close and we didn't just fail ...
        h = __select(h, tf - t, accept & isless(fabs((t + h) - tf), h_min));

        // Don't overshoot the final time ...
        h = __select(h, tf - t, __not(done) & isgreater((t + h), tf));

        ++iter;
        if (rk->max_iters && iter > rk->max_iters) {
            ierr = RK_TOO_MUCH_WORK;
            //printf("(iter > max_iters)\n");
            break;
        }
    }

    return ierr;

    #undef t
    #undef tf
    #undef t_round
    #undef h
    #undef h_min
    #undef h_max
    #undef iter
    #undef nst
}

#ifdef __EnableQueue
#warning 'Skipping rkf45_driver kernel'
#else

void __kernel
__attribute__((reqd_work_group_size(__blockSize, 1, 1)))
rkf45_driver (__global const double * __restrict__ param,
              __global const double * __restrict__ t_start,
              __global const double * __restrict__ t_end,
              __global double * __restrict__ phi,
              __global const rk_t * __restrict__ rk,
              __global double * __restrict__ rwk,
              __global rk_counters_t * __restrict__ rk_counters,
              const int numProblems)
{
    // Thread-local pointers ...
    // Ordering is phi_woring, rkf working, RHS working
    // such that we can 'peel' off working data easily in subcalls
    __global __ValueType *__restrict__ rwk_rk = rwk_src + __getOffset1D(neq);
    __private FUNC_SIZE(rk_counters_t) my_counters;

    for (int i = __ValueSize * get_global_id(0); i < numProblems; i += __ValueSize * get_global_size(0))
    {
        #if __ValueSize > 1
        for (int k = 0; k < neq; ++k)
        {
            for (int lane = 0; lane < __ValueSize; ++lane)
            {
                const int problem_id = min(i + lane, numProblems-1);
                __write_to(u_in[problem_id * neq + k], lane, rwk[__getIndex(k)]);
            }
        }
        #else
        for (int k = 0; k < neq; ++k)
            rwk[__getIndex(k)] = u_in[i*neq+ k ];
        #endif

        __ValueType h = 0;
        __IntType rkerr = FUNC_SIZE(rk_solve) (
                rk, t_start + i, t_end + i,
                &h, &my_counters, &my_limits, rwk, rwk_rk);

        #if __ValueSize > 1
        for (int k = 0; k < neq; ++k)
        {
            u_out[neq * i + k] = rwk[__getIndex(k)];
        }
        rk_counters[i].niters = my_counters.niters;
        rk_counters[i].nsteps = my_counters.nsteps;
        if (rkerr != RK_SUCCESS)
            rk_counters[i].niters = rkerr;
        #else
            for (int lane = 0; lane < __ValueSize; ++lane)
            {
                const int problem_id = problem_idx + lane;
                if (problem_id < numProblems)
                {
                    for (int k = 0; k < neq; ++k)
                        __read_from(rwk[__getIndex(k)], lane, u_out[neq * i + k]);

                    __read_from(my_counters.nsteps, lane, rk_counters[i].nsteps);
                    // Each lane has the same value ...
                    rk_counters[i].niters = my_counters.niters;
                    if (rkerr != RK_SUCCESS)
                        rk_counters[i].niters = rkerr;
                }
            }
        #endif
    }
}
#endif

#ifndef __EnableQueue
#warning 'Skipping rk_driver_queue kernel'
#else

void __kernel
__attribute__((reqd_work_group_size(__blockSize, 1, 1)))
rk_driver_queue (__global const double * __restrict__ param,
                 __global const double * __restrict__ t_start,
                 __global const double * __restrict__ t_end,
                 __global double * __restrict__ phi,
                 __global const rk_t * __restrict__ rk,
                 __global double * __restrict__ rwk,
                 __global rk_counters_t * __restrict__ rk_counters,
                 const int numProblems,
                 __global int *problemCounter)
{
    const int tid = get_global_id(0);

    const int lenrwk_rk = rk_lenrwk(rk);

    // Thread-local pointers ...
    // Ordering is phi_woring, rkf working, RHS working
    // such that we can 'peel' off working data easily in subcalls
    __global __ValueType *__restrict__ rwk_rk = rwk_src + __getOffset1D(neq);
    __private FUNC_SIZE(rk_counters_t) my_counters;

    __private int problem_idx;

    // Initial problem set and global counter.
    problem_idx = get_global_id(0);

    if (get_local_id(0) == 0)
        atomic_add( problemCounter, get_local_size(0));

    barrier(CLK_GLOBAL_MEM_FENCE);

    //while ((problem_idx = atomic_inc(problemCounter)) < numProblems)
    while (problem_idx < numProblems)
    {
        #if __ValueSize > 1
        for (int k = 0; k < neq; ++k)
        {
            for (int lane = 0; lane < __ValueSize; ++lane)
            {
                const int problem_id = min(i + lane, numProblems-1);
                __write_to(u_in[problem_id * neq + k], lane, rwk[__getIndex(k)]);
            }
        }
        #else
        for (int k = 0; k < neq; ++k)
            rwk[__getIndex(k)] = u_in[problem_idx * neq + k];
        #endif

        // determine maximum / minumum time steps for this set of problems
        __ValueType h = 0;
        __IntType rkerr = FUNC_SIZE(rk_solve) (
                rk, t_start + i, t_end + i,
                &h, &my_counters, &my_limits, rwk, rwk_rk);

        for (int k = 0; k < neq; ++k)
        {
            u_out[neq*(problem_idx) + k] = my_u[__getIndex(k)];
        }

        rk_counters[problem_idx].niters = my_counters.niters;
        rk_counters[problem_idx].nsteps = my_counters.nsteps;
        if (rkerr != RK_SUCCESS)
            rk_counters[problem_idx].niters = rkerr;

        // Get a new problem atomically.
        problem_idx = atomic_inc(problemCounter);
    }
}
#endif
