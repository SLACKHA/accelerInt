
// include error code definitions
#include "error_codes.h"
// and macros / solver definitions
#include "solver.h"
// include the source rate evaluation header
#include "dydt.h"
// and the struct types
#include "rkf45_types.h"

// check for required defines
#ifndef neq
#pragma error "Number of equations not defined!"
#endif
#ifndef rwk_lensrc
#pragma error "Length of source-rate evaluation working buffer not defined"
#endif
#ifndef rwk_lensol
#pragma error "Length of solver working buffer not defined"
#endif
#ifndef __ValueType
#pragma error "Value type not defined!"
#endif

// \brief Indexing macro for _all_ arrays in RKF45 solver (all arrays are size neq)
#define __getIndex(idx) (__getIndex1D(neq, idx))


// Single-step function
int rkf45 (const __ValueType h, const __ValueType t,
           __global const __ValueType* __restrict__ y,
           __global __ValueType* __restrict__ y_out,
           __global __ValueType* __restrict__ trunc_err,
           __global __ValueType* rwk,
           __global __ValueType const * __restrict__ user_data,
           const int solver_offset)
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
    #define a4 ( 0.535722994391612)
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
    __global __ValueType* __restrict__ f1   = rwk + __getOffset1D(solver_offset);
    __global __ValueType* __restrict__ f2   = rwk + __getOffset1D(solver_offset + neq);
    __global __ValueType* __restrict__ f3   = rwk + __getOffset1D(solver_offset + 2*neq);
    __global __ValueType* __restrict__ f4   = rwk + __getOffset1D(solver_offset + 3*neq);
    __global __ValueType* __restrict__ f5   = rwk + __getOffset1D(solver_offset + 4*neq);
    __global __ValueType* __restrict__ f6   = rwk + __getOffset1D(solver_offset + 5*neq);
    __global __ValueType* __restrict__ ytmp = rwk + __getOffset1D(solver_offset + 6*neq);
    // the portion of the rwk vector that's allocated for the source rate evaluation
    // y_out is at 7 * neq, hence we go to 8 for the total offset
    __global __ValueType* __restrict__ rwk_dydt = rwk + __getOffset1D(rwk_lensol);

    // 1)
    dydt(t, user_data, y, f1, rwk_dydt);

    for (int k = 0; k < neq; k++)
    {
        //f1[k] = h * ydot[k];
        f1[__getIndex(k)] *= h;
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
    dydt(t + h5 * h, user_data, ytmp, f5, rwk_dydt);

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
        trunc_err[__getIndex(k)] = fabs(r5 - r4);

        // Update solution.
        y_out[__getIndex(k)] = y[__getIndex(k)] + r5; // Local extrapolation
    }

    return OCL_SUCCESS;

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

__IntType rk_solve (__global const rk_t * __restrict__ rk,
                    __private __ValueType const t_start,
                    __private __ValueType const t_end,
                    __private __ValueType hcur,
                    __private rk_counters_t_vec * __restrict__ counters,
                    __global __ValueType* __restrict__ y,
                    __global __ValueType* rwk,
                    __global __IntType* __restrict__ iwk,
                    __global __ValueType const * __restrict__ user_data,
                    const int driver_offset)
{
    UNUSED(iwk);

    __ValueType t = t_start;
    #define t_round ((t_end - t_start) * DBL_EPSILON)
    #define h_min (t_round * 100)
    #define h_max ((t_end - t_start) / rk->min_iters)
    #define iter (counters->niters)
    #define nst (counters->nsteps)

    __IntType ierr = OCL_SUCCESS;
    #ifndef CONSTANT_TIMESTEP
    // Estimate the initial step size ...
    {
        __MaskType test = isless(hcur, h_min);
        if (__any(test))
        {
            ierr = get_hin(rk, t, t_end, &hcur, y, rwk, user_data,
                           driver_offset);
        }
    }
    #else
    hcur = CONSTANT_TIMESTEP;
    #endif

    nst = 0;
    iter = 0;

    __MaskType done = isless(fabs(t - t_end), fabs(t_round));
    __global __ValueType * __restrict__ ytmp = rwk + __getOffset1D(driver_offset);
    __global __ValueType * __restrict__ trunc_err = rwk + __getOffset1D(driver_offset + neq);

    while (__any(__not(done)))
    {

        // Take a trial step over h_cur ...
        rkf45(hcur, t, y, ytmp, trunc_err, rwk, user_data, driver_offset + 2 * neq);

        #ifndef CONSTANT_TIMESTEP
        __ValueType herr = fmax(1e-20, get_wnorm(rk, trunc_err, y));

        // Is there error acceptable?
        __MaskType accept = islessequal(herr, 1.0);
        accept |= islessequal(hcur, h_min);
        accept &= __not(done);
        #else
        __MaskType accept = TRUE;
        #endif

        // update solution ...
        if (__any(accept))
        {
            t   = __select (t,   t + hcur  , accept);
            nst = __select (nst, nst + 1, accept);

            for (int k = 0; k < neq; k++)
                y[__getIndex(k)] = __select(y[__getIndex(k)], ytmp[__getIndex(k)], accept);

            done = isless( fabs(t - t_end), fabs(t_round));
        }

        #ifndef CONSTANT_TIMESTEP
        __ValueType fact = sqrt( sqrt(1.0 / herr) ) * (0.840896415);

        // Restrict the rate of change in dt
        fact = fmax(fact, 1.0 / rk->adaption_limit);
        fact = fmin(fact,       rk->adaption_limit);

        // Apply grow/shrink factor for next step.
        hcur = __select(hcur * fact, hcur, done);

        // Limit based on the upper/lower bounds
        hcur = fmin(hcur, h_max);
        hcur = fmax(hcur, h_min);
        #endif

        // Stretch the final step if we're really close and we didn't just fail ...
        hcur = __select(hcur, t_end - t, accept & isless(fabs((t + hcur) - t_end), h_min));

        // Don't overshoot the final time ...
        hcur = __select(hcur, t_end - t, __not(done) & isgreater((t + hcur), t_end));

        ++iter;
        if (rk->max_iters && iter > rk->max_iters) {
            ierr = OCL_TOO_MUCH_WORK;
            //printf("(iter > max_iters)\n");
            break;
        }
    }

    return ierr;

    #undef t_round
    #undef h_min
    #undef h_max
    #undef iter
    #undef nst
}
