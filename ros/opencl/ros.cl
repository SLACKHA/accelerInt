
// include error code definitions
#include "error_codes.h"
// and macros / solver definitions
#include "solver.h"
// include the source rate evaluation header
#include "dydt.h"
// inclue the jacobian header
#include "jacob.h"
// and the struct types
#include "ros_types.h"

#ifndef rwk_lensrc
#pragma error "Length of source-rate evaluation working buffer not defined"
#endif
#ifndef rwk_lensol
#pragma error "Length of solver working buffer not defined"
#endif

// \brief Indexing macro for neq sized arrays in ROS solvers
#define __getIndex(idx) (__getIndex1D(neq, idx))
#define __getIndexJac(row, col) (__getIndex2D(neq, neq, row, col))

/*#if __ValueSize > 1
#define tid (!get_group_id(0))
#define print(prefix, size, arr) \
{ \
    if (tid) printf("%s={", prefix); \
    for (int i = 0; i < size; ++i) \
    { \
        if (tid && i) printf(", "); \
        if (tid) printf("%e", (arr)[i].s0); \
    } \
    if (tid) printf("}\n"); \
}

#define printi(prefix, size, arr) \
{ \
    if (tid) printf("%s={", prefix); \
    for (int i = 0; i < size; ++i) \
    { \
        if (tid && i) printf(", "); \
        if (tid) printf("%d", (arr)[i].s0); \
    } \
    if (tid) printf("}\n"); \
}

#define printv(prefix, val) \
{ \
    if (tid) printf("%s={%e}\n", prefix, (val.s0); \
}
#else
#define print(prefix, size, arr) \
{ \
    printf("%s={", prefix); \
    for (int i = 0; i < size; ++i) \
    { \
        if (i) printf(", "); \
        printf("%e", arr[i]); \
    } \
    printf("}\n"); \
}

#define printi(prefix, size, arr) \
{ \
    printf("%s={", prefix); \
    for (int i = 0; i < size; ++i) \
    { \
        if (i) printf(", "); \
        printf("%d", arr[i]); \
    } \
    printf("}\n"); \
}

#define printv(prefix, val) \
{ \
    printf("%s={%e}\n", prefix, val); \
}
#endif*/

// ROS internal routines ...
inline __ValueType ros_getewt(__global const ros_t * __restrict__ ros, const int k, __global const __ValueType * __restrict__  y)
{
    const __ValueType ewtk = (ros->s_rtol * fabs(y[__getIndex(k)])) + ros->s_atol;
    return (1.0 / ewtk);
}

inline void ros_dzero(__global __ValueType * __restrict__ x)
{
    for (int k = 0; k < neq; ++k)
        x[__getIndex(k)] = 0.0;
}
inline void ros_dcopy(const __global __ValueType * __restrict__ src, __global __ValueType * __restrict__ dst)
{
    for (int k = 0; k < neq; ++k)
        dst[__getIndex(k)] = src[__getIndex(k)];
}

//! \brief copy for ktmp
inline void ros_dcopy1(const __global __ValueType * __restrict__ src, __global __ValueType * __restrict__ dst)
{
    for (int k = 0; k < neq; ++k)
        dst[__getIndex(k)] = src[__getIndex(k)];
}
/*inline void dcopy_if (const int len, const MaskType &mask, const __global __ValueType src[], __global __ValueType dst[])
{
     for (int k = 0; k < len; ++k)
            dst[k] = if_then_else (mask, src[k], dst[k]);
}*/

inline void ros_daxpy1(const double alpha, const __global __ValueType * __restrict__ x, __global __ValueType * __restrict__ y)
{
    // Alpha is scalar type ... and can be easily checked.
    if (alpha == 1.0)
    {
        for (int k = 0; k < neq; ++k)
            y[__getIndex(k)] += x[__getIndex(k)];
    }
    else if (alpha == -1.0)
    {
        for (int k = 0; k < neq; ++k)
            y[__getIndex(k)] -= x[__getIndex(k)];
    }
    else if (alpha != 0.0)
    {
        for (int k = 0; k < neq; ++k)
            y[__getIndex(k)] += alpha * x[__getIndex(k)];
    }
}
inline void ros_daxpy(const __ValueType alpha, const __global __ValueType * __restrict__ x, __global __ValueType * __restrict__ y)
{
    // Alpha is vector type ... tedious to switch.
    for (int k = 0; k < neq; ++k)
        y[__getIndex(k)] += alpha * x[__getIndex(k)];
}

__IntType ros_ludec (__global __ValueType * __restrict__ A, __global __IntType * __restrict__ ipiv)
{
    __IntType ierr = OCL_SUCCESS;

    const int nelems = vec_step(__ValueType);

    int all_pivk[__ValueSize];

    /* k-th elimination step number */
    for (int k = 0; k < neq; ++k)
    {
        /* find pivot row number */
        for (int el = 0; el < nelems; ++el)
        {
            int pivk = k;
            double Akp;
            __read_from( A[__getIndexJac(pivk, k)], el, Akp);
            for (int i = k+1; i < neq; ++i)
            {
                //const double Aki = __read_from( A_k[__getIndex(i)], el);
                double Aki;
                __read_from( A[__getIndexJac(i, k)], el, Aki);
                if (fabs(Aki) > fabs(Akp))
                {
                    pivk = i;
                    Akp = Aki;
                }
            }

            // Test for singular value ...
            if (Akp == 0.0)
            {
                __write_to((k+1), el, ierr);
                //printf("Singular value %d %d\n", k, el);
                break;
            }

            /* swap a(k,1:N) and a(piv,1:N) if necessary */
            if (pivk != k)
            {
                for (int i = 0; i < neq; ++i)
                {
                    double Aik, Aip;
                    //const double Aik = __read_from( A_i[__getIndex(k)], el);
                    //const double Aip = __read_from( A_i[__getIndex(pivk)], el);
                    __read_from( A[__getIndexJac(k, i)], el, Aik);
                    __read_from( A[__getIndexJac(pivk, i)], el, Aip);
                    __write_to( Aip, el, A[__getIndexJac(k, i)]);
                    __write_to( Aik, el, A[__getIndexJac(pivk, i)]);
                }
            }

            all_pivk[el] = pivk;

        } // End scalar section

        //print("A-pivoted", neq * neq, A);

        ipiv[__getIndex(k)] = __vload(0, all_pivk);

        /* Scale the elements below the diagonal in
         * column k by 1.0/a(k,k). After the above swap
         * a(k,k) holds the pivot element. This scaling
         * stores the pivot row multipliers a(i,k)/a(k,k)
         * in a(i,k), i=k+1, ..., M-1.
         */
        const __ValueType mult = 1.0 / A[__getIndexJac(k, k)];
        for (int i = k+1; i < neq; ++i)
            A[__getIndexJac(i, k)] *= mult;

        //print("A-scaled", neq * neq, A);

        /* row_i = row_i - [a(i,k)/a(k,k)] row_k, i=k+1, ..., m-1 */
        /* row k is the pivot row after swapping with row l.      */
        /* The computation is done one column at a time,          */
        /* column j=k+1, ..., n-1.                                */
        for (int j = k+1; j < neq; ++j)
        {
            const __ValueType a_kj = A[__getIndexJac(k, j)];

            /* a(i,j) = a(i,j) - [a(i,k)/a(k,k)]*a(k,j)  */
            /* a_kj = a(k,j), col_k[i] = - a(i,k)/a(k,k) */
            //if (any(a_kj != 0.0)) {
            for (int i = k+1; i < neq; ++i) {
                A[__getIndexJac(i, j)] -= a_kj * A[__getIndexJac(i, k)];
            }
            //}
        }

        //print("A-backsub'd", neq * neq, A);
    }

    return ierr;
}

void ros_lusol(__global __ValueType * __restrict__ A, __global __IntType * __restrict__ ipiv,
               __global __ValueType * __restrict__ b)
{
    /* Permute b, based on pivot information in p */
    for (int k = 0; k < neq; ++k)
    {
        //__MaskType notequal_k = (ipiv[__getIndex(k)] != k);
        //if (__any(notequal_k) )
        //      if (__any( isnotequal(ipiv[__getIndex(k)], k) ))
        if (__any( ipiv[__getIndex(k)] != k  ) )
        {
            for (int el = 0; el < __ValueSize; ++el)
            {
                //const int pivk = __read_from(ipiv[__getIndex(k)], el);
                int pivk;
                __read_from(ipiv[__getIndex(k)], el, pivk);
                if (pivk != k)
                {
                    double bk, bp;
                    //const double bk = __read_from(b[__getIndex(k)], el);
                    //const double bp = __read_from(b[__getIndex(pivk)], el);
                    __read_from(b[__getIndex(k)], el, bk);
                    __read_from(b[__getIndex(pivk)], el, bp);
                    __write_to( bp, el, b[__getIndex(k)]);
                    __write_to( bk, el, b[__getIndex(pivk)]);
                }
            }
        }
    }

    /* Solve Ly = b, store solution y in b */
    for (int k = 0; k < neq-1; ++k)
    {
        const __ValueType bk = b[__getIndex(k)];
        for (int i = k+1; i < neq; ++i)
            b[__getIndex(i)] -= A[__getIndexJac(i, k)] * bk;
    }
    /* Solve Ux = y, store solution x in b */
    for (int k = neq-1; k > 0; --k)
    {
        b[__getIndex(k)] /= A[__getIndexJac(k, k)];
        const __ValueType bk = b[__getIndex(k)];
        for (int i = 0; i < k; ++i)
            b[__getIndex(i)] -= A[__getIndexJac(i, k)] * bk;
    }
    b[__getIndex(0)] /= A[__getIndexJac(0, 0)];
}

/*
void ros_fdjac(__global const ros_t *ros, const __ValueType tcur, const __ValueType hcur, __global __ValueType *y, __global __ValueType *fy, __global __ValueType *Jy, __private void *user_data)
{
    // Norm of fy(t) ...
    __ValueType fnorm = get_wnorm( ros, fy, y );

    // Safety factors ...
    const double sround = sqrt( ros_uround() );
    __ValueType r0 = (1000. * ros_uround() * neq) * (hcur * fnorm);
    //if (r0 == 0.) r0 = 1.;
    __MaskType r0_is_zero = isequal(r0, 0.0);
    r0 = __select(r0, 1.0, r0_is_zero);

    // Build each column vector ...
    for (int j = 0; j < neq; ++j)
    {
        const __ValueType ysav = y[__getIndex(j)];
        const __ValueType ewtj = ros_getewt(ros, j, y);
        const __ValueType dely = fmax( sround * fabs(ysav), r0 / ewtj );
        y[__getIndex(j)] += dely;

        //func (neq, tcur, y, jcol, user_data);
        dydt (tcur, user_data, y, jcol, rwk);

        const __ValueType delyi = 1. / dely;
        for (int i = 0; i < neq; ++i)
            Jy[__getIndexJac(i, j)] = (Jy[__getIndexJac(i, j)] - fy[__getIndex(i)]) * delyi;

        y[__getIndex(j)] = ysav;
    }
}*/

__IntType ros_solve (__global const ros_t * __restrict__ ros,
                     __private __ValueType const t_start,
                     __private __ValueType const t_end,
                     __private __ValueType hcur,
                     __private ros_counters_t_vec * __restrict__ counters,
                     __global __ValueType* __restrict__ y,
                     __global __ValueType* __restrict__ rwk,
                     __global __IntType* __restrict__ iwk,
                     __global __ValueType const * __restrict__ user_data,
                     const int driver_offset)
{
    __IntType ierr = OCL_SUCCESS;
    __ValueType t = t_start;
    #define t_round ((t_end - t_start) * DBL_EPSILON)
    #define h_min (t_round * 100)
    #define h_max ((t_end - t_start) / ros->min_iters)
    #define nst (counters->nsteps)
    #define nfe (counters->nfe)
    #define nje (counters->nje)
    #define nlu (counters->nlu)
    #define iter (counters->niters)
    #define h (hcur)
    #define A(_i,_j) (ros->A[ (((_i)-1)*(_i))/2 + (_j) ] )
    #define C(_i,_j) (ros->C[ (((_i)-1)*(_i))/2 + (_j) ] )

    //printv("t", t);
    //print("y0", neq, y);

        //printf("h = %e %e %e %f\n", h, ros->h_min, ros->h_max, y[__getIndex(neq-1)]);
    // Estimate the initial step size ...
    {
        __MaskType test = isless(h, h_min);
        if (__any(test))
        {
            ierr = get_hin(ros, t, t_end,
                           &h, y, rwk, user_data,
                           driver_offset);
            //if (ierr != RK_SUCCESS)
            //   return ierr;
        }
        //printv("h", hcur);
        #if 0
        #if (__ValueSize == 1)
                    printf("hin = %e %e %e\n", h, ros->h_min, ros->h_max);
        #else
                    printf("hin = %v"STRINGIFY(__ValueSize)"e %e %e\n", h, ros->h_min, ros->h_max);
        #endif
        #endif
    }
        //printf("hin = %e %e %e %f\n", h, ros->h_min, ros->h_max, y[__getIndex(neq-1)]);

    // Zero the counters ...
    nst = 0;
    //nfe = 0;
    //nlu = 0;
    //nje = 0;
    iter = 0;

    // Set the work arrays ...
    __global __ValueType *fy   = rwk + __getOffset1D(driver_offset);
    __global __ValueType *ynew = fy + __getOffset1D(neq);
    // ktmp is ros->numStages number of separate vectors each sized neq
    __global __ValueType *ktmp = ynew + __getOffset1D(neq);
    // \note IMPORTANT, Jy must be _after_ ktmp -- the reason is that `get_hin` reserves
    //       three neq-sized vectors.  If Jy was before ktmp, the ranges of the `get_hin`
    //       vectors and Jy would overlap, and we would end up with write-races between
    //       the various threads.  It may be safer to just use the additional memory such
    //       that we don't forget this in the future.
    __global __ValueType *Jy   = ktmp + __getOffset1D(ros->numStages * neq);
    __global __ValueType *rwk_jac = rwk + __getOffset1D(rwk_lensol);
    __global __ValueType *yerr = ynew;
    //__global double *ewt  = &Jy[neq*neq];

    __MaskType done = isless( fabs(t - t_end), t_round);
    while (__any(__not(done)))
    {
        // Set the error weight array.
        //ros_setewt (ros, y, ewt);

        // Compute the RHS and Jacobian matrix.
        dydt(t, user_data, y, fy, rwk_jac);
        //print("y", neq, y);
        //print("dy", neq, fy);
        //printv("t_iter", t);
        //printv("h_iter", hcur);
        //nfe++;

        //if (jac == NULL)
        {
            //print("J", neq * neq, Jy);
            jacob(t, user_data, y, Jy, rwk_jac);
         //nfe += neq;
        }
        //else
        //{
        //   jac (neq, t, y, Jy, user_data);
        //}

        //nje++;

        // Construct iteration matrix J' := 1/(gamma*h) - J
        {
            const __ValueType one_hgamma = 1.0 / (h * ros->gamma[0]);

            for (int j = 0; j < neq; ++j)
            {
                for (int i = 0; i < neq; ++i)
                    Jy[__getIndexJac(i, j)] = -Jy[__getIndexJac(i, j)];

                Jy[__getIndexJac(j, j)] += one_hgamma;
            }
        }

        //print("Jiter", neq * neq, Jy);

        // Factorization J'
        ros_ludec(Jy, iwk);

        //print("Jlu", neq * neq, Jy);
        //printi("ipiv", neq, iwk);
        //nlu++;

        for (int s = 0; s < ros->numStages; s++)
        {
            // if (tid) printf("stage: %d\n", s);
            // Compute the function at this stage ...
            if (s > 0 && ros->newFunc[s])
            {
                ros_dcopy(y, ynew);
                //print("ynew", neq, ynew);

                for (int j = 0; j < s; ++j)
                {
                    const double Asj = A(s,j);
                    //if (Asj != 0.0)
                    {
                        //printf("Asj = %f %d %d\n", Asj, s, j);
                        ros_daxpy1(Asj, ktmp + __getOffset1D(j * neq), ynew);
                        //print("ynew-iter", neq, ynew);
                    }
                }

                //print("y-eval", neq, ynew);
                dydt(t, user_data, ynew, fy, rwk_jac);
                //nfe++;
            }

            // Build the sub-space vector K
            //print("fy", neq, fy);
            //print("ktmp-precopy", neq, ktmp + __getOffset1D(s * neq));
            ros_dcopy1(fy, ktmp + __getOffset1D(s * neq));
            //print("ktmp", neq, ktmp + __getOffset1D(s * neq));

            for (int j = 0; j < s; j++)
            {
                //if (C(s,j) != 0.0)
                {
                    const __ValueType hCsj = C(s,j) / h;
                    //printf("C/h = %f %d %d\n", hCsj, s, j);

                    ros_daxpy(hCsj, ktmp + __getOffset1D(j * neq), ktmp + __getOffset1D(s * neq));
                }
            }

            //print("ktmp-Cupdate", neq, ktmp + __getOffset1D(s * neq));

            // Solve the current stage ..
            ros_lusol (Jy, iwk, ktmp + __getOffset1D(s * neq));

            //print("ktmp-lusol", neq, ktmp + __getOffset1D(s * neq));
        }

        // Compute the error estimation of the trial solution
        ros_dzero(yerr);

        for (int j = 0; j < ros->numStages; ++j)
        {
            //if (ros->E[j] != 0.0)
            {
                ros_daxpy1(ros->E[j], ktmp + __getOffset1D(j * neq), yerr);
            }
        }
        //print("yerr", neq, yerr);

        __ValueType herr = fmax(1.0e-20, get_wnorm(ros, yerr, y));
        //printv("herr", herr);

        // Is there error acceptable?
        //int accept = (herr <= 1.0) || (h <= ros->h_min);
        __MaskType accept = islessequal(herr, 1.0);
        accept |= islessequal(h, h_min);
        accept &= __not(done);

        if (__any(accept))
        {
            // Update solution for those lanes that've passed the smell test.

            t   = __select (t,   t + h  , accept);
            nst = __select (nst, nst + 1, accept);

            done = isless( fabs(t - t_end), t_round);

            // Need to actually compute the new solution since it was delayed from above.
            ros_dcopy(y, ynew);
            //print("ynew", neq, ynew)
            for (int j = 0; j < ros->numStages; ++j)
            {
                //printf("iter-stage:%d\n", j);
                //if (ros->M[j] != 0.0)
                {
                    ros_daxpy1 (ros->M[j], ktmp + __getOffset1D(j * neq), ynew);
                }
                //print("ynew-iter", neq, ynew)
            }

            for (int k = 0; k < neq; k++)
                y[__getIndex(k)] = __select(y[__getIndex(k)], ynew[__getIndex(k)], accept);

            //print("ynew-final", neq, y)
        }

        __ValueType fact = 0.9 * pow( 1.0 / herr, (1.0/ros->ELO));

        // Restrict the rate of change in dt
        fact = fmax(fact, 1.0 / ros->adaption_limit);
        fact = fmin(fact,       ros->adaption_limit);

        #if 0
            //if (iter % 100 == 0)
                    {
            #if (__ValueSize == 1)
                         printf("iter = %d: accept=%d, done=%d t=%e, fact=%f %f %e\n", iter, accept, done, t, fact, y[neq-1], h);
            #else
                         printf("iter = %d: accept=%v"STRINGIFY(__ValueSize)"d, done=%v"STRINGIFY(__ValueSize)"d t=%v"STRINGIFY(__ValueSize)"e, fact=%v"STRINGIFY(__ValueSize)"f %v"STRINGIFY(__ValueSize)"f %v"STRINGIFY(__ValueSize)"e\n", iter, accept, done, t, fact, y[neq-1], h);
            #endif
                    }
        #endif

        // Apply grow/shrink factor for next step.
        h = __select(h * fact, h, done);

        // Limit based on the upper/lower bounds
        h = fmin(h, h_max);
        h = fmax(h, h_min);

        // Stretch the final step if we're really close and we didn't just fail ...
        h = __select(h, t_end - t, accept & isless(fabs((t + h) - t_end), h_min));

        // Don't overshoot the final time ...
        h = __select(h, t_end - t, __not(done) & isgreater((t + h),  t_end));

        ++iter;
        if (ros->max_iters && iter > ros->max_iters) {
                 ierr = OCL_TOO_MUCH_WORK;
                 //printf("(iter > max_iters)\n");
                 break;
        }
    }

    return ierr;

    #undef t_round
    #undef h_min
    #undef h_max
    #undef nst
    #undef nfe
    #undef nje
    #undef nlu
    #undef iter
    #undef h
    #undef A
    #undef C
}
