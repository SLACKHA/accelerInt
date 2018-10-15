
// include error code definitions
#include "error_codes.h"
// and macros / solver definitions
#include "solver.h"
// include the source rate evaluation header
#include "dydt.h"
// inclue the jacobian header
#include "jacobian.h"
// and the struct types
#include "ros_types.h"



// ROS internal routines ...
inline __ValueType ros_getewt(__global const ros_t * __restrict__ ros, const int k, __global const __ValueType * __restrict__y)
{
    const __ValueType ewtk = (ros->s_rtol * fabs(y[__getIndex(k)])) + ros->s_atol;
    return (1.0 / ewtk);
}

inline void ros_dzero(__global __ValueType __restrict__ * x)
{
    for (int k = 0; k < neq; ++k)
        x[__getIndex(k)] = 0.0;
}
inline void ros_dcopy(const __global __ValueType __restrict__ * src, __global __ValueType __restrict__ * dst)
{
    for (int k = 0; k < neq; ++k)
        dst[__getIndex(k)] = src[__getIndex(k)];
}
/*inline void dcopy_if (const int len, const MaskType &mask, const __global __ValueType src[], __global __ValueType dst[])
{
     for (int k = 0; k < len; ++k)
            dst[k] = if_then_else (mask, src[k], dst[k]);
}*/

inline void ros_daxpy1(const double alpha, const __global __ValueType x, __global __ValueType __restrict__* y)
{
    // Alpha is scalar type ... and can be easily checked.
    if (alpha == 1.0)
    {
        for (int k = 0; k < len; ++k)
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
inline void ros_daxpy(const __ValueType alpha, const __global __ValueType __restrict__* x, __global __ValueType __restrict__* y)
{
     // Alpha is vector type ... tedious to switch.
     for (int k = 0; k < neq; ++k)
            y[__getIndex(k)] += alpha * x[__getIndex(k)];
}

__IntType ros_ludec (__global __ValueType * __restrict__ A, __global __IntType * __restrict__ ipiv)
{
    __IntType ierr = SUCCESS;

    const int nelems = vec_step(__ValueType);

    int all_pivk[__ValueSize];

    /* k-th elimination step number */
    for (int k = 0; k < neq; ++k)
    {
        __global __ValueType *A_k = A + __getIndex(k*neq); // pointer to the column

        /* find pivot row number */
        for (int el = 0; el < nelems; ++el)
        {
            int pivk = k;
            double Akp;
            __read_from( A_k[__getIndex(pivk)], el, Akp);
            for (int i = k+1; i < neq; ++i)
            {
                //const double Aki = __read_from( A_k[__getIndex(i)], el);
                double Aki;
                __read_from( A_k[__getIndex(i)], el, Aki);
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
                __global __ValueType *A_i = A; // pointer to the first column
                for (int i = 0; i < neq; ++i, A_i += __getIndex(neq))
                {
                    double Aik, Aip;
                    //const double Aik = __read_from( A_i[__getIndex(k)], el);
                    //const double Aip = __read_from( A_i[__getIndex(pivk)], el);
                    __read_from( A_i[__getIndex(k)], el, Aik);
                    __read_from( A_i[__getIndex(pivk)], el, Aip);
                    __write_to( Aip, el, A_i[__getIndex(k)]);
                    __write_to( Aik, el, A_i[__getIndex(pivk)]);
                }
            }

            all_pivk[el] = pivk;

        } // End scalar section

        ipiv[__getIndex(k)] = __vload(0, all_pivk);

        /* Scale the elements below the diagonal in
         * column k by 1.0/a(k,k). After the above swap
         * a(k,k) holds the pivot element. This scaling
         * stores the pivot row multipliers a(i,k)/a(k,k)
         * in a(i,k), i=k+1, ..., M-1.
         */
        const __ValueType mult = 1.0 / A_k[__getIndex(k)];
        for (int i = k+1; i < neq; ++i)
            A_k[__getIndex(i)] *= mult;

        /* row_i = row_i - [a(i,k)/a(k,k)] row_k, i=k+1, ..., m-1 */
        /* row k is the pivot row after swapping with row l.      */
        /* The computation is done one column at a time,          */
        /* column j=k+1, ..., n-1.                                */
        for (int j = k+1; j < neq; ++j)
        {
            __global __ValueType *A_j = A + __getIndex(j*neq);
            const __ValueType a_kj = A_j[__getIndex(k)];

            /* a(i,j) = a(i,j) - [a(i,k)/a(k,k)]*a(k,j)  */
            /* a_kj = a(k,j), col_k[i] = - a(i,k)/a(k,k) */
            //if (any(a_kj != 0.0)) {
            for (int i = k+1; i < neq; ++i) {
                A_j[__getIndex(i)] -= a_kj * A_k[__getIndex(i)];
            }
            //}
        }
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
        __global __ValueType *A_k = A + __getIndex(k*neq);
        const __ValueType bk = b[__getIndex(k)];
        for (int i = k+1; i < neq; ++i)
            b[__getIndex(i)] -= A_k[__getIndex(i)] * bk;
    }
    /* Solve Ux = y, store solution x in b */
    for (int k = neq-1; k > 0; --k)
    {
        __global __ValueType *A_k = A + __getIndex(k*neq);
        b[__getIndex(k)] /= A_k[__getIndex(k)];
        const __ValueType bk = b[__getIndex(k)];
        for (int i = 0; i < k; ++i)
                b[__getIndex(i)] -= A_k[__getIndex(i)] * bk;
    }
    b[__getIndex(0)] /= A[__getIndex(0)];
}

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

            __global __ValueType *jcol = &Jy[__getIndex(j*neq)];

            //func (neq, tcur, y, jcol, user_data);
            cklib_callback (neq, tcur, y, jcol, user_data);

            const __ValueType delyi = 1. / dely;
            for (int i = 0; i < neq; ++i)
                    jcol[__getIndex(i)] = (jcol[__getIndex(i)] - fy[__getIndex(i)]) * delyi;

            y[__getIndex(j)] = ysav;
    }
}


__IntType ros_solve(__global const ros_t * __restrict__ ros, __ValueType * __restrict__ tcur,
                                        __ValueType * __restrict__ hcur, ros_counters_t_vec *counters,
                                        __global __ValueType __restrict__ *y, __global __IntType __restrict__  *iwk,
                                        __global __ValueType __restrict__* rwk[],
                                        __global __ValueType const * __restrict__ user_data)
{
    __IntType ierr = SUCCESS;

    #define nst (counters->nsteps)
    #define nfe (counters->nfe)
    #define nje (counters->nje)
    #define nlu (counters->nlu)
    #define iter (counters->niters)
    #define h (*hcur)
    #define t (*tcur)
    #define neq (ros->neq)
    #define A(_i,_j) (ros->A[ (((_i)-1)*(_i))/2 + (_j) ] )
    #define C(_i,_j) (ros->C[ (((_i)-1)*(_i))/2 + (_j) ] )

        //printf("h = %e %e %e %f\n", h, ros->h_min, ros->h_max, y[__getIndex(neq-1)]);
    // Estimate the initial step size ...
    {
        __MaskType test = isless(h, ros->h_min);
        if (__any(test))
        {
            ierr = ros_hin(ros, t, &(h), y, rwk, user_data);
            //if (ierr != RK_SUCCESS)
            //   return ierr;
        }
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
    __global __ValueType *fy   = rwk;
    __global __ValueType *ynew = fy + __getOffset1D(neq);
    __global __ValueType *Jy   = ynew + __getOffset1D(neq);
    __global __ValueType *ktmp = Jy + __getOffset1D(neq*neq);
    __global __ValueType *yerr = ynew;
    //__global double *ewt  = &Jy[neq*neq];

    __MaskType done = isless( fabs(t - ros->t_stop), ros->t_round);
    //while (fabs(t - ros->t_stop) > ros->t_round)
    while (__any(__not(done)))
    {
        // Set the error weight array.
        //ros_setewt (ros, y, ewt);

        // Compute the RHS and Jacobian matrix.
        dydt(t, user_data, y, fy, rwk);
        //nfe++;

        //if (jac == NULL)
        {
                jacobian(t, h, y, Jy, user_data);
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
                __global __ValueType *jcol = Jy + __getIndex(j*neq);

                for (int i = 0; i < neq; ++i)
                     jcol[__getIndex(i)] = -jcol[__getIndex(i)];

                jcol[__getIndex(j)] += one_hgamma;
            }
        }

        // Factorization J'
        ros_ludec(neq, Jy, iwk);
        //nlu++;

        for (int s = 0; s < ros->numStages; s++)
        {
            // Compute the function at this stage ...
            if (s == 0)
            {
                 //func (neq, y, fy.getPointer());
            }
            else if (ros->newFunc[s])
            {
                ros_dcopy(y, ynew);

                for (int j = 0; j < s; ++j)
                {
                    const double Asj = A(s,j);
                    //if (Asj != 0.0)
                    {
                        //printf("Asj = %f %d %d\n", Asj, s, j);
                        __global __ValueType *k_j = &ktmp[__getIndex(j*neq)];

                        ros_daxpy1(neq, Asj, k_j, ynew);
                    }
                }

                dydt(t, user_data, y, fy, rwk);
                //nfe++;
            }

            //printf("stage=%d\n", s);
            //for (int k = 0; k < neq; ++k)
            //   printf("fy[%d] = %e\n", k, fy[k]);

            // Build the sub-space vector K
            __global __ValueType *k_s = &ktmp[__getIndex(s*neq)];
            ros_dcopy(fy, k_s);

            for (int j = 0; j < s; j++)
            {
                //if (C(s,j) != 0.0)
                {
                    const __ValueType hCsj = C(s,j) / h;
                    //printf("C/h = %f %d %d\n", hCsj, s, j);

                    __global __ValueType *k_j = &ktmp[__getIndex(j*neq)];
                    ros_daxpy(neq, hCsj, k_j, k_s);
                }
            }

            //printf("k before=%d\n", s);
            //for (int k = 0; k < neq; ++k)
            //   printf("k[%d] = %e\n", k, ks[k]);

            // Solve the current stage ..
            ros_lusol (Jy, iwk, k_s);

             //printf("k after=%d\n", s);
             //for (int k = 0; k < neq; ++k)
             //   printf("k[%d] = %e\n", k, ks[k]);
        }

        // Compute the error estimation of the trial solution
        ros_dzero(yerr);

        for (int j = 0; j < ros->numStages; ++j)
        {
            //if (ros->E[j] != 0.0)
            {
                __global __ValueType *k_j = &ktmp[__getIndex(j*neq)];
                ros_daxpy1(ros->E[j], k_j, yerr);
            }
        }

        __ValueType herr = fmax(1.0e-20, get_wnorm(ros, yerr, y));

        // Is there error acceptable?
        //int accept = (herr <= 1.0) || (h <= ros->h_min);
        __MaskType accept = islessequal(herr, 1.0);
        accept |= islessequal(h, ros->h_min);
        accept &= __not(done);

        if (__any(accept))
        {
            // Update solution for those lanes that've passed the smell test.

            t   = __select (t,   t + h  , accept);
            nst = __select (nst, nst + 1, accept);

            done = isless( fabs(t - ros->t_stop), ros->t_round);

            // Need to actually compute the new solution since it was delayed from above.
            ros_dcopy(neq, y, ynew);
            for (int j = 0; j < ros->numStages; ++j)
            {
                //if (ros->M[j] != 0.0)
                {
                    __global __ValueType *k_j = &ktmp[__getIndex(j*neq)];
                    ros_daxpy1 (ros->M[j], k_j, ynew);
                }
            }

            for (int k = 0; k < neq; k++)
                y[__getIndex(k)] = __select(y[__getIndex(k)], ynew[__getIndex(k)], accept);
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
        h = fmin(h, ros->h_max);
        h = fmax(h, ros->h_min);

        // Stretch the final step if we're really close and we didn't just fail ...
        h = __select(h, ros->t_stop - t, accept & isless(fabs((t + h) - ros->t_stop), ros->h_min));

        // Don't overshoot the final time ...
        h = __select(h, ros->t_stop - t, __not(done) & isgreater((t + h),  ros->t_stop));

        ++iter;
        if (ros->max_iters && iter > ros->max_iters) {
                 ierr = TOO_MUCH_WORK;
                 //printf("(iter > max_iters)\n");
                 break;
        }
    }

    return ierr;

    #undef nst
    #undef nfe
    #undef nje
    #undef nlu
    #undef iter
    #undef h
    #undef t
    #undef neq
    #undef A
    #undef C
}
