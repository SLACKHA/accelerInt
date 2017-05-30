#ifndef __OPENCL_VERSION__

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

#endif

//#include <assert.h>

#include <rk.h>

#define __inline inline

#define rk_uround() ( DBL_EPSILON )

//     f1 = dt*f(t,x)
//     f2 = dt*f(t+ c20*dt,x + c21*f1)
//     f3 = dt*f(t+ c30*dt,x + c31*f1 + c32*f2)
//     f4 = dt*f(t+ c40*dt,x + c41*f1 + c42*f2 + c43*f3)
//     f5 = dt*f(t+dt,x + c51*f1 + c52*f2 + c53*f3 + c54*f4)
//     f6 = dt*f(t+ c60*dt,x + c61*f1 + c62*f2 + c63*f3 + c64*f4 + c65*f5)
//
//     fifth-order runge-kutta integration
//        x5 = x + b1*f1 + b3*f3 + b4*f4 + b5*f5 + b6*f6
//     fourth-order runge-kutta integration
//        x  = x + a1*f1 + a3*f3 + a4*f4 + a5*f5

// Single-step function
__inline
int rkf45 (const int neq, const double h, __global double *restrict y, __global double *restrict y_out, __global double *rwk, RHS_Function_t func, __private void *user_data)
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
   __global double *restrict f1   = rwk ;
   __global double *restrict f2   = rwk + __getIndex(  neq) ;
   __global double *restrict f3   = rwk + __getIndex(2*neq) ;
   __global double *restrict f4   = rwk + __getIndex(3*neq) ;
   __global double *restrict f5   = rwk + __getIndex(4*neq) ;
   __global double *restrict f6   = rwk + __getIndex(5*neq) ;
   __global double *restrict ytmp = rwk + __getIndex(6*neq) ;

   // 1)
   cklib_callback(neq, 0.0, y, f1, user_data);

   for (int k = 0; k < neq; k++)
   {
      //f1[k] = h * ydot[k];
      f1[__getIndex(k)] *= h;
      ytmp[__getIndex(k)] = y[__getIndex(k)] + c21 * f1[__getIndex(k)];
   }

   // 2)
   cklib_callback(neq, 0.0, ytmp, f2, user_data);

   for (int k = 0; k < neq; k++)
   {
      //f2[k] = h * ydot[k];
      f2[__getIndex(k)] *= h;
      ytmp[__getIndex(k)] = y[__getIndex(k)] + c31 * f1[__getIndex(k)] + c32 * f2[__getIndex(k)];
   }

   // 3)
   cklib_callback(neq, 0.0, ytmp, f3, user_data);

   for (int k = 0; k < neq; k++) {
      //f3[k] = h * ydot[k];
      f3[__getIndex(k)] *= h;
      ytmp[__getIndex(k)] = y[__getIndex(k)] + c41 * f1[__getIndex(k)] + c42 * f2[__getIndex(k)] + c43 * f3[__getIndex(k)];
   }

   // 4)
   cklib_callback(neq, 0.0, ytmp, f4, user_data);

   for (int k = 0; k < neq; k++) {
      //f4[k] = h * ydot[k];
      f4[__getIndex(k)] *= h;
      ytmp[__getIndex(k)] = y[__getIndex(k)] + c51 * f1[__getIndex(k)] + c52 * f2[__getIndex(k)] + c53 * f3[__getIndex(k)] + c54 * f4[__getIndex(k)];
   }

   // 5)
   cklib_callback(neq, 0.0, ytmp, f5, user_data);

   for (int k = 0; k < neq; k++) {
      //f5[k] = h * ydot[k];
      f5[__getIndex(k)] *= h;
      ytmp[__getIndex(k)] = y[__getIndex(k)] + c61*f1[__getIndex(k)] + c62*f2[__getIndex(k)] + c63*f3[__getIndex(k)] + c64*f4[__getIndex(k)] + c65*f5[__getIndex(k)];
   }

   // 6)
   cklib_callback(neq, 0.0, ytmp, f6, user_data);

   for (int k = 0; k < neq; k++)
   {
      //const T f6 = h * ydot[k];
      f6[__getIndex(k)] *= h;

      // 5th-order RK value.
      const double r5 = b1*f1[__getIndex(k)] + b3*f3[__getIndex(k)] + b4*f4[__getIndex(k)] + b5*f5[__getIndex(k)] + b6*f6[__getIndex(k)];

      // 4th-order RK residual.
      const double r4 = a1*f1[__getIndex(k)] + a3*f3[__getIndex(k)] + a4*f4[__getIndex(k)] + a5*f5[__getIndex(k)];

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

__inline
double rk_wnorm (__global const rk_t *restrict rk, __global const double *restrict x, __global const double *restrict y)
{
   const int neq = rk->neq;
   double sum = 0;
   for (int k = 0; k < neq; k++)
   {
      double ewt = (rk->s_rtol * fabs(y[__getIndex(k)])) + rk->s_atol;
      double prod = x[__getIndex(k)] / ewt;
      sum += (prod*prod);
   }

   return sqrt(sum / (double)neq);
}

int rk_hin (__global const rk_t *restrict rk, const double t, double *h0, __global double *restrict y, __global double *rwk, RHS_Function_t func, __private void *user_data)
{
   //value_type tround = tdist * this->uround();
   //double tdist = t_stop - t;
   //double tround = tdist * rk_uround();

   // Set lower and upper bounds on h0, and take geometric mean as first trial value.
   // Exit with this value if the bounds cross each other.

   //rk->h_min = fmax(tround * 100.0, rk->h_min);
   //rk->h_max = fmin(tdist, rk->h_max);

   const int neq = rk->neq;

   __global double *restrict ydot  = rwk;
   __global double *restrict y1    = ydot + __getIndex(neq);
   __global double *restrict ydot1 = y1 + __getIndex(neq);

   int need_ydot = 1;

   // Adjust upper bound based on ydot ...
/*      if (0)
      {
         need_ydot = false;

         // compute ydot at t=t0
         func (neq, y, ydot);
         ++this->nfe;

         for (int k = 0; k < neq; k++)
         {
            value_type dely = 0.1 * fabs(y[k]) + this->atol;
            value_type hub0 = hub;
            if (hub * fabs(ydot[k]) > dely) hub = dely / fabs(ydot[k]);
            //printf("k=%d, hub0 = %e, hub = %e\n", k, hub0, hub);
         }
      }*/

   double hlb = rk->h_min;
   double hub = rk->h_max;

   double hg = sqrt(hlb*hub);

   if (hub < hlb)
   {
      *h0 = hg;
      return RK_SUCCESS;
   }

   // Start iteration to find solution to ... {WRMS norm of (h0^2 y'' / 2)} = 1

   const int miters = 10;
   int hnew_is_ok = 0;
   double hnew = hg;
   int iter = 0;
   int ierr = RK_SUCCESS;

   // compute ydot at t=t0
   if (need_ydot)
   {
      cklib_callback(neq, 0.0, y, ydot, user_data);
      //++rk->nfe;
      need_ydot = 0;
   }

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
      cklib_callback (neq, 0.0, y1, ydot1, user_data);
      //++rk->nfe;

      // Compute WRMS norm of y''
      #ifdef __INTEL_COMPILER
      #pragma ivdep
      #endif
      for (int k = 0; k < neq; k++)
         y1[__getIndex(k)] = (ydot1[__getIndex(k)] - ydot[__getIndex(k)]) / hg;

      double yddnrm = rk_wnorm (rk, y1, y);

      //std::cout << "iter " << iter << " hg " << hg << " y'' " << yddnrm << std::endl;
      //std::cout << "ydot " << ydot[neq-1] << std::endl;

      // should we accept this?
      if (hnew_is_ok || iter == miters)
      {
         hnew = hg;
         //if (iter == miters) fprintf(stderr, "ERROR_HIN_MAX_ITERS\n");
         ierr = (hnew_is_ok) ? RK_SUCCESS : RK_HIN_MAX_ITERS;
         break;
      }

      // Get the new value of h ...
      hnew = (yddnrm*hub*hub > 2.0) ? sqrt(2.0 / yddnrm) : sqrt(hg * hub);

      // test the stopping conditions.
      double hrat = hnew / hg;

      // Accept this value ... the bias factor should bring it within range.
      if ( (hrat > 0.5) && (hrat < 2.0) )
      //if ( all(hrat > 0.5) && all(hrat < 2.0) )
         hnew_is_ok = 1;

      // If y'' is still bad after a few iterations, just accept h and give up.
      //if ( (iter > 1) && all(hrat > 2.0) ) {
      if ( (iter > 1) && (hrat > 2.0) ) {
         hnew = hg;
         hnew_is_ok = 1;
      }

      //printf("iter=%d, yddnrw=%e, hnew=%e, hlb=%e, hub=%e\n", iter, yddnrm, hnew, hlb, hub);

      hg = hnew;
      iter ++;
   }

   // bound and bias estimate
   *h0 = hnew * 0.5;
   *h0 = fmax(*h0, hlb);
   *h0 = fmin(*h0, hub);

   //printf("h0=%e, hlb=%e, hub=%e\n", h0, hlb, hub);

   return ierr;
}

int rk_lenrwk (__global const rk_t *rk)
{
   return (8 * rk->neq);
}
int rk_create (__global rk_t *rk, const int neq)//, const int itol, const double *rtol, const double *atol)
{
   rk->neq = neq;

   //rk->lenrwk = 8*neq;
   //rk->rwk = (double *) malloc(sizeof(double)* rk->lenrwk);

   rk->max_iters = 1000;
   rk->min_iters = 1;

   //rk->h = 0.;
   rk->h_max = 0.;
   rk->h_min = 0.;

   rk->adaption_limit = 4;

// rk->itol = itol;
   //rk->v_rtol = NULL;
   //rk->v_atol = NULL;
// assert (itol == 1);

// rk->s_rtol = *rtol;
// rk->s_atol = *atol;
   rk->itol = 1;
   rk->s_rtol = 1.0e-11;
   rk->s_atol = 1.0e-9;

   //rk->iter = 0;
   //rk->nst  = 0;
   //rk->nfe  = 0;

   return RK_SUCCESS;
}
int rk_destroy (__global rk_t *rk)
{
   //free (rk->rwk);
   //if (rk->v_rtol) free(rk->v_rtol);
   //if (rk->v_atol) free(rk->v_atol);

   return RK_SUCCESS;
}

int rk_init (__global rk_t *rk, double t0, const double t_stop)
{
   rk->t_stop = t_stop;

   rk->h_min = 0.0;
   rk->h_max = 0.0;

   const double t_dist = rk->t_stop - t0;
   rk->t_round = t_dist * rk_uround();

   if (t_dist < (rk->t_round * 2.0))
   {
      //fprintf(stderr, "error: tdist < 2.0 * tround %e\n", tdist);
      return RK_TDIST_TOO_SMALL;
   }

   if (rk->h_min < rk->t_round) rk->h_min = rk->t_round * 100.0;
   if (rk->h_max < rk->t_round) rk->h_max = t_dist / (double)rk->min_iters;

   return RK_SUCCESS;

   // Estimate the initial step size ...
   //if (rk->h < rk->h_min)
   //   ierr = rk_hin (rk, t, t_stop, &rk->h, y, func, user_data);

   //printf("hin = %e %e\n", t_stop, rk->h);

   //return ierr;
}

int rk_solve (__global const rk_t *rk, double *tcur, double *hcur, __private rk_counters_t *counters, __global double y[], __global double rwk[], RHS_Function_t func, __private void *user_data)
{
   const int neq = rk->neq;

   int ierr = RK_SUCCESS;

      //printf("h = %e %e %e\n", *hcur, rk->h_min, rk->h_max);
   // Estimate the initial step size ...
   if (*hcur < rk->h_min)
   {
      ierr = rk_hin (rk, *tcur, hcur, y, rwk, func, user_data);
      //printf("hin = %e %e %e %d\n", *hcur, rk->h_min, rk->h_max, ierr);
      if (ierr != RK_SUCCESS)
      {
         y[__getIndex(neq-1)] = *hcur;
         return ierr;
      }
   }

   #define t (*tcur)
   #define h (*hcur)
   #define nst (counters->nsteps)
   #define iter (counters->niters)

   //double t = *tcur;

   //int nst = 0, nfe = 0, iter = 0;
   nst = 0;
   //nfe = 0;
   iter = 0;

   while (fabs(t - rk->t_stop) > rk->t_round)
   {
      //const double h = *hnext;

      __global double *ytmp = rwk + __getIndex(neq*7);

      // Take a trial step over h_cur ...
      rkf45 (neq, h, y, ytmp, rwk, func, user_data);

      double herr = fmax(1.0e-20, rk_wnorm (rk, rwk, y));

      // Is there error acceptable?
      int accept = (herr <= 1.0) || (h <= rk->h_min);
      if (accept)
      {
         // update solution ...
         t += h;
         nst++;

         for (int k = 0; k < neq; k++)
            y[__getIndex(k)] = ytmp[__getIndex(k)];
      }

      double fact = sqrt( sqrt(1.0 / herr) ) * (0.840896415);

      // Restrict the rate of change in dt
      fact = fmax(fact, 1.0 / rk->adaption_limit);
      fact = fmin(fact,       rk->adaption_limit);

      //if (iter % 100 == 0)
      //   printf("iter = %d: passed=%d ... t = %e, fact = %f %f %e\n", iter, (accept ? (h <= rk->h_min ? -1 : 1) : 0), t, fact, y[__getIndex(neq-1)], h);

      // Apply grow/shrink factor for next step.
      h = h * fact;

      // Limit based on the upper/lower bounds
      h = fmin(h, rk->h_max);
      h = fmax(h, rk->h_min);

      // Stretch the final step if we're really close and we didn't just fail ...
      if (herr <= 1.0 && fabs((t + h) - rk->t_stop) < rk->h_min)
         h = rk->t_stop - t;

      // Don't overshoot the final time ...
      if (t + h > rk->t_stop)
         h = rk->t_stop - t;

      //nfe += 6;
      ++iter;
      if (rk->max_iters && iter > rk->max_iters) {
         ierr = RK_TOO_MUCH_WORK;
         //printf("(iter > max_iters)\n");
         break;
      }
   }

   //*tcur = t;

   //counters->niters = iter;
   //counters->nsteps = nst;

   #undef t
   #undef h
   #undef nst
   #undef iter

   return ierr;
}
