#ifndef __OPENCL_VERSION__

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

#endif

#include <ros.h>

#define ros_uround() ( DBL_EPSILON )

/*void ros_setewt (__global const ros_t *ros, __global const double *y, __global double *ewt)
{
   const int neq = ros->neq;
   for (int k = 0; k < neq; k++)
   {
      const double ewt_ = (ros->s_rtol * fabs(y[__getIndex(k)])) + ros->s_atol;
      ewt[k] = 1.0 / ewt_;
   }
}*/
inline double ros_getewt (__global const ros_t *ros, const int k, __global const double *y)
{
   const double ewtk = (ros->s_rtol * fabs(y[__getIndex(k)])) + ros->s_atol;
   return (1.0/ewtk);
}
inline double ros_wnorm (__global const ros_t *ros, __global const double *x, __global const double *y)
{
   const int neq = ros->neq;
   double sum = 0;
   for (int k = 0; k < neq; k++)
   {
      const double ewtk = ros_getewt(ros, k, y);
      double prod = x[__getIndex(k)] * ewtk;
      sum += (prod*prod);
   }

   return sqrt(sum / (double)neq);
}
inline void ros_dzero (const int len, __global double x[])
{
   #ifdef __INTEL_COMPILER
   #pragma ivdep
   #endif
   for (int k = 0; k < len; ++k)
      x[__getIndex(k)] = 0.0;
}
inline void ros_dcopy (const int len, const __global double src[], __global double dst[])
{
   #ifdef __INTEL_COMPILER
   #pragma ivdep
   #endif
   for (int k = 0; k < len; ++k)
      dst[__getIndex(k)] = src[__getIndex(k)];
}
/*inline void dcopy_if (const int len, const MaskType &mask, const __global double src[], __global double dst[])
{
   #ifdef __INTEL_COMPILER
   #pragma ivdep
   #endif
   for (int k = 0; k < len; ++k)
      dst[k] = if_then_else (mask, src[k], dst[k]);
}*/

inline void ros_daxpy (const int len, const double alpha, const __global double x[], __global double y[])
{
   // Alpha is scalar type ... and can be easily checked.
   if (alpha == 1.0)
   {
      #ifdef __INTEL_COMPILER
   #pragma ivdep
   #endif
      for (int k = 0; k < len; ++k)
         y[__getIndex(k)] += x[__getIndex(k)];
   }
   else if (alpha != 0.0)
   {
      #ifdef __INTEL_COMPILER
   #pragma ivdep
   #endif
      for (int k = 0; k < len; ++k)
         y[__getIndex(k)] += alpha * x[__getIndex(k)];
   }
}
/*template <typename T1, typename T2>
inline void ros_daxpy (const int& len, const T1& alpha, const T2 x[], T2 y[], vector_type)
{
   // Alpha is SIMD type -- hard to switch on value.
   #ifdef __INTEL_COMPILER
   #pragma ivdep
   #endif
   for (int k = 0; k < len; ++k)
      y[k] += (alpha * x[k]);
}*/

/*
inline int _find_pivot (const int n, const int k, __global const double *A_k, int *piv)
{
   // Find the row pivot
   *piv = k;
   for (int i = k+1; i < n; ++i)
   {
      if (fabs(A_k[i]) > fabs(A_k[*piv])) *piv = i;
   }

   // Test for singular value ...
   if (A_k[*piv] == 0.0)
      return (k+1);
   else
      return 0;
}*/
/*template <typename ValueType, typename PivotType>
inline int _find_pivot (const int &n, const int &k, ValueType *A_k, PivotType &piv, vector_type)
{
   // Make sure the simd's are equal width!
   {
      typedef typename enable_if< (ValueType::width == PivotType::width), bool>::type cond_type;
   }

   const int width = ValueType::width;

   // Find the row pivot for each element ...
   for (int elem = 0; elem < width; ++elem)
   {
      int ipiv = k;
      for (int i = k+1; i < n; ++i)
      {
         if (fabs(A_k[i][elem]) > fabs(A_k[ipiv][elem])) ipiv = i;
      }

      // Test for singular value ...
      if (A_k[ipiv][elem] == 0.0)
         return(k+1);

      piv[elem] = ipiv;
   }

   return 0;
}*/

// Pivot is a vector
/*template <typename ValueType, typename PivotType>
inline void swap (const int &k, const PivotType &piv, ValueType *A, vector_type)
{
   // Make sure the simd's are equal width!
   {
      typedef typename enable_if< (ValueType::width == PivotType::width), bool>::type cond_type;
   }

   const int width = ValueType::width;

   // Find the row pivot for each element ...
   for (int elem = 0; elem < width; ++elem)
   {
      const int ipiv = piv[elem];
      if (ipiv != k)
         std::swap(A[ipiv][elem], A[k][elem]);
   }
}

// Pivot is a scalar ... simple
template <typename ValueType, typename PivotType>
inline void swap (const int &k, const PivotType &piv, ValueType *A, scalar_type)
{
   std::swap (A[piv], A[k]);
}

template <typename ValueType, typename PivotType>
inline void swap (const int &k, const PivotType &piv, ValueType *A)
{
   swap (k, piv, A, typename is_scalar_or_vector<PivotType>::type());
}
// Pivot is a scalar ... simple
inline void _swap (const int k, const int piv, __global const double *A)
{
   const double tmp = A[piv];
   A[piv] = A[k];
   A[k] = tmp;
}*/

int ros_ludec (const int n, __global double *A, __global int *ipiv)
{
   int ierr = ROS_SUCCESS;

   /* k-th elimination step number */
   for (int k = 0; k < n; ++k)
   {
     __global double *A_k = A + __getIndex(k*n); // pointer to the column

     /* find pivot row number */
     //ipiv[k] = k;
     int pivk = k;
     for (int i = k+1; i < n; ++i)
     {
        //if (fabs(A_k[i]) > fabs(A_k[ipiv[k]])) ipiv[k] = i;
        if (fabs(A_k[__getIndex(i)]) > fabs(A_k[__getIndex(pivk)])) pivk = i;
     }
     ipiv[__getIndex(k)] = pivk;

     // Test for singular value ...
     //if (A_k[ipiv[k]] == 0.0)
     if (A_k[__getIndex(pivk)] == 0.0)
     {
        //return (k+1);
        ierr = (k+1);
        break;
     }

     //ierr = _find_pivot (n, k, A_k, ipiv[k]);
     //if (ierr) break;

     /* swap a(k,1:N) and a(piv,1:N) if necessary */
     //if (any(ipiv[k] != k))
     //if (ipiv[k] != k)
     if (pivk != k)
     {
        //swap_rows (n, k, ipiv[k], A, n);
        __global double *A_i = A; // pointer to the first column
        //for (int i = 0; i < n; ++i, A_i += n)
        for (int i = 0; i < n; ++i, A_i += __getIndex(n))
        {
           //double tmp = A_i[k];
           //A_i[k] = A_i[ipiv[k]];
           //A_i[ipiv[k]] = tmp;
           const double tmp = A_i[__getIndex(k)];
           A_i[__getIndex(k)] = A_i[__getIndex(pivk)];
           A_i[__getIndex(pivk)] = tmp;
        }
     }

     /* Scale the elements below the diagonal in
      * column k by 1.0/a(k,k). After the above swap
      * a(k,k) holds the pivot element. This scaling
      * stores the pivot row multipliers a(i,k)/a(k,k)
      * in a(i,k), i=k+1, ..., M-1.
      */
     const double mult = 1.0 / A_k[__getIndex(k)];
     for (int i = k+1; i < n; ++i)
       A_k[__getIndex(i)] *= mult;

     /* row_i = row_i - [a(i,k)/a(k,k)] row_k, i=k+1, ..., m-1 */
     /* row k is the pivot row after swapping with row l.      */
     /* The computation is done one column at a time,          */
     /* column j=k+1, ..., n-1.                                */

     for (int j = k+1; j < n; ++j)
     {
       __global double *A_j = A + __getIndex(j*n);
       const double a_kj = A_j[__getIndex(k)];

       /* a(i,j) = a(i,j) - [a(i,k)/a(k,k)]*a(k,j)  */
       /* a_kj = a(k,j), col_k[i] = - a(i,k)/a(k,k) */
       if (a_kj != 0.0) {
       //if (any(a_kj != 0.0)) {
         for (int i = k+1; i < n; ++i) {
           A_j[__getIndex(i)] -= a_kj * A_k[__getIndex(i)];
         }
       }
     }
   }

   return ierr;
   //if (ierr)
   //{
   //  fprintf(stderr,"Singular pivot j=%d\n", ierr-1);
   //  exit(-1);
   //}
}
void ros_lusol (const int n, __global double *A, __global int *ipiv, __global double *b)
{
   /* Permute b, based on pivot information in p */
   // Difficult to do with SIMDvectors ...
   for (int k = 0; k < n; ++k)
   {
     //if (any(ipiv[k] != k))
     const int pivk = ipiv[__getIndex(k)];
     //if (ipiv[k] != k)
     if (pivk != k)
     {
       //double tmp = b[k];
       //b[k] = b[ipiv[k]];
       //b[ipiv[k]] = tmp;
       double tmp = b[__getIndex(k)];
       b[__getIndex(k)] = b[__getIndex(pivk)];
       b[__getIndex(pivk)] = tmp;
     }
   }

   /* Solve Ly = b, store solution y in b */
   for (int k = 0; k < n-1; ++k)
   {
     //__global double *A_k = &A[(k*n)];
     __global double *A_k = A + __getIndex(k*n);
     //const double bk = b[k];
     const double bk = b[__getIndex(k)];
     for (int i = k+1; i < n; ++i)
       b[__getIndex(i)] -= A_k[__getIndex(i)]*bk;
       //b[i] -= A_k[i]*bk;
       //b[i] -= A_k[i]*b[k];
   }
   /* Solve Ux = y, store solution x in b */
   for (int k = n-1; k > 0; --k)
   {
     //__global double *A_k = &A[(k*n)];
     __global double *A_k = A + __getIndex(k*n);
     //b[k] /= A_k[k];
     b[__getIndex(k)] /= A_k[__getIndex(k)];
     //const double bk = b[k];
     const double bk = b[__getIndex(k)];
     for (int i = 0; i < k; ++i)
       b[__getIndex(i)] -= A_k[__getIndex(i)]*bk;
       //b[i] -= A_k[i]*bk;
       //b[i] -= A_k[i]*b[k];
   }
   //b[0] /= A[0];
   b[__getIndex(0)] /= A[__getIndex(0)];
}
int ros_hin (__global const ros_t *ros, const double t, double *h0, __global double* y, __global double *rwk, ROS_Function_t func, __private void *user_data)
{
   //value_type tround = tdist * this->uround();
   //double tdist = t_stop - t;
   //double tround = tdist * ros_uround();

   // Set lower and upper bounds on h0, and take geometric mean as first trial value.
   // Exit with this value if the bounds cross each other.

   //ros->h_min = fmax(tround * 100.0, ros->h_min);
   //ros->h_max = fmin(tdist, ros->h_max);

   const int neq = ros->neq;

   __global double *ydot  = rwk;
   __global double *y1    = ydot + __getIndex(neq);
   __global double *ydot1 = y1 + __getIndex(neq);

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

   double hlb = ros->h_min;
   double hub = ros->h_max;

   double hg = sqrt(hlb*hub);

   if (hub < hlb)
   {
      *h0 = hg;
      return ROS_SUCCESS;
   }

   // Start iteration to find solution to ... {WRMS norm of (h0^2 y'' / 2)} = 1

   const int miters = 10;
   int hnew_is_ok = 0;
   double hnew = hg;
   int iter = 0;
   int ierr = ROS_SUCCESS;

   // compute ydot at t=t0
   if (need_ydot)
   {
      //func(neq, 0.0, y, ydot, user_data);
      cklib_callback(neq, 0.0, y, ydot, user_data);
      //ros->nfe++;
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
      //func (neq, 0.0, y1, ydot1, user_data);
      cklib_callback (neq, 0.0, y1, ydot1, user_data);
      //ros->nfe++;

      // Compute WRMS norm of y''
      #ifdef __INTEL_COMPILER
      #pragma ivdep
      #endif
      for (int k = 0; k < neq; k++)
         y1[__getIndex(k)] = (ydot1[__getIndex(k)] - ydot[__getIndex(k)]) / hg;

      double yddnrm = ros_wnorm (ros, y1, y);

      //std::cout << "iter " << iter << " hg " << hg << " y'' " << yddnrm << std::endl;
      //std::cout << "ydot " << ydot[neq-1] << std::endl;

      // should we accept this?
      if (hnew_is_ok || iter == miters)
      {
         hnew = hg;
         //if (iter == miters) fprintf(stderr, "ERROR_HIN_MAX_ITERS\n");
         ierr = (hnew_is_ok) ? ROS_SUCCESS : ROS_HIN_MAX_ITERS;
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

int ros_lenrwk (__global const ros_t *ros)
{
   int lenrwk = 0;
   lenrwk +=  ros->neq;			// fy
   lenrwk +=  ros->neq;			// ynew & yerr
   lenrwk += (ros->neq * ros->neq);	// Jy
   //lenrwk +=  ros->neq;			// ewt
   lenrwk +=  ros->neq * ros->numStages;// ktmp

   return lenrwk;
}
int ros_leniwk (__global const ros_t *ros)
{
   int leniwk = ros->neq; // ipiv

   return leniwk;
}
void ros_Ros3 (__global ros_t *ros)
{
   ros->solverTag = Ros3;
   ros->numStages = 3;
   ros->ELO = 3;

   ros->A[0] = 1.0;
   ros->A[1] = 1.0;
   ros->A[2] = 0.0;

   ros->C[0] =-1.0156171083877702091975600115545;
   ros->C[1] = 4.0759956452537699824805835358067;
   ros->C[2] = 9.2076794298330791242156818474003;

   ros->newFunc[0] = 1;
   ros->newFunc[1] = 1;
   ros->newFunc[2] = 0;

   ros->M[0] = 1.0;
   ros->M[1] = 6.169794704382824559255361568973;
   ros->M[2] =-0.42772256543218573326238373806514;

   ros->E[0] = 0.5;
   ros->E[1] =-2.9079558716805469821718236208017;
   ros->E[2] = 0.22354069897811569627360909276199;

   ros->alpha[0] = 0.0;
   ros->alpha[1] = 0.43586652150845899941601945119356;
   ros->alpha[2] = 0.43586652150845899941601945119356;

   ros->gamma[0] = 0.43586652150845899941601945119356;
   ros->gamma[1] = 0.24291996454816804366592249683314;
   ros->gamma[2] = 2.1851380027664058511513169485832;
}
void ros_Rodas3 (__global ros_t *ros)
{
   ros->solverTag = Rodas3;
   ros->numStages = 4;
   ros->ELO = 3;

   ros->A[0] = 0.0;
   ros->A[1] = 2.0;
   ros->A[2] = 0.0;
   ros->A[3] = 2.0;
   ros->A[4] = 0.0;
   ros->A[5] = 1.0;

   ros->C[0] = 4.0;
   ros->C[1] = 1.0;
   ros->C[2] =-1.0;
   ros->C[3] = 1.0;
   ros->C[4] =-1.0;
   ros->C[5] =-(8.0/3.0);

   ros->newFunc[0] = 1;
   ros->newFunc[1] = 0;
   ros->newFunc[2] = 1;
   ros->newFunc[3] = 1;

   // M_i = Coefficients for new step solution
   ros->M[0] = 2.0;
   ros->M[1] = 0.0;
   ros->M[2] = 1.0;
   ros->M[3] = 1.0;

   ros->E[0] = 0.0;
   ros->E[1] = 0.0;
   ros->E[2] = 0.0;
   ros->E[3] = 1.0;

   ros->alpha[0] = 0.0;
   ros->alpha[1] = 0.0;
   ros->alpha[2] = 1.0;
   ros->alpha[3] = 1.0;

   ros->gamma[0] = 0.5;
   ros->gamma[1] = 1.5;
   ros->gamma[2] = 0.;
   ros->gamma[3] = 0.;
}
void ros_Rodas4 (__global ros_t *ros)
{
   ros->solverTag = Rodas4;
   ros->numStages = 6;
   ros->ELO = 4;

   ros->A[ 0] = 1.544;
   ros->A[ 1] = 0.9466785280815826;
   ros->A[ 2] = 0.2557011698983284;
   ros->A[ 3] = 3.314825187068521;
   ros->A[ 4] = 2.896124015972201;
   ros->A[ 5] = 0.9986419139977817;
   ros->A[ 6] = 1.221224509226641;
   ros->A[ 7] = 6.019134481288629;
   ros->A[ 8] = 12.53708332932087;
   ros->A[ 9] =-0.687886036105895;
   ros->A[10] = ros->A[6];
   ros->A[11] = ros->A[7];
   ros->A[12] = ros->A[8];
   ros->A[13] = ros->A[9];
   ros->A[14] = 1.0;

   ros->C[ 0] =-5.6688;
   ros->C[ 1] =-2.430093356833875;
   ros->C[ 2] =-0.2063599157091915;
   ros->C[ 3] =-0.1073529058151375;
   ros->C[ 4] =-0.9594562251023355e+01;
   ros->C[ 5] =-0.2047028614809616e+02;
   ros->C[ 6] = 0.7496443313967647e+01;
   ros->C[ 7] =-0.1024680431464352e+02;
   ros->C[ 8] =-0.3399990352819905e+02;
   ros->C[ 9] = 0.1170890893206160e+02;
   ros->C[10] = 0.8083246795921522e+01;
   ros->C[11] =-0.7981132988064893e+01;
   ros->C[12] =-0.3152159432874371e+02;
   ros->C[13] = 0.1631930543123136e+02;
   ros->C[14] =-0.6058818238834054e+01;

   ros->newFunc[0] = 1;
   ros->newFunc[1] = 1;
   ros->newFunc[2] = 1;
   ros->newFunc[3] = 1;
   ros->newFunc[4] = 1;
   ros->newFunc[5] = 1;

   // M_i = Coefficients for new step solution
   ros->M[0] = ros->A[6];
   ros->M[1] = ros->A[7];
   ros->M[2] = ros->A[8];
   ros->M[3] = ros->A[9];
   ros->M[4] = 1.0;
   ros->M[5] = 1.0;

   ros->E[0] = 0.0;
   ros->E[1] = 0.0;
   ros->E[2] = 0.0;
   ros->E[3] = 0.0;
   ros->E[4] = 0.0;
   ros->E[5] = 1.0;

   ros->alpha[0] = 0.0;
   ros->alpha[1] = 0.386;
   ros->alpha[2] = 0.210;
   ros->alpha[3] = 0.630;
   ros->alpha[4] = 1.0;
   ros->alpha[5] = 1.0;

   ros->gamma[0] = 0.25;
   ros->gamma[1] =-0.1043;
   ros->gamma[2] = 0.1035;
   ros->gamma[3] =-0.3620000000000023E-01;
   ros->gamma[4] = 0.0;
   ros->gamma[5] = 0.0;
}

// 4th/3rd-order L-stable Rosenbrock method with 4 stages.
// -- E. Hairer and G. Wanner, "Solving ordinary differential equations II:
//    stiff and differential-algebraic problems," Springer series in
//    computational mathematics, Springer-Verlag (1990).
void ros_Ros4 (__global ros_t *ros)
{
   ros->solverTag = Ros4;
   ros->numStages = 4;
   ros->ELO = 4;

   // A and C are strictly lower-triangular matrices in row-major order!!!!
   // -- A(i,j) = [(i)*(i-1)/2 + j] ... A(1,0) = A[0], A(2,0) = A[1]
   ros->A[0] = 2.0;
   ros->A[1] = 1.867943637803922;
   ros->A[2] = 0.2344449711399156;
   ros->A[3] = ros->A[1];
   ros->A[4] = ros->A[2];
   ros->A[5] = 0.0;

   ros->C[0] =-7.137615036412310;
   ros->C[1] = 2.580708087951457;
   ros->C[2] = 0.6515950076447975;
   ros->C[3] =-2.137148994382534;
   ros->C[4] =-0.3214669691237626;
   ros->C[5] =-0.6949742501781779;

   // Does the stage[i] need a new function eval or can it reuse the
   // prior one from stage[i-1]?
   ros->newFunc[0] = 1;
   ros->newFunc[1] = 1;
   ros->newFunc[2] = 1;
   ros->newFunc[3] = 0;

   // M_i = Coefficients for new step solution
   ros->M[0] = 2.255570073418735;
   ros->M[1] = 0.2870493262186792;
   ros->M[2] = 0.4353179431840180;
   ros->M[3] = 1.093502252409163;

   // E_i = Coefficients for error estimator
   ros->E[0] =-0.2815431932141155;
   ros->E[1] =-0.07276199124938920;
   ros->E[2] =-0.1082196201495311;
   ros->E[3] =-1.093502252409163;

   // Y( T + h*alpha_i )
   ros->alpha[0] = 0.0;
   ros->alpha[1] = 1.14564;
   ros->alpha[2] = 0.65521686381559;
   ros->alpha[3] = ros->alpha[2];

   // gamma_i = \Sum_j  gamma_{i,j}
   ros->gamma[0] = 0.57282;
   ros->gamma[1] =-1.769193891319233;
   ros->gamma[2] = 0.7592633437920482;
   ros->gamma[3] =-0.104902108710045;
}
int ros_create (__global ros_t *ros, const int neq, ros_solverTags solver_tag)//, const int itol, const double *rtol, const double *atol)
{
   ros->neq = neq;

   //ros->lenrwk = 8*neq;
   //ros->rwk = (double *) malloc(sizeof(double)* ros->lenrwk);

   ros->max_iters = 0;
   ros->min_iters = 1;

   //ros->h = 0.;
   ros->h_max = 0.;
   ros->h_min = 0.;

   ros->adaption_limit = 5;

// ros->itol = itol;
   //ros->v_rtol = NULL;
   //ros->v_atol = NULL;
// assert (itol == 1);

// ros->s_rtol = *rtol;
// ros->s_atol = *atol;
   ros->itol = 1;
   ros->s_rtol = 1.0e-11;
   ros->s_atol = 1.0e-9;

   //ros->iter = 0;
   //ros->nst  = 0;
   //ros->nfe  = 0;

   //ros_Rodas4(ros); // Default solver settings ...
   if (solver_tag == Ros3)
      ros_Ros3(ros);
   else if (solver_tag == Rodas3)
      ros_Rodas3(ros);
   else if (solver_tag == Ros4)
      ros_Ros4(ros);
   else if (solver_tag == Rodas4)
      ros_Rodas4(ros);
   else
   {
      //fprintf(stderr,"Invalid solver_tag = %d\n", solver_tag);
      ros_Rodas4(ros);
      return ROS_ERROR;
   }

   return ROS_SUCCESS;
}
int ros_destroy (__global ros_t *ros)
{
   //free (ros->rwk);
   //if (ros->v_rtol) free(ros->v_rtol);
   //if (ros->v_atol) free(ros->v_atol);

   return ROS_SUCCESS;
}

int ros_init (__global ros_t *ros, double t0, const double t_stop)
{
   ros->t_stop = t_stop;

   ros->h_min = 0.0;
   ros->h_max = 0.0;

   const double t_dist = ros->t_stop - t0;
   ros->t_round = t_dist * ros_uround();

   if (t_dist < (ros->t_round * 2.0))
   {
      //fprintf(stderr, "error: tdist < 2.0 * tround %e\n", tdist);
      return ROS_TDIST_TOO_SMALL;
   }

   if (ros->h_min < ros->t_round) ros->h_min = ros->t_round * 100.0;
   if (ros->h_max < ros->t_round) ros->h_max = t_dist / (double)ros->min_iters;

   return ROS_SUCCESS;

   // Estimate the initial step size ...
   //if (ros->h < ros->h_min)
   //   ierr = ros_hin (ros, t, t_stop, &ros->h, y, func, user_data);

   //printf("hin = %e %e\n", t_stop, ros->h);

   //return ierr;
}
void ros_fdjac (__global const ros_t *ros, const double tcur, const double hcur, __global double *y, __global double *fy, __global double *Jy, ROS_Function_t func, __private void *user_data)
{
   const int neq = ros->neq;

   // Norm of fy(t) ...
   double fnorm = ros_wnorm( ros, fy, y );

   // Safety factors ...
   const double sround = sqrt( ros_uround() );
   double r0 = (1000. * ros_uround() * neq) * (hcur * fnorm);
   if (r0 == 0.) r0 = 1.;

   // Build each column vector ...
   for (int j = 0; j < neq; ++j)
   {
      const double ysav = y[__getIndex(j)];
      const double ewtj = ros_getewt(ros, j, y);
      const double dely = fmax( sround * fabs(ysav), r0 / ewtj );
      y[__getIndex(j)] += dely;

      __global double *jcol = &Jy[__getIndex(j*neq)];

      //func (neq, tcur, y, jcol, user_data);
      cklib_callback (neq, tcur, y, jcol, user_data);

      const double delyi = 1. / dely;
      for (int i = 0; i < neq; ++i)
         jcol[__getIndex(i)] = (jcol[__getIndex(i)] - fy[__getIndex(i)]) * delyi;

      y[__getIndex(j)] = ysav;
   }
}

int ros_solve (__global const ros_t *ros, double *tcur, double *hcur, __private ros_counters_t *counters, __global double y[], __global int iwk[], __global double rwk[], ROS_Function_t func, ROS_Jacobian_t jac, __private void *user_data)
{
   int ierr = ROS_SUCCESS;

   #define nst (counters->nst)
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
   if (h < ros->h_min)
   {
      ierr = ros_hin (ros, t, hcur, y, rwk, func, user_data);
      if (ierr != ROS_SUCCESS)
         return ierr;
   }
      //printf("hin = %e %e %e %f\n", h, ros->h_min, ros->h_max, y[__getIndex(neq-1)]);

   // Zero the counters ...
   nst = 0;
   nfe = 0;
   nlu = 0;
   nje = 0;
   iter = 0;

   // Set the work arrays ...
   __global double *fy   = rwk;
   __global double *ynew = fy + __getIndex(neq);
   __global double *Jy   = ynew + __getIndex(neq);
   __global double *ktmp = Jy + __getIndex(neq*neq);
   __global double *yerr = ynew;
   //__global double *ewt  = &Jy[neq*neq];

   while (fabs(t - ros->t_stop) > ros->t_round)
   {
      // Set the error weight array.
      //ros_setewt (ros, y, ewt);

      // Compute the RHS and Jacobian matrix.
      //func (neq, t, y, fy, user_data);
      cklib_callback (neq, t, y, fy, user_data);
      nfe++;

      //if (jac == NULL)
      {
         ros_fdjac (ros, t, h, y, fy, Jy, func, user_data);
         nfe += neq;
      }
      //else
      //{
      //   jac (neq, t, y, Jy, user_data);
      //}

      nje++;

      // Construct iteration matrix J' := 1/(gamma*h) - J
      {
         const double one_hgamma = 1.0 / (h * ros->gamma[0]);

         for (int j = 0; j < neq; ++j)
         {
            __global double *jcol = &Jy[__getIndex(j*neq)];
            for (int i = 0; i < neq; ++i)
               jcol[__getIndex(i)] = -jcol[__getIndex(i)];

            jcol[__getIndex(j)] += one_hgamma;
         }
      }

      // Factorization J'
      ros_ludec(neq, Jy, iwk);
      nlu++;

      for (int s = 0; s < ros->numStages; s++)
      {
         // Compute the function at this stage ...
         if (s == 0)
         {
            //func (neq, y, fy.getPointer());
         }
         else if (ros->newFunc[s])
         {
            ros_dcopy (neq, y, ynew);

            for (int j = 0; j < s; ++j)
            {
               const double Asj = A(s,j);
               //printf("Asj = %f %d %d\n", Asj, s, j);
               __global double *k_j = &ktmp[__getIndex(j*neq)];

               ros_daxpy (neq, Asj, k_j, ynew);
            }

            //func (neq, t, ynew, fy, user_data);
            cklib_callback (neq, t, ynew, fy, user_data);
            nfe++;

            //printf("newF=%d\n", s);
            //for (int k = 0; k < neq; ++k)
            //   printf("ynew[%d] = %e %e\n", k, ynew[k], fy[k]);
         }

         //printf("stage=%d\n", s);
         //for (int k = 0; k < neq; ++k)
         //   printf("fy[%d] = %e\n", k, fy[k]);

         // Build the sub-space vector K
         __global double *k_s = &ktmp[__getIndex(s*neq)];
         ros_dcopy (neq, fy, k_s);

         for (int j = 0; j < s; j++)
         {
            const double hCsj = C(s,j) / h;
            //printf("C/h = %f %d %d\n", hCsj, s, j);

            __global double *k_j = &ktmp[__getIndex(j*neq)];
            ros_daxpy (neq, hCsj, k_j, k_s);
         }

         //printf("k before=%d\n", s);
         //for (int k = 0; k < neq; ++k)
         //   printf("k[%d] = %e\n", k, ks[k]);

         // Solve the current stage ..
         ros_lusol (neq, Jy, iwk, k_s);

         //printf("k after=%d\n", s);
         //for (int k = 0; k < neq; ++k)
         //   printf("k[%d] = %e\n", k, ks[k]);

      }

      // Compute the error estimation of the trial solution
      ros_dzero (neq, yerr);

      for (int j = 0; j < ros->numStages; ++j)
      {
         __global double *k_j = &ktmp[__getIndex(j*neq)];
         ros_daxpy (neq, ros->E[j], k_j, yerr);
      }

      double herr = fmax(1.0e-20, ros_wnorm (ros, yerr, y));

      // Is there error acceptable?
      int accept = (herr <= 1.0) || (h <= ros->h_min);
      if (accept)
      {
         // Actually compute the new solution ... delayed from above.
         ros_dcopy (neq, y, ynew);
         for (int j = 0; j < ros->numStages; ++j)
         {
            __global double *k_j = &ktmp[__getIndex(j*neq)];
            ros_daxpy (neq, ros->M[j], k_j, ynew);
         }

         // update solution ...
         t += h;
         nst++;

         for (int k = 0; k < neq; k++)
            y[__getIndex(k)] = ynew[__getIndex(k)];
      }

      double fact = 0.9 * pow( 1.0 / herr, (1.0/ros->ELO));

      // Restrict the rate of change in dt
      fact = fmax(fact, 1.0 / ros->adaption_limit);
      fact = fmin(fact,       ros->adaption_limit);

#ifdef VERBOSE
      if (iter % (VERBOSE) == 0)
         printf("iter = %d: passed=%d ... t = %e, fact = %f %f %e\n", iter, (accept ? (h <= ros->h_min ? -1 : 1) : 0), t, fact, y[__getIndex(neq-1)], h);
#endif

      // Apply grow/shrink factor for next step.
      h = h * fact;

      // Limit based on the upper/lower bounds
      h = fmin(h, ros->h_max);
      h = fmax(h, ros->h_min);

      // Stretch the final step if we're really close and we didn't just fail ...
      if (herr <= 1.0 && fabs((t + h) - ros->t_stop) < ros->h_min)
         h = ros->t_stop - t;

      // Don't overshoot the final time ...
      if (t + h > ros->t_stop)
         h = ros->t_stop - t;

      ++iter;
      if (ros->max_iters && iter > ros->max_iters) {
         ierr = ROS_TOO_MUCH_WORK;
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
