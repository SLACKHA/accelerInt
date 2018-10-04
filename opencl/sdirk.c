#ifndef __OPENCL_VERSION__

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

#endif

#include <sdirk.h>

#define sdirk_uround() ( DBL_EPSILON )

/*void sdirk_setewt (__global const sdirk_t *sdirk, __global const double *y, __global double *ewt)
{
   const int neq = sdirk->neq;
   for (int k = 0; k < neq; k++)
   {
      const double ewt_ = (sdirk->s_rtol * fabs(y[__getIndex(k)])) + sdirk->s_atol;
      ewt[k] = 1.0 / ewt_;
   }
}*/
inline double sdirk_getewt (__global const sdirk_t *sdirk, const int k, __global const double *y)
{
   const double ewtk = (sdirk->s_rtol * fabs(y[__getIndex(k)])) + sdirk->s_atol;
   return (1.0/ewtk);
}
inline double sdirk_wnorm (__global const sdirk_t *sdirk, __global const double *x, __global const double *y)
{
   const int neq = sdirk->neq;
   double sum = 0;
   for (int k = 0; k < neq; k++)
   {
      const double ewtk = sdirk_getewt(sdirk, k, y);
      double prod = x[__getIndex(k)] * ewtk;
      sum += (prod*prod);
   }

   return sqrt(sum / (double)neq);
}
inline void sdirk_dzero (const int len, __global double x[])
{
   #ifdef __INTEL_COMPILER
   #pragma ivdep
   #endif
   for (int k = 0; k < len; ++k)
      x[__getIndex(k)] = 0.0;
}
inline void sdirk_dcopy (const int len, const __global double src[], __global double dst[])
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

inline void sdirk_dscal (const int len, const double alpha, __global double y[])
{
   // Alpha is scalar type ... and can be easily checked.
   if (alpha == 1.0)
      return;
   else
   {
      #ifdef __INTEL_COMPILER
   #pragma ivdep
   #endif
      for (int k = 0; k < len; ++k)
         y[__getIndex(k)] = alpha * y[__getIndex(k)];
   }
}
inline void sdirk_daxpy (const int len, const double alpha, const __global double x[], __global double y[])
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
inline void sdirk_daxpy (const int& len, const T1& alpha, const T2 x[], T2 y[], vector_type)
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

int sdirk_ludec (const int n, __global double *A, __global int *ipiv)
{
   int ierr = SDIRK_SUCCESS;

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
void sdirk_lusol (const int n, __global double *A, __global int *ipiv, __global double *b)
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
int sdirk_hin (__global const sdirk_t *sdirk, const double t, double *h0, __global double* y, __global double *rwk, SDIRK_Function_t func, __private void *user_data)
{
   //value_type tround = tdist * this->uround();
   //double tdist = t_stop - t;
   //double tround = tdist * sdirk_uround();

   // Set lower and upper bounds on h0, and take geometric mean as first trial value.
   // Exit with this value if the bounds cross each other.

   //sdirk->h_min = fmax(tround * 100.0, sdirk->h_min);
   //sdirk->h_max = fmin(tdist, sdirk->h_max);

   const int neq = sdirk->neq;

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

   double hlb = sdirk->h_min;
   double hub = sdirk->h_max;

   double hg = sqrt(hlb*hub);

   if (hub < hlb)
   {
      *h0 = hg;
      return SDIRK_SUCCESS;
   }

   // Start iteration to find solution to ... {WRMS norm of (h0^2 y'' / 2)} = 1

   const int miters = 10;
   int hnew_is_ok = 0;
   double hnew = hg;
   int iter = 0;
   int ierr = SDIRK_SUCCESS;

   // compute ydot at t=t0
   if (need_ydot)
   {
      if (func)
         func(neq, 0.0, y, ydot, user_data);
      else
         cklib_callback(neq, 0.0, y, ydot, user_data);
      //sdirk->nfe++;
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
      if (func)
         func (neq, 0.0, y1, ydot1, user_data);
      else
         cklib_callback (neq, 0.0, y1, ydot1, user_data);
      //sdirk->nfe++;

      // Compute WRMS norm of y''
      #ifdef __INTEL_COMPILER
      #pragma ivdep
      #endif
      for (int k = 0; k < neq; k++)
         y1[__getIndex(k)] = (ydot1[__getIndex(k)] - ydot[__getIndex(k)]) / hg;

      double yddnrm = sdirk_wnorm (sdirk, y1, y);

      //std::cout << "iter " << iter << " hg " << hg << " y'' " << yddnrm << std::endl;
      //std::cout << "ydot " << ydot[neq-1] << std::endl;

      // should we accept this?
      if (hnew_is_ok || iter == miters)
      {
         hnew = hg;
         //if (iter == miters) fprintf(stderr, "ERROR_HIN_MAX_ITERS\n");
         ierr = (hnew_is_ok) ? SDIRK_SUCCESS : SDIRK_HIN_MAX_ITERS;
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

int sdirk_lenrwk (__global const sdirk_t *sdirk)
{
   int lenrwk = 0;
   lenrwk +=  sdirk->neq;			// fy
   lenrwk +=  sdirk->neq;			// del & yerr
   lenrwk += (sdirk->neq * sdirk->neq);		// Jy
   lenrwk += (sdirk->neq * sdirk->neq);		// M
   lenrwk += (sdirk->neq * sdirk->numStages);	// z
   lenrwk +=  sdirk->neq;			// g

   return lenrwk;
}
int sdirk_leniwk (__global const sdirk_t *sdirk)
{
   int leniwk = sdirk->neq; // ipiv

   return leniwk;
}

#define __matrix_index(_i,_j,_var) (sdirk->_var[(_i)][(_j)])
#define A_(_i,_j)     ( __matrix_index(_i,_j,A) )
#define Theta_(_i,_j) ( __matrix_index(_i,_j,Theta) )
#define Alpha_(_i,_j) ( __matrix_index(_i,_j,Alpha) )

// 4th/3rd-order L-stable SDIRK method with 5 stages.
// -- E. Hairer and G. Wanner, "Solving ordinary differential equations II:
//    stiff and differential-algebraic problems," Springer series in
//    computational mathematics, Springer-Verlag (1990).
void sdirk_S4a (__global sdirk_t *sdirk)
{
   sdirk->solverTag = S4a;
   sdirk->numStages = 5;
   sdirk->ELO = 4;

   // Constant diagonal
   sdirk->gamma = 8.0 / 30.0; //0.2666666666666666666666666666666667;

   // A and C are lower-triangular matrices in column-major order!!!!
   // -- A(i,j) = [(i)*(i+1)/2 + j] ... A(1,0) = A[0], A(2,0) = A[1]
   for (int i = 0; i < sdirk_maxStages; ++i)
      for (int j = 0; j < sdirk_maxStages; ++j)
         A_(i,j) = 0.0;
   A_(0,0) = sdirk->gamma;
   A_(1,0) = 0.5;
   A_(1,1) = sdirk->gamma;
   A_(2,0) = 0.3541539528432732316227461858529820;
   A_(2,1) =-0.05415395284327323162274618585298197;
   A_(2,2) = sdirk->gamma;
   A_(3,0) = 0.08515494131138652076337791881433756;
   A_(3,1) =-0.06484332287891555171683963466229754;
   A_(3,2) = 0.07915325296404206392428857585141242;
   A_(3,3) = sdirk->gamma;
   A_(4,0) = 2.100115700566932777970612055999074;
   A_(4,1) =-0.7677800284445976813343102185062276;
   A_(4,2) = 2.399816361080026398094746205273880;
   A_(4,3) =-2.998818699869028161397714709433394;
   A_(4,4) = sdirk->gamma;

   sdirk->B[0]    = 2.100115700566932777970612055999074;
   sdirk->B[1]    =-0.7677800284445976813343102185062276;
   sdirk->B[2]    = 2.399816361080026398094746205273880;
   sdirk->B[3]    =-2.998818699869028161397714709433394;
   sdirk->B[4]    = sdirk->gamma;

   sdirk->Bhat[0] = 2.885264204387193942183851612883390;
   sdirk->Bhat[1] =-0.1458793482962771337341223443218041;
   sdirk->Bhat[2] = 2.390008682465139866479830743628554;
   sdirk->Bhat[3] =-4.129393538556056674929560012190140;
   sdirk->Bhat[4] = 0.;

   sdirk->C[0]    = 8.0  / 30.0; //0.2666666666666666666666666666666667;
   sdirk->C[1]    = 23.0 / 30.0; // 0.7666666666666666666666666666666667;
   sdirk->C[2]    = 17.0 / 30.0; // 0.5666666666666666666666666666666667;
   sdirk->C[3]    = 0.3661315380631796996374935266701191;
   sdirk->C[4]    = 1.;

   // Ynew = Yold + h*Sum_i {rkB_i*k_i} = Yold + Sum_i {rkD_i*Z_i}
   sdirk->D[0] = 0.;
   sdirk->D[1] = 0.;
   sdirk->D[2] = 0.;
   sdirk->D[3] = 0.;
   sdirk->D[4] = 1.;

   // Err = h * Sum_i {(rkB_i-rkBhat_i)*k_i} = Sum_i {rkE_i*Z_i}
   sdirk->E[0] =-0.6804000050475287124787034884002302;
   sdirk->E[1] = 1.558961944525217193393931795738823;
   sdirk->E[2] =-13.55893003128907927748632408763868;
   sdirk->E[3] = 15.48522576958521253098585004571302;
   sdirk->E[4] = 1.;

   // h*Sum_j {rkA_ij*k_j} = Sum_j {rkTheta_ij*Z_j}
   for (int i = 0; i < sdirk_maxStages; ++i)
      for (int j = 0; j < sdirk_maxStages; ++j)
         Theta_(i,j) = 0.0;
   Theta_(1,0) = 1.875;
   Theta_(2,0) = 1.708847304091539528432732316227462;
   Theta_(2,1) =-0.2030773231622746185852981969486824;
   Theta_(3,0) = 0.2680325578937783958847157206823118;
   Theta_(3,1) =-0.1828840955527181631794050728644549;
   Theta_(3,2) = 0.2968246986151577397160821594427966;
   Theta_(4,0) = 0.9096171815241460655379433581446771;
   Theta_(4,1) =-3.108254967778352416114774430509465;
   Theta_(4,2) = 12.33727431701306195581826123274001;
   Theta_(4,3) =-11.24557012450885560524143016037523;

   // Starting value for Newton iterations: Z_i^0 = Sum_j {rkAlpha_ij*Z_j}
   for (int i = 0; i < sdirk_maxStages; ++i)
      for (int j = 0; j < sdirk_maxStages; ++j)
         Alpha_(i,j) = 0.0;
   Alpha_(1,0) = 2.875000000000000000000000000000000;
   Alpha_(2,0) = 0.8500000000000000000000000000000000;
   Alpha_(2,1) = 0.4434782608695652173913043478260870;
   Alpha_(3,0) = 0.7352046091658870564637910527807370;
   Alpha_(3,1) =-0.09525565003057343527941920657462074;
   Alpha_(3,2) = 0.4290111305453813852259481840631738;
   Alpha_(4,0) =-16.10898993405067684831655675112808;
   Alpha_(4,1) = 6.559571569643355712998131800797873;
   Alpha_(4,2) =-15.90772144271326504260996815012482;
   Alpha_(4,3) = 25.34908987169226073668861694892683;
}
int sdirk_create (__global sdirk_t *sdirk, const int neq, sdirk_solverTags solver_tag)//, const int itol, const double *rtol, const double *atol)
{
   sdirk->neq = neq;

   //sdirk->lenrwk = 8*neq;
   //sdirk->rwk = (double *) aligned_malloc(sizeof(double)* sdirk->lenrwk);

   sdirk->max_iters = 10000;
   sdirk->min_iters = 1;

   //sdirk->h = 0.;
   sdirk->h_max = 0.;
   sdirk->h_min = 0.;

   sdirk->adaption_limit = 5;

// sdirk->itol = itol;
   //sdirk->v_rtol = NULL;
   //sdirk->v_atol = NULL;
// assert (itol == 1);

// sdirk->s_rtol = *rtol;
// sdirk->s_atol = *atol;
   sdirk->itol = 1;
   sdirk->s_rtol = 1.0e-11;
   sdirk->s_atol = 1.0e-9;

   //sdirk->iter = 0;
   //sdirk->nst  = 0;
   //sdirk->nfe  = 0;

   if (solver_tag == S4a)
      sdirk_S4a(sdirk);
   else
   {
      fprintf(stderr,"Invalid solver_tag = %d, using default %d\n", solver_tag, S4a);
      sdirk_S4a(sdirk);
      return SDIRK_ERROR;
   }

   return SDIRK_SUCCESS;
}
int sdirk_destroy (__global sdirk_t *sdirk)
{
   //free (sdirk->rwk);
   //if (sdirk->v_rtol) free(sdirk->v_rtol);
   //if (sdirk->v_atol) free(sdirk->v_atol);

   return SDIRK_SUCCESS;
}

int sdirk_init (__global sdirk_t *sdirk, double t0, const double t_stop)
{
   sdirk->t_stop = t_stop;

   sdirk->h_min = 0.0;
   sdirk->h_max = 0.0;

   const double t_dist = sdirk->t_stop - t0;
   sdirk->t_round = t_dist * sdirk_uround();

   if (t_dist < (sdirk->t_round * 2.0))
   {
      //fprintf(stderr, "error: tdist < 2.0 * tround %e\n", tdist);
      return SDIRK_TDIST_TOO_SMALL;
   }

   if (sdirk->h_min < sdirk->t_round) sdirk->h_min = sdirk->t_round * 100.0;
   if (sdirk->h_max < sdirk->t_round) sdirk->h_max = t_dist / (double)sdirk->min_iters;

   //sdirk->NewtonThetaMin = 0.001;
   //sdirk->NewtonTolerance = 0.03;

   return SDIRK_SUCCESS;

   // Estimate the initial step size ...
   //if (sdirk->h < sdirk->h_min)
   //   ierr = sdirk_hin (sdirk, t, t_stop, &sdirk->h, y, func, user_data);

   //printf("hin = %e %e\n", t_stop, sdirk->h);

   //return ierr;
}
void sdirk_fdjac (__global const sdirk_t *sdirk, const double tcur, const double hcur, __global double *y, __global double *fy, __global double *Jy, SDIRK_Function_t func, __private void *user_data)
{
   const int neq = sdirk->neq;

   // Norm of fy(t) ...
   double fnorm = sdirk_wnorm( sdirk, fy, y );

   // Safety factors ...
   const double sround = sqrt( sdirk_uround() );
   double r0 = (1000. * sdirk_uround() * neq) * (hcur * fnorm);
   if (r0 == 0.) r0 = 1.;

   // Build each column vector ...
   for (int j = 0; j < neq; ++j)
   {
      const double ysav = y[__getIndex(j)];
      const double ewtj = sdirk_getewt(sdirk, j, y);
      const double dely = fmax( sround * fabs(ysav), r0 / ewtj );
      y[__getIndex(j)] += dely;

      __global double *jcol = &Jy[__getIndex(j*neq)];

      if (func)
         func (neq, tcur, y, jcol, user_data);
      else
         cklib_callback (neq, tcur, y, jcol, user_data);

      const double delyi = 1. / dely;
      for (int i = 0; i < neq; ++i)
         jcol[__getIndex(i)] = (jcol[__getIndex(i)] - fy[__getIndex(i)]) * delyi;

      y[__getIndex(j)] = ysav;
   }
}

int sdirk_solve (__global const sdirk_t *sdirk, double *tcur, double *hcur, __private sdirk_counters_t *counters, __global double y[], __global int iwk[], __global double rwk[], SDIRK_Function_t func, SDIRK_Jacobian_t jac, __private void *user_data)
{
   int ierr = SDIRK_SUCCESS;

   #define nst (counters->nst)
   #define nfe (counters->nfe)
   #define nje (counters->nje)
   #define nlu (counters->nlu)
   #define nni (counters->nni)
   #define iter (counters->niters)
   #define h (*hcur)
   #define t (*tcur)
   #define neq (sdirk->neq)
   //#define InterpolateNewton	(0)	// Start at zero (0) or interpolate a starting guess (1)
   #define InterpolateNewton	(1)	// Start at zero (0) or interpolate a starting guess (1)
   #define MaxNewtonIterations	(8)	// Max number of newton iterations
   #define NewtonThetaMin	(0.005)	// Minimum convergence rate for the Newton Iteration (0.001)
   #define NewtonThetaMax	(0.999)	// Maximum residual drop acceptable
   #define NewtonTolerance	(0.03)	// Convergence criteria
   #define Qmax (1.2)			// Max h-adaption to recycle M
   #define Qmin (1.)			// Min ""

      //printf("h = %e %e %e %f\n", h, sdirk->h_min, sdirk->h_max, y[__getIndex(neq-1)]);
   // Estimate the initial step size ...
   if (h < sdirk->h_min)
   {
      ierr = sdirk_hin (sdirk, t, hcur, y, rwk, func, user_data);
      if (ierr != SDIRK_SUCCESS)
         return ierr;
   }
      //printf("hin = %e %e %e %f\n", h, sdirk->h_min, sdirk->h_max, y[__getIndex(neq-1)]);

   // Zero the counters ...
   nst = 0;
   nfe = 0;
   nlu = 0;
   nje = 0;
   nni = 0;
   iter = 0;

   // Set the work arrays ...
   __global double *fy   = rwk;
   __global double *del  = fy + __getIndex(neq);
   __global double *Jy   = del + __getIndex(neq);
   __global double *M    = Jy + __getIndex(neq*neq);
   __global double *z    = M + __getIndex(neq*neq);
   __global double *g    = z + __getIndex(neq*sdirk->numStages);
   __global double *yerr = del;
   //__global double *ewt  = &Jy[neq*neq];

   int ComputeJ = 1;
   int ComputeM = 1;

   while (fabs(t - sdirk->t_stop) > sdirk->t_round)
   {
      // Set the error weight array.
      //sdirk_setewt (sdirk, y, ewt);

      // Construct the Iteration matrix ... if needed.
      if (ComputeM)
      {
         // Compute the Jacobian matrix or recycle an old one.
         if (ComputeJ)
         {
            if (jac)
               jac (neq, t, y, Jy, user_data);
            else
            {
               // Compute the RHS ... the fd algorithm expects it.
               if (func)
                  func (neq, t, y, fy, user_data);
               else
                  cklib_callback (neq, t, y, fy, user_data);
               nfe++;

               sdirk_fdjac (sdirk, t, h, y, fy, Jy, func, user_data);
               nfe += neq;
            }

            nje++;
         }

         // Compute M := 1/(gamma*h) - J
         const double one_hgamma = 1.0 / (h * sdirk->gamma);

         for (int j = 0; j < neq; ++j)
         {
            __global double *Mcol = &M[__getIndex(j*neq)];
            __global double *Jcol = &Jy[__getIndex(j*neq)];
            for (int i = 0; i < neq; ++i)
               Mcol[__getIndex(i)] = -Jcol[__getIndex(i)];

            Mcol[__getIndex(j)] += one_hgamma;
         }

         // Factorization M
         sdirk_ludec(neq, M, iwk);
         nlu++;
      }

      int Accepted;
      double HScalingFactor;
      double NewtonTheta = NewtonThetaMin;

      for (int s = 0; s < sdirk->numStages; s++)
      {
         // Initial the RK stage vectors Z_i and G.
         __global double *z_s = &z[__getIndex(s*neq)];
         sdirk_dzero (neq, z_s);
         sdirk_dzero (neq, g);

         if (s)
         {
            for (int j = 0; j < s; ++j)
            {
               // G = \Sum_j Theta_i,j*Z_j = h * \Sum_j A_i,j*F(Z_j)
               __global double *z_j = &z[__getIndex(j*neq)];
               sdirk_daxpy (neq, Theta_(s,j), z_j, g);

               // Z_i = \Sum_j Alpha(i,j)*Z_j
               if (InterpolateNewton)
                  sdirk_daxpy (neq, Alpha_(s,j), z_j, z_s);
            }
         }

         // Solve the non-linear problem with the Newton-Raphson method.
         Accepted = 0;
         HScalingFactor = 0.8;
         NewtonTheta = NewtonThetaMin;
         double NewtonResidual;

         for (int NewtonIter = 0; NewtonIter < MaxNewtonIterations && !(Accepted); NewtonIter++, nni++)
         {
            // 1) Build the RHS of the root equation: F := G + (h*gamma)*f(y+Z_s) - Z_s
               for (int k = 0; k < neq; ++k)
                  del[__getIndex(k)] = y[__getIndex(k)] + z_s[__getIndex(k)];

               if (func)
                  func (neq, t, del, fy, user_data);
               else
                  cklib_callback (neq, t, del, fy, user_data);
               nfe++;

               const double hgamma = h*sdirk->gamma;
               for (int k = 0; k < neq; ++k)
                  del[__getIndex(k)] = g[__getIndex(k)] + hgamma * fy[__getIndex(k)] - z_s[__getIndex(k)];
                //del[__getIndex(k)] = g[__getIndex(k)] - z_s[__getIndex(k)] + hgamma * fy[__getIndex(k)];

            // 2) Solve the linear problem: M*delz = (1/hgamma)F ... so scale F first.
            sdirk_dscal (neq, 1.0/hgamma, del);
            sdirk_lusol (neq, M, iwk, del);

            // 3) Check the convergence and convergence rate
            // 3.1) Compute the norm of the correction.
            double dnorm = sdirk_wnorm (sdirk, del, y);
            // 3.2) If not the first iteration, estimate the rate.
            if (NewtonIter > 1)
            {
               NewtonTheta = dnorm / NewtonResidual;
               if (NewtonTheta < NewtonThetaMax)
               {
                  double ConvergenceRate = NewtonTheta / (1.0 - NewtonTheta);
                  Accepted = (ConvergenceRate * dnorm < NewtonTolerance);
                  //printf("miter=%d, norm=%e, rate=%f\n", NewtonIter, dnorm, ConvergenceRate);

                  // Predict the error after the maximum # of iterations.
                  double PredictedError = dnorm * pow(NewtonTheta, (MaxNewtonIterations-NewtonIter)/(1.0-NewtonTheta));
                  if (PredictedError > NewtonTolerance)
                  {
                     // Error is probably too large, shrink h and try again.
                     double QNewton = fmin(10.0, PredictedError / NewtonTolerance);
                     HScalingFactor = 0.9 * pow( QNewton, -1.0 / (1.0 + MaxNewtonIterations - NewtonIter));
                     fprintf(stderr,"PredictedError > NewtonTolerance %e %f %f %d %d %d %d\n", h, HScalingFactor, PredictedError, nst, iter, s, NewtonIter);
                     break;
                  }
               }
               else
               {
                  //fprintf(stderr,"NewtonTheta >= NewtonThetaMax %e %e %d %d %d %d\n", h, NewtonTheta, nst, iter, s, NewtonIter);
                  break;
               }
            }

            // Save the residual norm for the next iteration.
            NewtonResidual = dnorm;

            // 4) Update the solution: Z_s <- Z_s + delta
            sdirk_daxpy (neq, 1.0, del, z_s);
         }

         if (!Accepted)
         {
            //printf("failed to converge %d %d.\n", iter, s);
            ComputeJ = 0; // Jacobian is still valid
            ComputeM = 1; // M is invalid since h will change (perhaps) drastically.
            break;
            //return 0;
         }

      } // ... stages

      if (Accepted)
      {
         // Compute the error estimation of the trial solution.
         sdirk_dzero (neq, yerr);
         for (int j = 0; j < sdirk->numStages; ++j)
         {
            __global double *z_j = &z[__getIndex(j*neq)];
            if (sdirk->E[j] != 0.0)
               sdirk_daxpy (neq, sdirk->E[j], z_j, yerr);
         }

         //const double hgamma = h*sdirk->gamma;
         //sdirk_dscal (neq, 1.0/hgamma, yerr);
         //sdirk_lusol (neq, M, iwk, yerr);

         double herr = fmax(1.0e-20, sdirk_wnorm (sdirk, yerr, y));

         // Is there error acceptable?
         //int accept = (herr <= 1.0) || (h <= sdirk->h_min);
         //if (accept)
         Accepted = (herr <= 1.0) || (h <= sdirk->h_min);
         if (Accepted)
         {
            // If stiffly-accurate, Z_s with s := numStages, is the solution.
            // Else, sum the stage solutions: y_n+1 <- y_n + \Sum_j D_j * Z_j
            for (int j = 0; j < sdirk->numStages; ++j)
            {
               __global double *z_j = &z[__getIndex(j*neq)];
               if (sdirk->D[j] != 0.0)
                  sdirk_daxpy (neq, sdirk->D[j], z_j, y);
            }

            t += h;
            nst++;
         }

         HScalingFactor = 0.9 * pow( 1.0 / herr, (1.0 / sdirk->ELO));

         // Reuse the Jacobian if the Newton Solver is converging fast enough.
         ComputeJ = (NewtonTheta > NewtonThetaMin);

         // Don't refine if it's not a big step and we could reuse the M matrix.
         int recycle_M = !(ComputeJ) && (HScalingFactor >= Qmin && HScalingFactor <= Qmax);
         if (recycle_M)
         {
            ComputeM = 0;
            HScalingFactor = 1.0;
         }
         else
            ComputeM = 1;
      }

      // Restrict the rate of change in dt
      HScalingFactor = fmax( HScalingFactor, 1.0 / sdirk->adaption_limit);
      HScalingFactor = fmin( HScalingFactor,       sdirk->adaption_limit);

#ifdef VERBOSE
      if (iter % (VERBOSE) == 0)
         printf("iter = %d: passed=%d ... t = %e, HScalingFactor = %f %f %e\n", iter, (Accepted ? (h <= sdirk->h_min ? -1 : 1) : 0), t, HScalingFactor, y[__getIndex(neq-1)], h);
#endif

      // Apply grow/shrink factor for next step.
      h *= HScalingFactor;

      // Limit based on the upper/lower bounds
      h = fmin(h, sdirk->h_max);
      h = fmax(h, sdirk->h_min);

      // Stretch the final step if we're really close and we didn't just fail ...
      //if (herr <= 1.0 && fabs((t + h) - sdirk->t_stop) < sdirk->h_min)
      if (Accepted && fabs((t + h) - sdirk->t_stop) < sdirk->h_min)
      {
         h = sdirk->t_stop - t;
         //fprintf(stderr,"fabs((t + h) - sdirk->t_stop) < sdirk->h_min: %d %d %e\n", nst, iter, h);
         ComputeM = 1;
      }

      // Don't overshoot the final time ...
      if (t + h > sdirk->t_stop)
      {
         //fprintf(stderr,"t + h > sdirk->t_stop: %d %d %e %e\n", nst, iter, h, sdirk->t_stop - t);
         h = sdirk->t_stop - t;
         ComputeM = 1;
      }

      ++iter;
      if (sdirk->max_iters && iter > sdirk->max_iters) {
         ierr = SDIRK_TOO_MUCH_WORK;
         //printf("(iter > max_iters)\n");
         break;
      }
   }

#ifdef VERBOSE
   printf("nst=%d nit=%d nfe=%d nje=%d nlu=%d nni=%d (%4.1f, %3.1f)\n", nst, iter, nfe, nje, nlu, nni, (double)(nfe-nje*(neq+1)) / nst, (double)nni / (nst*sdirk->numStages));
#endif

   return ierr;

   #undef nst
   #undef nfe
   #undef nje
   #undef nlu
   #undef nni
   #undef iter
   #undef h
   #undef t
   #undef neq
   #undef InterpolateNewton
   #undef MaxNewtonIterations
   #undef Qmax
   #undef Qmin
   #undef NewtonThetaMin
   #undef NewtonThetaMax
   #undef NewtonTolerance
}

#undef __matrix_index
#undef A_
#undef Theta_
#undef Alpha_
