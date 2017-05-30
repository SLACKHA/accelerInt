#include <cl_macros.h>

// OpenCL path ...
#ifdef __OPENCL_VERSION__

#else

// Normal GCC/CPU path ...

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#endif // __OPENCL_VERSION__

//#ifndef __cplusplus
//  #define __restrict__ restrict
//#endif

#include <cklib.h>

// Internal utility functions ...

__inline double sqr (const double p) { return (p*p); }

#ifndef __OPENCL_VERSION__
// This is in the OpenCL spec but not in c99 ... argh.
__inline double exp10 (const double p) { return exp(p * 2.30258509299404568402); }
#endif

#ifndef FAST_MATH
#define FAST_MATH
#endif

#ifdef FAST_MATH
#warning 'enabled FAST MATH pow() functions'
//__inline bool is_odd (unsigned int q) { return (bool)(q % 2); }

// p^q where q is a positive integral
__inline double __fast_powu (double p, unsigned q)
{
   if      (q == 0) return 1.0;
   else if (q == 1) return p;
   else if (q == 2) return p*p;
   else if (q == 3) return p*p*p;
   else if (q == 4) return p*p*p*p;
   else
   {
      // q^p -> (q^(p/2))^2 ... recursively takes log(q) ops
      double r = 1;
      while (q)
      {
         if (q % 2) //is_odd(q)) // odd power ...
         {
            r *= p;
            --q;
         }
         else
         {
            p *= p; // square the base ...
            q /= 2; // drop the power by two ...
         }
      }
      return r;
   }
}
// p^q where q is an integral
__inline double __fast_powi (double p, int q)
{
   if (p == 0.0)
   {
      if (q == 0)
         return 1.0;
      //else if (q < 0)
      //   return std::numeric_limits<double>::infinity();
      else
         return 0.0;
   }

   if      (q > 0) return __fast_powu(p,q);
   else if (q < 0) return __fast_powu(1.0/p,(unsigned int)(-q));
   else            return 1.0;
}
#else
__inline double __fast_powu (double p, unsigned int q) { return __builtin_powi(p,q); }
__inline double __fast_powi (double p, int q) { return __builtin_powi(p,q); }
#endif

//__inline double pow(const double &a, const double &b) { return std::pow(a,b); }
__inline double __powi(const double a, const int b) { return __fast_powi(a,b); }
__inline double __powu(const double a, const unsigned int b) { return __fast_powu(a,b); }

// Skip this for OpenCL ...
#ifndef __OPENCL_VERSION__

ckdata_t* ck_create
          (int kk,
           char* sp_name[], double *sp_mwt,
           double *th_tmid, double *th_alo, double *th_ahi,
           int ii,
           double *rx_A, double *rx_b, double *rx_E,
           int *rx_nu, int *rx_nuk,
           int n_rev,
           int *rx_rev_idx, double *rx_rev_A, double *rx_rev_b, double *rx_rev_E,
           int n_irrev,
           int *rx_irrev_idx,
           int n_thdbdy,
           int *rx_thdbdy_idx, int *rx_thdbdy_offset, int *rx_thdbdy_spidx, double *rx_thdbdy_alpha,
           int n_falloff,
           int *rx_falloff_idx, int *rx_falloff_type, int *rx_falloff_spidx, double *rx_falloff_params)
{
   ckdata_t *ck = (ckdata_t*)malloc(sizeof(ckdata_t));

   const int verbose = 0;

   // Species info ...
   ck->n_species = kk;
   if (verbose) printf("n_species = %d\n", kk);

   if (kk > __ck_max_sp)
   {
      fprintf(stderr,"kk > __ck_max_sp %d %d\n", kk, __ck_max_sp);
      exit(-1);
   }

   //ck->sp_name = (char**)malloc(sizeof(char*) * kk);
   //ck->sp_mwt  = (double*)malloc(sizeof(double) * kk);

   //ck->th_tmid = (double*)malloc(sizeof(double) * kk);
   //ck->th_alo  = (double*)malloc(sizeof(double) * __ck_max_th_terms * kk);
   //ck->th_ahi  = (double*)malloc(sizeof(double) * __ck_max_th_terms * kk);

   for (int k = 0; k < kk; ++k)
   {
      //ck->sp_name[k] = (char*)malloc(sizeof(char)*__ck_max_strlen);
      //strncpy (ck->sp_name[k], sp_name[k], __ck_max_strlen);

      ck->sp_mwt[k] = sp_mwt[k];

      ck->th_tmid[k] = th_tmid[k];
      for (int i = 0; i < __ck_max_th_terms; ++i) ck->th_alo[k][i] = th_alo[k*__ck_max_th_terms+i];
      for (int i = 0; i < __ck_max_th_terms; ++i) ck->th_ahi[k][i] = th_ahi[k*__ck_max_th_terms+i];

      if (verbose) printf("%3d: name = %s, mwt = %f, tmid = %f\n", k, sp_name[k], ck->sp_mwt[k], ck->th_tmid[k]);
   }

   // Reaction info ...
   ck->n_reactions = ii;
   if (verbose) printf("n_reactions = %d\n", ii);

   if (ii > __ck_max_rx)
   {
      fprintf(stderr,"ii > __ck_max_rx %d %d\n", ii, __ck_max_rx);
      exit(-1);
   }
   //ck->rx_A = (double*)malloc(sizeof(double) * ii);
   //ck->rx_b = (double*)malloc(sizeof(double) * ii);
   //ck->rx_E = (double*)malloc(sizeof(double) * ii);

   //ck->rx_nu    = (int*)malloc(sizeof(int) * ii*__ck_max_rx_order*2);
   //ck->rx_nuk   = (int*)malloc(sizeof(int) * ii*__ck_max_rx_order*2);
   //ck->rx_sumnu = (int*)malloc(sizeof(int) * ii);

   //ck->rx_info  = (int*)malloc(sizeof(int) * ii);

   for (int i = 0; i < ii; ++i)
   {
      ck->rx_A[i] = rx_A[i];
      ck->rx_b[i] = rx_b[i];
      ck->rx_E[i] = rx_E[i];
      if (verbose) printf("%3d: %e %e %e\n", i, ck->rx_A[i],ck->rx_b[i],ck->rx_E[i]);

      int sumnu = 0;
      for (int n = 0; n < __ck_max_rx_order*2; ++n)
      {
         //ck->rx_nu [i*__ck_max_rx_order*2 +n] = rx_nu [i*__ck_max_rx_order*2 +n];
         //ck->rx_nuk[i*__ck_max_rx_order*2 +n] = rx_nuk[i*__ck_max_rx_order*2 +n];
         ck->rx_nu [i][n] = rx_nu [i*__ck_max_rx_order*2 +n];
         ck->rx_nuk[i][n] = rx_nuk[i*__ck_max_rx_order*2 +n];

         //sumnu += ck->rx_nu [i*__ck_max_rx_order*2 +n];
         sumnu += ck->rx_nu[i][n];
         //printf("%3d: nu, nuk=%d %d\n", i, nu[n], nuk[n]);
      }
      ck->rx_sumnu[i] = sumnu;

      // Initialize the rxn info flag
      ck->rx_info[i] = __rx_flag_nil;
   }

   // ... Reversible reaction with explicit parameters ...
   ck->n_reversible_reactions = n_rev;
   if (verbose) printf("n_reversible_reactions = %d\n", n_rev);

   if (n_rev > __ck_max_rev_rx)
   {
      fprintf(stderr,"n_rev > __ck_max_rev_rx %d %d\n", n_rev, __ck_max_rev_rx);
      exit(-1);
   }

   if (n_rev > 0)
   {
      //ck->rx_rev_idx = (int   *)malloc(sizeof(int   ) * n_rev);
      //ck->rx_rev_A   = (double*)malloc(sizeof(double) * n_rev);
      //ck->rx_rev_b   = (double*)malloc(sizeof(double) * n_rev);
      //ck->rx_rev_E   = (double*)malloc(sizeof(double) * n_rev);

      for (int n = 0; n < n_rev; ++n)
      {
         ck->rx_rev_idx[n] = rx_rev_idx[n];
         ck->rx_rev_A[n] = rx_rev_A[n];
         ck->rx_rev_b[n] = rx_rev_b[n];
         ck->rx_rev_E[n] = rx_rev_E[n];

         //printf("%3d: [%3d], rev_A = %e, rev_b = %f, rev_E = %e; \n", n, ck->rx_rev_idx[n], ck->rx_rev_A[n], ck->rx_rev_b[n], ck->rx_rev_E[n]);

         int k = ck->rx_rev_idx[n];
         __enable(ck->rx_info[k], __rx_flag_rparams);
      }
   }

   // ... Irreversible reactions ...
   ck->n_irreversible_reactions = n_irrev;
   if (verbose) printf("n_irreversible_reactions = %d\n", n_irrev);

   if (n_irrev > __ck_max_irrev_rx)
   {
      fprintf(stderr,"n_irrev > __ck_max_irrev_rx %d %d\n", n_irrev, __ck_max_irrev_rx);
      exit(-1);
   }

   if (n_irrev > 0)
   {
      //ck->rx_irrev_idx = (int*)malloc(sizeof(int) * n_irrev);

      for (int n = 0; n < n_irrev; ++n)
      {
         ck->rx_irrev_idx[n] = rx_irrev_idx[n];

         int k = ck->rx_irrev_idx[n];
         __enable(ck->rx_info[k], __rx_flag_irrev);
         if (verbose) printf("%3d: is irreversible\n", k);
      }
   }

   // ... 3rd-body efficiencies for pressure dependence ...
   ck->n_thdbdy = n_thdbdy;
   if (verbose) printf("n_thdbdy = %d\n", n_thdbdy);

   if (n_thdbdy > __ck_max_thdbdy_rx)
   {
      fprintf(stderr,"n_thdbdy > __ck_max_thdbdy_rx %d %d\n", n_thdbdy, __ck_max_thdbdy_rx);
      exit(-1);
   }
   if (n_thdbdy > 0)
   {
      //ck->rx_thdbdy_idx = (int*)malloc(sizeof(int) * n_thdbdy);
      //ck->rx_thdbdy_offset = (int*)malloc(sizeof(int) * n_thdbdy+1);

      for (int n = 0; n < n_thdbdy; ++n)
      {
         ck->rx_thdbdy_idx[n] = rx_thdbdy_idx[n];

         int k = ck->rx_thdbdy_idx[n];
         __enable(ck->rx_info[k], __rx_flag_thdbdy);
         if (verbose) printf("%d, rxn %d is third-body\n", n, k);

         //printf("%d, ck->rx_thdbdy_idx=%d, ck->rx_thdbdy_offset=%d, nsp=%d\n", n, ck->rx_thdbdy_idx[n], ck->rx_thdbdy_offset[n], n_sp);
      }

      for (int n = 0; n <= n_thdbdy; ++n)
         ck->rx_thdbdy_offset[n] = rx_thdbdy_offset[n];
      //printf("n_coefs=%d\n", n_thdbdy_coefs);

      int n_thdbdy_coefs = rx_thdbdy_offset[n_thdbdy];

      if (n_thdbdy_coefs > __ck_max_thdbdy_coefs)
      {
         fprintf(stderr,"n_thdbdy_coefs > __ck_max_thdbdy_coefs %d %d\n", n_thdbdy_coefs, __ck_max_thdbdy_coefs);
         exit(-1);
      }

      //ck->rx_thdbdy_spidx = (int*)malloc(sizeof(int) *n_thdbdy_coefs);
      //ck->rx_thdbdy_alpha = (double*)malloc(sizeof(double) *n_thdbdy_coefs);

      for (int i = 0; i < n_thdbdy_coefs; ++i)
      {
         ck->rx_thdbdy_spidx[i] = rx_thdbdy_spidx[i];
         ck->rx_thdbdy_alpha[i] = rx_thdbdy_alpha[i];
      }
   }

   // ... Fall-off pressure dependencies ...
   ck->n_falloff = n_falloff;
   if (verbose) printf("n_falloff = %d\n", n_falloff);

   if (n_falloff > __ck_max_falloff_rx)
   {
      fprintf(stderr,"n_falloff > __ck_max_falloff_rx %d %d\n", n_falloff, __ck_max_falloff_rx);
      exit(-1);
   }

   if (n_falloff > 0)
   {
      //ck->rx_falloff_idx = (int*)malloc(sizeof(int) * n_falloff);
      //ck->rx_falloff_spidx = (int*)malloc(sizeof(int) * n_falloff);
      //ck->rx_falloff_params = (double*)malloc(sizeof(double) * n_falloff*__ck_max_falloff_params);

      for (int n = 0; n < n_falloff; ++n)
      {
         ck->rx_falloff_idx[n] = rx_falloff_idx[n];
         ck->rx_falloff_spidx[n] = rx_falloff_spidx[n];

         for (int i = 0; i < __ck_max_falloff_params; ++i)
            ck->rx_falloff_params[n][i] = rx_falloff_params[n*__ck_max_falloff_params+i];
            //ck->rx_falloff_params[n*__ck_max_falloff_params+i] = rx_falloff_params[n*__ck_max_falloff_params+i];

         int type = rx_falloff_type[n];

         int k = ck->rx_falloff_idx[n];
         __enable(ck->rx_info[k], __rx_flag_falloff);
         if (type == 1) {
            printf("SRI fall-off rxn not ready\n");
            exit(-1);
            __enable(ck->rx_info[k], __rx_flag_falloff_sri);
            if (type == 2)
               __enable(ck->rx_info[k], __rx_flag_falloff_sri5);
         }
         else if (type == 3) {
            __enable(ck->rx_info[k], __rx_flag_falloff_troe);
            if (type == 4)
               __enable(ck->rx_info[k], __rx_flag_falloff_troe4);
         }

         if (verbose) printf("falloff: %3d, %3d, %3d, %3d\n", n, ck->rx_falloff_idx[n], ck->rx_falloff_spidx[n], type);
      }
   }

   // Scratch space ...
   //ck->lenrwk_ = 2*ck->n_species + 4*ck->n_reactions;
   //ck->rwk_    = (double *) malloc(sizeof(double)*ck->lenrwk_);

   return ck;
}

void ck_destroy (ckdata_t **ck_)
{
   if (ck_ == NULL)
      return;

   ckdata_t *ck = *ck_;
   if (ck == NULL)
      return;

 //if (ck->sp_name) free(ck->sp_name);
 /*  if (ck->sp_mwt ) free(ck->sp_mwt );

   if (ck->th_tmid) free(ck->th_tmid);
   if (ck->th_alo ) free(ck->th_alo );
   if (ck->th_ahi ) free(ck->th_ahi );

   if (ck->rx_A) free(ck->rx_A);
   if (ck->rx_b) free(ck->rx_b);
   if (ck->rx_E) free(ck->rx_E);
   if (ck->rx_nu) free(ck->rx_nu);
   if (ck->rx_nuk) free(ck->rx_nuk);
   if (ck->rx_sumnu) free(ck->rx_sumnu);

   if (ck->rx_rev_A) free(ck->rx_rev_A);
   if (ck->rx_rev_b) free(ck->rx_rev_b);
   if (ck->rx_rev_E) free(ck->rx_rev_E);
   if (ck->rx_rev_idx) free(ck->rx_rev_idx);

   if (ck->rx_irrev_idx) free(ck->rx_irrev_idx);

   if (ck->rx_thdbdy_idx) free(ck->rx_thdbdy_idx);
   if (ck->rx_thdbdy_offset) free(ck->rx_thdbdy_offset);
   if (ck->rx_thdbdy_spidx) free(ck->rx_thdbdy_spidx);
   if (ck->rx_thdbdy_alpha) free(ck->rx_thdbdy_alpha);

   if (ck->rx_falloff_idx) free(ck->rx_falloff_idx);
   if (ck->rx_falloff_spidx) free(ck->rx_falloff_spidx);
   if (ck->rx_falloff_params) free(ck->rx_falloff_params);

   if (ck->rx_info) free(ck->rx_info);

   if (ck->rwk_) free (ck->rwk_); */

   free (ck);
}

#endif // __OPENCL_VERSION__

__inline size_t ck_lenrwk (__ckdata_attr const ckdata_t *ck) { return 2*ck->n_species + 4*ck->n_reactions; }

__inline double compute_H_RT (const int k, const double T, __ckdata_attr const ckdata_t* ck)
{
   //return a[0] + a[5] / T + T * (a[1] / 2.0 + T * (a[2] / 3.0 + T * (a[3] / 4.0 + T * a[4] / 5.0)));

   #define __equ(__a) (__a[0] + __a[5] / T + T * (__a[1] / 2.0 + T * (__a[2] / 3.0 + T * (__a[3] / 4.0 + T * __a[4] / 5.0))))

#if defined(__OPENCL_VERSION__) && (0)
   return select(__equ((ck->th_alo[k])), __equ((ck->th_ahi[k])), (long)isgreater(T, ck->th_tmid[k]));
#else
   if (T > ck->th_tmid[k])
      return __equ((ck->th_ahi[k]));
   else
      return __equ((ck->th_alo[k]));
#endif // __OPENCL_VERSION__

   #undef __equ
}

__inline double compute_Cp_R (const int k, const double T, __ckdata_attr const ckdata_t *ck)
{
   // Cp / R = Sum_(i=1)^(5){ a_i * T^(i-1) }

   //return a[0] + T * (a[1] + T * (a[2] + T * (a[3] + T * a[4])));
   #define __equ(__a) ( __a[0] + T * (__a[1] + T * (__a[2] + T * (__a[3] + T * __a[4]))) )

#if defined(__OPENCL_VERSION__) && (0)
   return select(__equ((ck->th_alo[k])), __equ((ck->th_ahi[k])), (long)isgreater(T, ck->th_tmid[k]));
#else
   if (T > ck->th_tmid[k])
      return __equ((ck->th_ahi[k]));
   else
      return __equ((ck->th_alo[k]));
#endif // __OPENCL_VERSION__

   #undef __equ
}

// Mean molecular weight given mole fractions ... g / mol
__inline double ckmmwx (__global const double x[], __ckdata_attr const ckdata_t *ck)
{
   // <W> = Sum_k { x_k * w_k }
   double mean_mwt = 0.0;
   for (int k = 0; k < ck->n_species; ++k)
      mean_mwt += x[__getIndex(k)] * ck->sp_mwt[k];

   return mean_mwt;
}

// Mean molecular weight given mass fractions ... g / mol
__inline double ckmmwy (__global const double y[], __ckdata_attr const ckdata_t *ck)
{  
   // <W> = 1 / Sum_k { y_k / w_k }
   double sumyow = 0.0;
   for (int k = 0; k < ck->n_species; ++k)
      sumyow += (y[__getIndex(k)] / ck->sp_mwt[k]);

   return 1.0 / sumyow;
}

// Return pointer to molecular weights ... g / mol
//__inline __global const double* ckwt (__ckdata_attr const ckdata_t *ck)
//{
//   return ck->sp_mwt;
//}

// Return pointer to internal real scratch array
//double *ckrwk (const ckdata_t *ck)
//{
//   return ck->rwk_;
//}

// Species enthalpies in mass units given temperature ... erg / g
__inline void ckhms (const double T, __global double *restrict h, __ckdata_attr const ckdata_t *ck)
{
   const double RUT = __RU__ * T;

   #ifdef __INTEL_COMPILER
   #pragma ivdep
   #endif
   for (int k = 0; k < ck->n_species; ++k)
   {
      double h_k = compute_H_RT(k, T, ck);
      h_k *= (RUT / ck->sp_mwt[k]);

      h[__getIndex(k)] = h_k;

      //h[k] *= (__RU__ / ck->sp_mwt[k]);
      //h[k] *= T;
   }
}
// Species internal energy in mass units given temperature ... erg / g
__inline void ckums (const double T, double *restrict u, __ckdata_attr const ckdata_t *ck)
{
   const double RUT = __RU__ * T;

   #ifdef __INTEL_COMPILER
   #pragma ivdep
   #endif
   for (int k = 0; k < ck->n_species; ++k)
   {
      // U = H - RT
      double u_k = compute_H_RT(k, T, ck) - 1.0;
      u_k *= (RUT / ck->sp_mwt[k]);

      u[__getIndex(k)] = u_k;

      //u[k] *= (__RU__ / ck->sp_mwt[k]);
      //u[k] *= T;
   }
}
// Species Cp in mass units given temperature ... erg / (g * k)
__inline void ckcpms (const double T, __global double *restrict cp, __ckdata_attr const ckdata_t *ck)
{
   #ifdef __INTEL_COMPILER
   #pragma ivdep
   #endif
   for (int k = 0; k < ck->n_species; ++k)
   {
      double cp_k = compute_Cp_R(k, T, ck);
      cp_k *= (__RU__ / ck->sp_mwt[k]);

      cp[__getIndex(k)] = cp_k;
   }
}
// Species Cv in mass units given temperature ... erg / (g * k)
__inline void ckcvms (const double T, __global double *restrict cv, __ckdata_attr const ckdata_t *ck)
{
   #ifdef __INTEL_COMPILER
   #pragma ivdep
   #endif
   for (int k = 0; k < ck->n_species; ++k)
   {
      //cv[k] = compute_Cp_R(k, T, ck) - 1.0;
      //cv[k] *= (RU / ck->sp_mwt[k]);

      double cv_k = compute_Cp_R(k, T, ck) - 1.0;
      cv_k *= (__RU__ / ck->sp_mwt[k]);

      cv[__getIndex(k)] = cv_k;
   }
}
// Mixture enthalpy in mass units given mass fractions and temperature ... erg / g
__inline double ckhbms (const double T, __global double y[], __ckdata_attr const ckdata_t *ck)
{
   //const ValueType RUT = RU * T;
   double h_mix = 0.0;

   #ifdef __INTEL_COMPILER
   #pragma ivdep
   #endif
   for (int k = 0; k < ck->n_species; ++k)
   {
      double h_k = compute_H_RT(k, T, ck);
      //h_k *= (RUT / ck->sp_mwt[k]);
      h_k /= ck->sp_mwt[k];
      //h_mix += (y[k] * h_k);
      h_k *= y[__getIndex(k)];
      h_mix += h_k;
   }

   //return h_mix;
   //return h_mix * RUT;
   h_mix *= (T * __RU__);
   return h_mix;
}

// Mixture internal energy in mass units given mass fractions and temperature ... erg / g
__inline double ckubms (const double T, __global const double y[], __ckdata_attr const ckdata_t *ck)
{
   double u_mix = 0.0;

   #ifdef __INTEL_COMPILER
   #pragma ivdep
   #endif
   for (int k = 0; k < ck->n_species; ++k)
   {
      const double u_k = compute_H_RT(k, T, ck) - 1.0;

      u_mix += (u_k * y[__getIndex(k)] / ck->sp_mwt[k]);
   }

   return (u_mix * __RU__ * T);
}

// Mixture Cp in mass units given mass fractions and temperature ... erg / (g * k)
__inline double ckcpbs (const double T, __global const double y[], __ckdata_attr const ckdata_t *ck)
{
   double cp_mix = 0.;

   #ifdef __INTEL_COMPILER
   #pragma ivdep
   #endif
   for (int k = 0; k < ck->n_species; ++k)
   {
      double cp_k = compute_Cp_R(k, T, ck);
      cp_k *= (__RU__ / ck->sp_mwt[k]);
      cp_k *= y[__getIndex(k)];
      cp_mix += cp_k;
      //cp_mix += (y[k] * cp_k);
   }

   return cp_mix;
}

// Mixture Cv in mass units given mass fractions and temperature ... erg / (g * k)
__inline double ckcvbs (const double T, __global const double y[], __ckdata_attr const ckdata_t *ck)
{
   double cv_mix = 0.;

   #ifdef __INTEL_COMPILER
   #pragma ivdep
   #endif
   for (int k = 0; k < ck->n_species; ++k)
   {
      //const double cv_k = compute_Cp_R(k, T, ck) - 1.0;
      //cv_mix += (cv_k * y[k] / ck->sp_mwt[k]);

      double cv_k = compute_Cp_R(k, T, ck) - 1.0;
      cv_k *= (__RU__ / ck->sp_mwt[k]);
      cv_k *= y[__getIndex(k)];

      cv_mix += cv_k;
   }

   //return (cv_mix * RU);
   return cv_mix;
}

// Mixture Cp in molar units given mole fractions and temperature ... erg / (g * k)
__inline double ckcpbl (const double T, __global const double x[], __ckdata_attr const ckdata_t *ck)
{ 
   double cp_mix = 0.;

   #ifdef __INTEL_COMPILER
   #pragma ivdep
   #endif
   for (int k = 0; k < ck->n_species; ++k)
   {
      const double cp_k = compute_Cp_R(k, T, ck); 

      cp_mix += (cp_k * x[__getIndex(k)]);
   }

   return (cp_mix * __RU__);
}

// Species S/R - H/RT ... special function.
__inline void cksmh (const double T,
                   const double logT,
                   __global double *restrict smh,
                   __ckdata_attr const ckdata_t *ck)
{
   //const double logTm1 = log(T) - 1.;
   const double logTm1 = logT - 1.;
   const double invT   = 1. / T;
   const double T1     = T / 2.;
   const double T2     = T*T / 6.;
   const double T3     = T*T*T / 12.;
   const double T4     = T*T*T*T / 20.;

   #ifdef __INTEL_COMPILER
   #pragma ivdep
   #endif
   for (int k = 0; k < ck->n_species; ++k)
   {
      #define __equ(__a) ( __a[0] * logTm1 + T1 * __a[1] + T2 * __a[2] + T3 * __a[3] + T4 * __a[4] - __a[5] * invT + __a[6] )

#if defined(__OPENCL_VERSION__) && (0)
      smh[__getIndex(k)] = select(__equ((ck->th_alo[k])), __equ((ck->th_ahi[k])), (long)isgreater(T, ck->th_tmid[k]));
#else
      if (T > ck->th_tmid[k])
         smh[__getIndex(k)] = __equ((ck->th_ahi[k]));
      else
         smh[__getIndex(k)] = __equ((ck->th_alo[k]));
#endif // __OPENCL_VERSION__

      #undef __equ
   }
}

// Mixture density given pressure, temperature and mass fractions ... g / cm^3
__inline double ckrhoy (const double p, const double T, __global const double y[], __ckdata_attr const ckdata_t *ck)
{
   //const double mean_mwt = ckmmwy(y, ck);
   //const ValueType mean_mwt = ckmmwy(y, ck);
   double mean_mwt = ckmmwy(y, ck);

   // rho = p / (<R> * T) = p / (RU / <W> * T)

   //return p / (T * RU / mean_mwt);
   mean_mwt *= p;
   mean_mwt /= T;
   mean_mwt /= __RU__;
   return mean_mwt;
}

// Mixture pressure given mixture density, temperature and mass fractions ... dyne / cm^2
__inline double ckpy (const double rho, const double T, __global const double y[], __ckdata_attr const ckdata_t *ck)
{
   const double mean_mwt = ckmmwy(y, ck);

   // p = rho * (RU / <W>) * T

   //return rho * T * RU / mean_mwt;
   return (rho * T / mean_mwt) * __RU__;
}
// Compute mass fractions given mole fractions
__inline void ckxty (__global const double *restrict x, __global double *restrict y, __ckdata_attr const ckdata_t *ck)
{
   // y_k = x_k * w_k / <W>
   const double mean_mwt_inv = 1. / ckmmwx(x,ck);

   #ifdef __INTEL_COMPILER
   #pragma ivdep
   #endif
   for (int k = 0; k < ck->n_species; ++k)
      y[__getIndex(k)] = x[__getIndex(k)] * (ck->sp_mwt[k] * mean_mwt_inv);
}
// Compute the molar concentration given rho/y_k
__inline void ckytcr (const double rho, __global const double *restrict y, __global double *restrict c, __ckdata_attr const ckdata_t *ck)
{
   // [c]_k = rho * y_k / w_k

   #ifdef __INTEL_COMPILER
   #pragma ivdep
   #endif
   for (int k = 0; k < ck->n_species; ++k)
   {
      double c_k = rho * y[__getIndex(k)] / ck->sp_mwt[k];
      c[__getIndex(k)] = c_k;
   }
}
// Compute the molar concentration given p/T/y_k
__inline void ckytcp (const double p, const double T, __global const double *restrict y, __global double *restrict c, __ckdata_attr const ckdata_t *ck)
{
   // [c]_k = rho * y_k / w_k
   const double rho = ckrhoy (p, T, y, ck);

   //ckytcr (rho, y, c, ck);
   #ifdef __INTEL_COMPILER
   #pragma ivdep
   #endif
   for (int k = 0; k < ck->n_species; ++k)
   {
      //c[k] = rho * y[k] / ck->sp_mwt[k];
      double c_k = rho * y[__getIndex(k)] / ck->sp_mwt[k];
      c[__getIndex(k)] = c_k;
   }
}

// Compute temperature-dependent molar forward/reverse reaction rates
// ... utility function ...
__inline void ckratt_ (const double T,
                     __global double *restrict smh,
                     __global double *restrict eqk,
                     __global double *restrict rkf,
                     __global double *restrict rkr,
                     __ckdata_attr const ckdata_t *ck)
{
   const int kk = ck->n_species;
   const int ii = ck->n_reactions;

   const double logT = log(T);
   const double invT = 1.0 / T;
   const double pfac = __PA__ / (__RU__ * T); // (dyne / cm^2) / (erg / mol / K) / (K)

   // I. Temperature-dependent rates ...

   // S/R - H/RT ... only needed for equilibrium.
   cksmh (T, logT, smh, ck);

   #ifdef __INTEL_COMPILER
   #pragma ivdep
   #endif
   for (int i = 0; i < ii; ++i)
   {
      // Basic Arrhenius rates: A * exp( logT * b - E_R / T)
      rkf[__getIndex(i)] = ck->rx_A[i] * exp(ck->rx_b[i] * logT - ck->rx_E[i] * invT);
   }

   #ifdef __INTEL_COMPILER
   #pragma ivdep
   #endif
   for (int i = 0; i < ii; ++i)
   {
      // Irreversible reaction ...
      if (__is_enabled(ck->rx_info[i], __rx_flag_irrev))
      {
         rkr[__getIndex(i)] = 0.0;
         //eqk[i] = 0.0;
      }
      // Reversible parameters ...
      else if (__is_enabled(ck->rx_info[i], __rx_flag_rparams))
      {
         rkr[__getIndex(i)] = ck->rx_rev_A[i] * exp(ck->rx_rev_b[i] * logT - ck->rx_rev_E[i] * invT);
         //eqk[i] = rkf[i] / rkr[i];
      }
      // Use equilibrium for reversible rate ...
      else
      {
         // Sum_k { nu_k * (S/R - H/RT)_k }

         #define __nu (ck->rx_nu[i])
         #define __nuk (ck->rx_nuk[i])

         double              sumsmh  = (__nu[0] * smh[__getIndex(__nuk[0])]);
         if (__nuk[1] != -1) sumsmh += (__nu[1] * smh[__getIndex(__nuk[1])]);
         if (__nuk[2] != -1) sumsmh += (__nu[2] * smh[__getIndex(__nuk[2])]);
                             sumsmh += (__nu[3] * smh[__getIndex(__nuk[3])]);
         if (__nuk[4] != -1) sumsmh += (__nu[4] * smh[__getIndex(__nuk[4])]);
         if (__nuk[5] != -1) sumsmh += (__nu[5] * smh[__getIndex(__nuk[5])]);

         #undef __nu
         #undef __nuk

         //eqk[__getIndex(i)] = exp(fmin(sumsmh, __exparg__));
         double eqk_ = exp(fmin(sumsmh, __exparg__));

         if (ck->rx_sumnu[i] != 0)
            eqk_ *= __powi(pfac,ck->rx_sumnu[i]);
            //eqk[__getIndex(i)] *= __powi(pfac,ck->rx_sumnu[i]);

         //if (!(ck->rx_info[i] & __rx_flag_irrev))
            //rkr[__getIndex(i)] = rkf[__getIndex(i)] / fmax(eqk[__getIndex(i)],__small__);
            rkr[__getIndex(i)] = rkf[__getIndex(i)] / fmax(eqk_,__small__);
      }
   }

}
__inline void ckratc_ (const double T,
                     __global const double *restrict c,
                     __global       double *restrict ctb,
                     __global       double *restrict rkf,
                     __global       double *restrict rkr,
                     __ckdata_attr const ckdata_t *ck)
{
   const int kk = ck->n_species;
   const int ii = ck->n_reactions;

   const double logT = log(T);
   const double invT = 1.0 / T;

   // II. Concentration-dependent rates ...

   for (int i = 0; i < ii; ++i)
      ctb[__getIndex(i)] = 1.0;

   // Third-body reactions ...
   if (ck->n_thdbdy > 0)
   {
      double ctot = 0.0;
      for (int k = 0; k < kk; ++k)
         ctot += c[__getIndex(k)];

      for (int n = 0; n < ck->n_thdbdy; ++n)
      {
         const int rxn_idx = ck->rx_thdbdy_idx[n];

         ctb[__getIndex(rxn_idx)] = ctot;

         // Add in the specific efficiencies ...

         for (int m = ck->rx_thdbdy_offset[n]; m < ck->rx_thdbdy_offset[n+1]; ++m)
         {
            const int k = ck->rx_thdbdy_spidx[m];
            ctb[__getIndex(rxn_idx)] += (ck->rx_thdbdy_alpha[m] - 1.0) * c[__getIndex(k)];
         }
      }
   }

   // Fall-off pressure dependencies ...
   if (ck->n_falloff > 0)
   {
      #ifdef __INTEL_COMPILER
      #pragma ivdep
      #endif
      for (int n = 0; n < ck->n_falloff; ++n)
      {
         const int rxn_idx = ck->rx_falloff_idx[n];

         // Concentration of the third-body ... could be a specific species, too.
         double cthb;
         if (ck->rx_falloff_spidx[n] != -1)
         {
            cthb = ctb[__getIndex(rxn_idx)];
            ctb[__getIndex(rxn_idx)] = 1.0;
         }
         else
            cthb = c[ __getIndex( ck->rx_falloff_spidx[n] ) ];

         #define __fpar (ck->rx_falloff_params[n])

         // Low-pressure limit rate ...
         double rklow = __fpar[0] * exp(__fpar[1] * logT - __fpar[2] * invT);

         // Reduced pressure ...
         double pr    = rklow * cthb / rkf[__getIndex(rxn_idx)];

         // Correction ... k_infty (pr / (1+pr)) * F()
         double p_cor;

         // Different F()'s ...
         //if (ck->rx_info[rxn_idx] & __rx_flag_falloff_sri)
         //{
         //   printf("SRI fall-off rxn not ready\n");
         //   exit(-1);
         //}
         //else if (ck->rx_info[rxn_idx] & __rx_flag_falloff_troe)
         if (__is_enabled(ck->rx_info[rxn_idx], __rx_flag_falloff_troe))
         {
            // 3-parameter Troe form ...
            double Fcent = (1.0 - __fpar[3]) * exp(-T / __fpar[4]) + __fpar[3] * exp(-T / __fpar[5]);

            // Additional 4th (T**) parameter ...
            if (__is_enabled(ck->rx_info[rxn_idx], __rx_flag_falloff_troe4))
               Fcent += exp(-__fpar[6] * invT);

            double log_Fc = log10( fmax(Fcent,__small__) );
            double eta    = 0.75 - 1.27 * log_Fc;
            double log_pr = log10( fmax(pr,__small__) );
            double plus_c = log_pr - (0.4 + 0.67 * log_Fc);
          //double _tmp   = plus_c / (eta - 0.14 * plus_c);
            double log_F  = log_Fc / (1.0 + sqr(plus_c / (eta - 0.14 * plus_c)));
            double Fc     = exp10(log_F);

            p_cor = Fc * (pr / (1.0 + pr));
         }
         else // Lindermann form
         {
            p_cor = pr / (1.0 + pr);
         }

         #undef __fpar

         rkf[__getIndex(rxn_idx)] *= p_cor;
         rkr[__getIndex(rxn_idx)] *= p_cor;
         //printf("%3d, %3d, %e, %e\n", n, rxn_idx, ck->rx_info[rxn_idx], p_cor, cthb);
      }

   } // fall-off's

   // II. Stoichiometry rates ...

//#ifdef __MIC__
//   #warning 'ivdep for Stoichiometric rates'
   #ifdef __INTEL_COMPILER
   #pragma ivdep
   #endif
//#endif
   for (int i = 0; i < ii; ++i)
   {
      #define __nu (ck->rx_nu[i])
      #define __nuk (ck->rx_nuk[i])

      double rkf_ = rkf[__getIndex(i)] * ctb[__getIndex(i)];
      double rkr_ = rkr[__getIndex(i)] * ctb[__getIndex(i)];

                             rkf_ *= __powu( c[__getIndex(__nuk[0])],-__nu[0]);
      if (__nuk[1] != -1) {  rkf_ *= __powu( c[__getIndex(__nuk[1])],-__nu[1]);
         if (__nuk[2] != -1) rkf_ *= __powu( c[__getIndex(__nuk[2])],-__nu[2]);
      }

                             rkr_ *= __powu( c[__getIndex(__nuk[3])], __nu[3]);
      if (__nuk[4] != -1) {  rkr_ *= __powu( c[__getIndex(__nuk[4])], __nu[4]);
         if (__nuk[5] != -1) rkr_ *= __powu( c[__getIndex(__nuk[5])], __nu[5]);
      }

      #undef __nu
      #undef __nuk

      rkf[__getIndex(i)] = rkf_;
      rkr[__getIndex(i)] = rkr_;
   }
}

void ckwyp (const double p,
            const double T,
            __global const double *restrict y,
            __global       double *restrict wdot,
            __ckdata_attr const ckdata_t *ck,
            __global double rwk[])
{
   const int kk = ck->n_species;
   const int ii = ck->n_reactions;

   __global double *rkf = rwk;
   __global double *rkr = rkf + __getIndex(ii);
   __global double *ctb = rkr + __getIndex(ii);
   __global double *c   = ctb + __getIndex(ii);
   __global double *smh = c;
   //double * restrict eqk = ctb;

   // Compute temperature-dependent forward/reverse rates ... mol / cm^3 / s
   //ckratt_ (T, smh, eqk, rkf, rkr, ck);
   ckratt_ (T, smh, 0, rkf, rkr, ck);

   // Convert to molar concentrations ... mol / cm^3
   ckytcp (p, T, y, c, ck);

   // Compute concentration-dependent forward/reverse rates ... mol / cm^3 / s
   ckratc_ (T, c, ctb, rkf, rkr, ck);

   // Compute species net production rates ... mol / cm^3 / s

   for (int k = 0; k < kk; ++k)
      wdot[__getIndex(k)] = 0.0;

   for (int i = 0; i < ii; ++i)
   {
      const double rop = rkf[__getIndex(i)] - rkr[__getIndex(i)];

      #define __nu (ck->rx_nu[i])
      #define __nuk (ck->rx_nuk[i])

                             wdot[__getIndex(__nuk[0])] += (rop * __nu[0]);
      if (__nuk[1] != -1) {  wdot[__getIndex(__nuk[1])] += (rop * __nu[1]);
         if (__nuk[2] != -1) wdot[__getIndex(__nuk[2])] += (rop * __nu[2]);
      }

                             wdot[__getIndex(__nuk[3])] += (rop * __nu[3]);
      if (__nuk[4] != -1) {  wdot[__getIndex(__nuk[4])] += (rop * __nu[4]);
         if (__nuk[5] != -1) wdot[__getIndex(__nuk[5])] += (rop * __nu[5]);
      }

      #undef __nu
      #undef __nuk
   }
}
/*void ckwc (const double T, double const c[], double wdot[], const ckdata_t *ck)
{
   const int kk = ck->n_species;
   const int ii = ck->n_reactions;

   double *rkf = ck->rwk_;
   double *rkr = rkf + ii;
   double *ctb = rkr + ii;
   double *smh = ctb + ii;
   double *eqk = ctb;

   // Compute temperature-dependent forward/reverse rates ... mol / cm^3 / s
   ckratt_ (T, smh, eqk, rkf, rkr, ck);

   // Compute concentration-dependent forward/reverse rates ... mol / cm^3 / s
   ckratc_ (T, c, ctb, rkf, rkr, ck);

   // Compute species net production rates ... mol / cm^3 / s

   for (int k = 0; k < kk; ++k)
      wdot[k] = 0.0;

   for (int i = 0; i < ii; ++i)
   {
      const double rop = rkf[i] - rkr[i];

      const int *nu  = &ck->rx_nu[i*__ck_max_rx_order*2];
      const int *nuk = &ck->rx_nuk[i*__ck_max_rx_order*2];

                           wdot[nuk[0]] += (rop * nu[0]);
      if (nuk[1] != -1) {  wdot[nuk[1]] += (rop * nu[1]);
         if (nuk[2] != -1) wdot[nuk[2]] += (rop * nu[2]);
      }

                           wdot[nuk[3]] += (rop * nu[3]);
      if (nuk[4] != -1) {  wdot[nuk[4]] += (rop * nu[4]);
         if (nuk[5] != -1) wdot[nuk[5]] += (rop * nu[5]);
      }
   }
}*/

// Meta-function to compute RHS for constant-pressure reaction system
void ckrhs (const double p,
            const double T,
            __global const double y[],
            __global       double ydot[],
            __ckdata_attr const ckdata_t *ck,
            __global       double rwk[])
{
   const int kk = ck->n_species;

   /* Compute local density given p/T/y_k */
   //const double rho = ckrhoy(p, T, y, ck);

   /* Compute the molar concentration ( mol / cm^3 ) */
   //ckytcr (rho, y, ydot /* as c[] */, ck);

   /* Compute molar reaction rate. (mol / (s*cm^3) */
   //ckwc (T, ydot /* as c[]*/, ydot, ck);
   ckwyp (p, T, y, ydot, ck, rwk);

   /* Compute mixture Cp (ergs / gm*K) */
   const double cp_mix = ckcpbs(T, y, ck);

   /* Compute species enthalpy (ergs / K) */
   ckhms(T, rwk, ck);

   /* Extract the molecular weights of the species ... this could just be a pointer. */
   const double rho = ckrhoy(p, T, y, ck);

   double Tdot = 0.0;
   for (int k = 0; k < kk; ++k)
   {
      /* Convert from molar to mass units. */
      ydot[__getIndex(k)] *= ck->sp_mwt[k];
      ydot[__getIndex(k)] /= rho;

      /* Sum up the net enthalpy change. */
      Tdot -= (rwk[__getIndex(k)] * ydot[__getIndex(k)]);
   }

   ydot[__getIndex(kk)] = Tdot / cp_mix;

   return;
}
