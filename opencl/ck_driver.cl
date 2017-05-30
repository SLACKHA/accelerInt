//#include <cklib.c>

#if defined(__Alignment) && (__Alignment > 0)
__global char *align_pointer (__global void *vptr, size_t type_size)
{
   type_size = max(type_size, (size_t)__Alignment);
   __global char *ptr = (__global char *) vptr;
   size_t offset = ((size_t)ptr) % type_size;
   if (offset)
      ptr += (type_size - offset);

   return ptr;
}
#endif

#if 1
#warning 'Skipping ck_driver'
#else
void __kernel
__attribute__((reqd_work_group_size(__blockSize, 1, 1)))
ck_driver (const double p,
           const double T,
           __global const double *u,
           __global       double *udot,
           __ckdata_attr const ckdata_t *ck,
           __global       double *rwk,
           const int num_evals)
{
   const int tid = get_global_id(0);
// const int gid = get_group_id(0);
// const int blockSize = get_local_size(0); // == __blockSize
// const int num_threads = get_global_size(0);

   const int kk = ck->n_species;
   const int neq = kk+1;
   const int lenrwk = ck_lenrwk(ck);

   // Thread-local pointers ...
   //__global double *my_rwk = rwk + gid*(__getIndex(lenrwk+2*neq)) + get_local_id(0);//tid;
//   if (__arrayStride == 1) my_rwk = rwk + tid*(__getIndex(lenrwk+2*neq));
#if defined(__Alignment) && (__Alignment > 0)
   rwk = (__global double *) align_pointer(rwk, sizeof(double));
#endif
   __global double *my_rwk = rwk
                        + (tid / __arrayStride) * (__getIndex(lenrwk+2*neq))
                        + (tid % __arrayStride);
   __global double *my_u = my_rwk + (__getIndex(lenrwk));
   __global double *my_udot = my_u + (__getIndex(neq));

   for (int i = get_global_id(0); i < num_evals; i += get_global_size(0))
   {
      const double T0 = T + (1000.*i) / num_evals;

      for (int k = 0; k < neq; ++k)
         my_u[__getIndex(k)] = u[ k ];

      ckrhs (p, T0, my_u, my_udot, ck, my_rwk);
      //ckwyp (p, T0, my_u, my_udot, ck, my_rwk);
      //ckhms (T0, my_udot, ck);
      //ckytcp (p, T0, my_u, my_udot, ck);

      //my_udot[__getIndex(kk)] = ckcpbs (T0, my_u, ck);
      //my_udot[__getIndex(0)] = ckrhoy (p, T0, my_u, ck);

      for (int k = 0; k < neq; ++k)
         udot[ neq*i + k ] = my_udot[__getIndex(k)];
   }
}
#endif

#define __STRINGIFY(__x) #__x
#define STRINGIFY(__x) __STRINGIFY(__x)
#define __PASTE(a,b) a ## b
#define PASTE(a,b) __PASTE(a,b)

#ifndef __ValueSize
  #define __ValueSize 1
#endif

#if (__ValueSize > 1)// && defined(__OPENCL_VERSION__)
  #define __ValueType PASTE(double, __ValueSize)
  #define __IntType PASTE(int, __ValueSize)
  #define __MaskType PASTE(long, __ValueSize)
#else
  #define __ValueType double
  #define __IntType int
  #define __MaskType int
#endif

#define FUNC_SIZE(__a) PASTE( __a, PASTE(__, __ValueSize) )
#define FUNC_TYPE(__a) PASTE( __a, PASTE(__, __ValueType) )

#ifdef __ckobj_name__
  #define __ckobj__ (__ckobj_name__)
#else
  #define __ckobj__ (*ck)
#endif

#if 1
  #pragma message "__ValueSize  = " STRINGIFY(__ValueSize)
  #pragma message "__ValueType  = " STRINGIFY(__ValueType)
  #pragma message "__MaskType   = " STRINGIFY(__MaskType)
  #pragma message "__ckobj__    = " STRINGIFY(__ckobj__)
  //#pragma message "FUNC_TYPE(func)   = " FUNC_TYPE("func")
#endif

inline __ValueType FUNC_TYPE(__fast_powu) (__ValueType p, unsigned q)
{
   if      (q == 0) return 1.0;
   else if (q == 1) return p;
   else if (q == 2) return p*p;
   else if (q == 3) return p*p*p;
   else if (q == 4) return p*p*p*p;
   else
   {
      // q^p -> (q^(p/2))^2 ... recursively takes log(q) ops
      __ValueType r = 1;
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
inline __ValueType FUNC_TYPE(__fast_powi) (__ValueType p, int q)
{
#if (__ValueSize == 1)
   if (p == 0.0)
   {
      if (q == 0)
         return 1.0;
      //else if (q < 0)
      //   return std::numeric_limits<double>::infinity();
      else
         return 0.0;
   }
#endif
   if      (q > 0) return FUNC_TYPE(__fast_powu)(p,q);
   else if (q < 0) return FUNC_TYPE(__fast_powu)(1.0/p,(unsigned int)(-q));
   else            return 1.0;
}

//inline double pow(const double &a, const double &b) { return std::pow(a,b); }
inline __ValueType FUNC_TYPE(__powi)(const __ValueType a, const int b) { return FUNC_TYPE(__fast_powi)(a,b); }
inline __ValueType FUNC_TYPE(__powu)(const __ValueType a, const unsigned int b) { return FUNC_TYPE(__fast_powu)(a,b); }

inline __ValueType FUNC_TYPE(__sqr) (const __ValueType p) { return (p*p); }

inline __ValueType FUNC_TYPE(compute_H_RT) (const int k, const __ValueType T, __ckdata_attr const ckdata_t* ck)
{
   #define __equ(__a) (__a[0] + __a[5] / T + T * (__a[1] / 2.0 + T * (__a[2] / 3.0 + T * (__a[3] / 4.0 + T * __a[4] / 5.0))))

#if (__ValueSize > 1)
   return select(__equ((__ckobj__.th_alo[k])), __equ((__ckobj__.th_ahi[k])), isgreater(T, __ckobj__.th_tmid[k]));
#else
   return isgreater(T, __ckobj__.th_tmid[k]) ? __equ((__ckobj__.th_ahi[k])) : __equ((__ckobj__.th_alo[k]));
#endif

   #undef __equ
}
inline __ValueType FUNC_TYPE(compute_Cp_R) (const int k, const __ValueType T, __ckdata_attr const ckdata_t *ck)
{
   // Cp / R = Sum_(i=1)^(5){ a_i * T^(i-1) }

   //return a[0] + T * (a[1] + T * (a[2] + T * (a[3] + T * a[4])));
   #define __equ(__a) ( __a[0] + T * (__a[1] + T * (__a[2] + T * (__a[3] + T * __a[4]))) )

#if (__ValueSize > 1)
   return select(__equ((__ckobj__.th_alo[k])), __equ((__ckobj__.th_ahi[k])), isgreater(T, __ckobj__.th_tmid[k]));
#else
   return isgreater(T, __ckobj__.th_tmid[k]) ? __equ((__ckobj__.th_ahi[k])) : __equ((__ckobj__.th_alo[k]));
#endif

   #undef __equ
}
// Species S/R - H/RT ... special function.
inline void FUNC_TYPE(cksmh) (const __ValueType T,
                         const __ValueType logT,
                         __global __ValueType *restrict smh,
                         __ckdata_attr const ckdata_t *ck)
{
   const __ValueType logTm1 = logT - 1.;
   const __ValueType invT   = 1. / T;
   const __ValueType T1     = T / 2.;
   const __ValueType T2     = T*T / 6.;
   const __ValueType T3     = T*T*T / 12.;
   const __ValueType T4     = T*T*T*T / 20.;

   #ifdef __INTEL_COMPILER
   #pragma ivdep
   #endif
   for (int k = 0; k < __ckobj__.n_species; ++k)
   {
      #define __equ(__a) ( __a[0] * logTm1 + T1 * __a[1] + T2 * __a[2] + T3 * __a[3] + T4 * __a[4] - __a[5] * invT + __a[6] )

#if (VectorSize > 1)
      smh[__getIndex(k)] = select(__equ((__ckobj__.th_alo[k])), __equ((__ckobj__.th_ahi[k])), isgreater(T, __ckobj__.th_tmid[k]));
#else
      smh[__getIndex(k)] = isgreater(T, __ckobj__.th_tmid[k]) ? __equ((__ckobj__.th_ahi[k])) : __equ((__ckobj__.th_alo[k]));
#endif

      #undef __equ
   }
}
inline void FUNC_TYPE(ckhms) (const __ValueType T, __global __ValueType *restrict h, __ckdata_attr const ckdata_t *ck)
{
   const __ValueType RUT = __RU__ * T;

   #ifdef __INTEL_COMPILER
   #pragma ivdep
   #endif
   for (int k = 0; k < __ckobj__.n_species; ++k)
   {
      __ValueType h_k = FUNC_TYPE(compute_H_RT)(k, T, ck);
      h_k *= (RUT / __ckobj__.sp_mwt[k]);

      h[__getIndex(k)] = h_k;
   }
}
// Mean molecular weight given mass fractions ... g / mol
inline __ValueType FUNC_TYPE(ckmmwy) (__global const __ValueType y[], __ckdata_attr const ckdata_t *ck)
{  
   // <W> = 1 / Sum_k { y_k / w_k }
   __ValueType sumyow = 0.0;
   for (int k = 0; k < __ckobj__.n_species; ++k)
      sumyow += (y[__getIndex(k)] / __ckobj__.sp_mwt[k]);

   return 1.0 / sumyow;
}
// Mixture density given pressure, temperature and mass fractions ... g / cm^3
inline __ValueType FUNC_TYPE(ckrhoy) (const __ValueType p, const __ValueType T, __global const __ValueType y[], __ckdata_attr const ckdata_t *ck)
{
   __ValueType mean_mwt = FUNC_TYPE(ckmmwy)(y, ck);

   // rho = p / (<R> * T) = p / (RU / <W> * T)

   //return p / (T * RU / mean_mwt);
   mean_mwt *= p;
   mean_mwt /= T;
   mean_mwt /= __RU__;

   return mean_mwt;
}
// Mixture Cp in mass units given mass fractions and temperature ... erg / (g * k)
inline __ValueType FUNC_TYPE(ckcpbs) (const __ValueType T, __global const __ValueType y[], __ckdata_attr const ckdata_t *ck)
{
   __ValueType cp_mix = 0.;

   #ifdef __INTEL_COMPILER
   #pragma ivdep
   #endif
   for (int k = 0; k < __ckobj__.n_species; ++k)
   {
      __ValueType cp_k = FUNC_TYPE(compute_Cp_R)(k, T, ck);
      cp_k *= (__RU__ / __ckobj__.sp_mwt[k]);
      cp_k *= y[__getIndex(k)];
      cp_mix += cp_k;
      //cp_mix += (y[k] * cp_k);
   }

   return cp_mix;
}
// Compute the molar concentration given p/T/y_k
inline void FUNC_TYPE(ckytcp) (const __ValueType p, const __ValueType T, __global const __ValueType *restrict y, __global __ValueType *restrict c, __ckdata_attr const ckdata_t *ck)
{
   // [c]_k = rho * y_k / w_k
   const __ValueType rho = FUNC_TYPE(ckrhoy) (p, T, y, ck);

   #ifdef __INTEL_COMPILER
   #pragma ivdep
   #endif
   for (int k = 0; k < __ckobj__.n_species; ++k)
   {
      //c[k] = rho * y[k] / __ckobj__.sp_mwt[k];
      __ValueType c_k = rho * y[__getIndex(k)] / __ckobj__.sp_mwt[k];
      c[__getIndex(k)] = c_k;
   }
}

// Compute temperature-dependent molar forward/reverse reaction rates
// ... utility function ...
inline void FUNC_TYPE(ckratt_) (const __ValueType T,
                     __global __ValueType *restrict smh,
                     __global __ValueType *restrict eqk,
                     __global __ValueType *restrict rkf,
                     __global __ValueType *restrict rkr,
                     __ckdata_attr const ckdata_t *ck)
{
   const int kk = __ckobj__.n_species;
   const int ii = __ckobj__.n_reactions;

   const __ValueType logT = log(T);
   const __ValueType invT = 1.0 / T;
   const __ValueType pfac = __PA__ / (__RU__ * T); // (dyne / cm^2) / (erg / mol / K) / (K)

   // I. Temperature-dependent rates ...

   // S/R - H/RT ... only needed for equilibrium.
   FUNC_TYPE(cksmh) (T, logT, smh, ck);

   #ifdef __INTEL_COMPILER
   #pragma ivdep
   #endif
   for (int i = 0; i < ii; ++i)
   {
      // Basic Arrhenius rates: A * exp( logT * b - E_R / T)
      rkf[__getIndex(i)] = __ckobj__.rx_A[i] * exp(__ckobj__.rx_b[i] * logT - __ckobj__.rx_E[i] * invT);
   }

   #ifdef __INTEL_COMPILER
   #pragma ivdep
   #endif
   for (int i = 0; i < ii; ++i)
   {
      // Irreversible reaction ...
      if (__is_enabled(__ckobj__.rx_info[i], __rx_flag_irrev))
      {
         rkr[__getIndex(i)] = 0.0;
         //eqk[i] = 0.0;
      }
      // Reversible parameters ...
      else if (__is_enabled(__ckobj__.rx_info[i], __rx_flag_rparams))
      {
         rkr[__getIndex(i)] = __ckobj__.rx_rev_A[i] * exp(__ckobj__.rx_rev_b[i] * logT - __ckobj__.rx_rev_E[i] * invT);
         //eqk[i] = rkf[i] / rkr[i];
      }
      // Use equilibrium for reversible rate ...
      else
      {
         // Sum_k { nu_k * (S/R - H/RT)_k }

         #define __nu (__ckobj__.rx_nu[i])
         #define __nuk (__ckobj__.rx_nuk[i])

         __ValueType         sumsmh  = (__nu[0] * smh[__getIndex(__nuk[0])]);
         if (__nuk[1] != -1) sumsmh += (__nu[1] * smh[__getIndex(__nuk[1])]);
         if (__nuk[2] != -1) sumsmh += (__nu[2] * smh[__getIndex(__nuk[2])]);
                             sumsmh += (__nu[3] * smh[__getIndex(__nuk[3])]);
         if (__nuk[4] != -1) sumsmh += (__nu[4] * smh[__getIndex(__nuk[4])]);
         if (__nuk[5] != -1) sumsmh += (__nu[5] * smh[__getIndex(__nuk[5])]);

         #undef __nu
         #undef __nuk

         __ValueType eqk_ = exp(fmin(sumsmh, __exparg__));

         if (__ckobj__.rx_sumnu[i] != 0)
            eqk_ *= FUNC_TYPE(__powi)(pfac,__ckobj__.rx_sumnu[i]);

         rkr[__getIndex(i)] = rkf[__getIndex(i)] / fmax(eqk_,__small__);
      }
   }
}
inline void FUNC_TYPE(ckratc_) (const __ValueType T,
                           __global const __ValueType *restrict c,
                           __global       __ValueType *restrict ctb,
                           __global       __ValueType *restrict rkf,
                           __global       __ValueType *restrict rkr,
                           __ckdata_attr const ckdata_t *ck)
{
   const int kk = __ckobj__.n_species;
   const int ii = __ckobj__.n_reactions;

   const __ValueType logT = log(T);
   const __ValueType invT = 1.0 / T;

   // II. Concentration-dependent rates ...

   for (int i = 0; i < ii; ++i)
      ctb[__getIndex(i)] = 1.0;

   // Third-body reactions ...
   if (__ckobj__.n_thdbdy > 0)
   {
      __ValueType ctot = 0.0;
      for (int k = 0; k < kk; ++k)
         ctot += c[__getIndex(k)];

      for (int n = 0; n < __ckobj__.n_thdbdy; ++n)
      {
         const int rxn_idx = __ckobj__.rx_thdbdy_idx[n];

         ctb[__getIndex(rxn_idx)] = ctot;

         // Add in the specific efficiencies ...

         for (int m = __ckobj__.rx_thdbdy_offset[n]; m < __ckobj__.rx_thdbdy_offset[n+1]; ++m)
         {
            const int k = __ckobj__.rx_thdbdy_spidx[m];
            ctb[__getIndex(rxn_idx)] += (__ckobj__.rx_thdbdy_alpha[m] - 1.0) * c[__getIndex(k)];
         }
      }
   }

   // Fall-off pressure dependencies ...
   if (__ckobj__.n_falloff > 0)
   {
      #ifdef __INTEL_COMPILER
      #pragma ivdep
      #endif
      for (int n = 0; n < __ckobj__.n_falloff; ++n)
      {
         const int rxn_idx = __ckobj__.rx_falloff_idx[n];

         // Concentration of the third-body ... could be a specific species, too.
         __ValueType cthb;
         if (__ckobj__.rx_falloff_spidx[n] != -1)
         {
            cthb = ctb[__getIndex(rxn_idx)];
            ctb[__getIndex(rxn_idx)] = 1.0;
         }
         else
            cthb = c[ __getIndex( __ckobj__.rx_falloff_spidx[n] ) ];

         #define __fpar (__ckobj__.rx_falloff_params[n])

         // Low-pressure limit rate ...
         __ValueType rklow = __fpar[0] * exp(__fpar[1] * logT - __fpar[2] * invT);

         // Reduced pressure ...
         __ValueType pr    = rklow * cthb / rkf[__getIndex(rxn_idx)];

         // Correction ... k_infty (pr / (1+pr)) * F()
         __ValueType p_cor;

         // Different F()'s ...
         //if (__ckobj__.rx_info[rxn_idx] & __rx_flag_falloff_sri)
         //{
         //   printf("SRI fall-off rxn not ready\n");
         //   exit(-1);
         //}
         //else if (__ckobj__.rx_info[rxn_idx] & __rx_flag_falloff_troe)
         if (__is_enabled(__ckobj__.rx_info[rxn_idx], __rx_flag_falloff_troe))
         {
            // 3-parameter Troe form ...
            __ValueType Fcent = (1.0 - __fpar[3]) * exp(-T / __fpar[4]) + __fpar[3] * exp(-T / __fpar[5]);

            // Additional 4th (T**) parameter ...
            if (__is_enabled(__ckobj__.rx_info[rxn_idx], __rx_flag_falloff_troe4))
               Fcent += exp(-__fpar[6] * invT);

            __ValueType log_Fc = log10( fmax(Fcent,__small__) );
            __ValueType eta    = 0.75 - 1.27 * log_Fc;
            __ValueType log_pr = log10( fmax(pr,__small__) );
            __ValueType plus_c = log_pr - (0.4 + 0.67 * log_Fc);
            __ValueType log_F  = log_Fc / (1.0 + FUNC_TYPE(__sqr)(plus_c / (eta - 0.14 * plus_c)));
            __ValueType Fc     = exp10(log_F);

            p_cor = Fc * (pr / (1.0 + pr));
         }
         else // Lindermann form
         {
            p_cor = pr / (1.0 + pr);
         }

         #undef __fpar

         rkf[__getIndex(rxn_idx)] *= p_cor;
         rkr[__getIndex(rxn_idx)] *= p_cor;
         //printf("%3d, %3d, %e, %e\n", n, rxn_idx, __ckobj__.rx_info[rxn_idx], p_cor, cthb);
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
      #define __nu (__ckobj__.rx_nu[i])
      #define __nuk (__ckobj__.rx_nuk[i])

      __ValueType rkf_ = rkf[__getIndex(i)] * ctb[__getIndex(i)];
      __ValueType rkr_ = rkr[__getIndex(i)] * ctb[__getIndex(i)];

                             rkf_ *= FUNC_TYPE(__powu)( c[__getIndex(__nuk[0])],-__nu[0]);
      if (__nuk[1] != -1) {  rkf_ *= FUNC_TYPE(__powu)( c[__getIndex(__nuk[1])],-__nu[1]);
         if (__nuk[2] != -1) rkf_ *= FUNC_TYPE(__powu)( c[__getIndex(__nuk[2])],-__nu[2]);
      }

                             rkr_ *= FUNC_TYPE(__powu)( c[__getIndex(__nuk[3])], __nu[3]);
      if (__nuk[4] != -1) {  rkr_ *= FUNC_TYPE(__powu)( c[__getIndex(__nuk[4])], __nu[4]);
         if (__nuk[5] != -1) rkr_ *= FUNC_TYPE(__powu)( c[__getIndex(__nuk[5])], __nu[5]);
      }

      #undef __nu
      #undef __nuk

      rkf[__getIndex(i)] = rkf_;
      rkr[__getIndex(i)] = rkr_;
   }
}

void FUNC_TYPE(ckwyp) (const __ValueType p,
            const __ValueType T,
            __global const __ValueType *restrict y,
            __global       __ValueType *restrict wdot,
            __ckdata_attr const ckdata_t *ck,
            __global __ValueType rwk[])
{
   const int kk = __ckobj__.n_species;
   const int ii = __ckobj__.n_reactions;

   __global __ValueType *rkf = rwk;
   __global __ValueType *rkr = rkf + __getIndex(ii);
   __global __ValueType *ctb = rkr + __getIndex(ii);
   __global __ValueType *c   = ctb + __getIndex(ii);
   __global __ValueType *smh = c;
   //__ValueType * restrict eqk = ctb;

   // Compute temperature-dependent forward/reverse rates ... mol / cm^3 / s
   //ckratt_ (T, smh, eqk, rkf, rkr, ck);
   FUNC_TYPE(ckratt_) (T, smh, 0, rkf, rkr, ck);

   // Convert to molar concentrations ... mol / cm^3
   FUNC_TYPE(ckytcp) (p, T, y, c, ck);

   // Compute concentration-dependent forward/reverse rates ... mol / cm^3 / s
   FUNC_TYPE(ckratc_) (T, c, ctb, rkf, rkr, ck);

   // Compute species net production rates ... mol / cm^3 / s

   for (int k = 0; k < kk; ++k)
      wdot[__getIndex(k)] = 0.0;

   for (int i = 0; i < ii; ++i)
   {
      const __ValueType rop = rkf[__getIndex(i)] - rkr[__getIndex(i)];

      #define __nu (__ckobj__.rx_nu[i])
      #define __nuk (__ckobj__.rx_nuk[i])

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
// Meta-function to compute RHS for constant-pressure reaction system
void FUNC_TYPE(ckrhs) (const __ValueType p,
                  const __ValueType T,
                  __global const __ValueType y[],
                  __global       __ValueType ydot[],
                  __ckdata_attr const ckdata_t *ck,
                  __global       __ValueType rwk[])
{
   const int kk = __ckobj__.n_species;

   /* Compute molar reaction rate. (mol / (s*cm^3) */
   FUNC_TYPE(ckwyp) (p, T, y, ydot, ck, rwk);

   /* Compute mixture Cp (ergs / gm*K) */
   const __ValueType cp_mix = FUNC_TYPE(ckcpbs)(T, y, ck);

   /* Compute species enthalpy (ergs / K) */
   FUNC_TYPE(ckhms)(T, rwk, ck);

   /* Extract the molecular weights of the species ... this could just be a pointer. */
   const __ValueType rho = FUNC_TYPE(ckrhoy)(p, T, y, ck);

   __ValueType Tdot = 0.0;
   for (int k = 0; k < kk; ++k)
   {
      /* Convert from molar to mass units. */
      //ydot[__getIndex(k)] *= __ckobj__.sp_mwt[k];
      //ydot[__getIndex(k)] /= rho;
      ydot[__getIndex(k)] *= (__ckobj__.sp_mwt[k] / rho);

      /* Sum up the net enthalpy change. */
      Tdot -= (rwk[__getIndex(k)] * ydot[__getIndex(k)]);
   }

   ydot[__getIndex(kk)] = Tdot / cp_mix;

   return;
}

#if 0
void FUNC_SIZE(dcopy_vector_to_scalar) (__global double *dest, __private double *src, const int stride, const int len)
{
#if (__ValueSize == 1)
   *dest = *src;
#else
                    dest[ stride*(  0) ] = (*src).s0;
   if ((  1) < len) dest[ stride*(  1) ] = (*src).s1;
#if (__ValueSize  > 2)
   if ((  2) < len) dest[ stride*(  2) ] = (*src).s2;
   if ((  3) < len) dest[ stride*(  3) ] = (*src).s3;
#if (__ValueSize  > 4)
   if ((  4) < len) dest[ stride*(  4) ] = (*src).s4;
   if ((  5) < len) dest[ stride*(  5) ] = (*src).s5;
   if ((  6) < len) dest[ stride*(  6) ] = (*src).s6;
   if ((  7) < len) dest[ stride*(  7) ] = (*src).s7;
#if (__ValueSize  > 8)
   if ((  8) < len) dest[ stride*(  8) ] = (*src).s8;
   if ((  9) < len) dest[ stride*(  9) ] = (*src).s9;
   if (( 10) < len) dest[ stride*( 10) ] = (*src).sA;
   if (( 11) < len) dest[ stride*( 11) ] = (*src).sB;
   if (( 12) < len) dest[ stride*( 12) ] = (*src).sC;
   if (( 13) < len) dest[ stride*( 13) ] = (*src).sD;
   if (( 14) < len) dest[ stride*( 14) ] = (*src).sE;
   if (( 15) < len) dest[ stride*( 15) ] = (*src).sF;
#endif
#endif
#endif
#endif
}
#endif

#if 1
#warning 'Skipping ck_driver_vec'
#else
void __kernel
//__attribute__((vec_type_hint(__ValueType)))
__attribute__((reqd_work_group_size(__blockSize, 1, 1)))
ck_driver_vec (const double p,
           const double T,
           __global const double *u,
           __global       double *udot,
           __ckdata_attr const ckdata_t *ck,
           __global       __ValueType *rwk,
           const int num_evals)
{
   const int tid = get_global_id(0);
// const int gid = get_group_id(0);
// const int blockSize = get_local_size(0); // == __blockSize
// const int num_threads = get_global_size(0);

   const int kk = __ckobj__.n_species;
   const int neq = kk+1;
   const int lenrwk = ck_lenrwk(ck);

   // Thread-local pointers ...
#if defined(__Alignment) && (__Alignment > 0)
   rwk = (__global __ValueType *) align_pointer(rwk, sizeof(__ValueType));
#endif
   __global __ValueType *my_rwk = rwk +
                ((tid / __arrayStride) * (__getIndex(lenrwk+2*neq)) + (tid % __arrayStride));
   __global __ValueType *my_u = my_rwk + (__getIndex(lenrwk));
   __global __ValueType *my_udot = my_u + (__getIndex(neq));

   const int nelems = vec_step(__ValueType);

   for (int i = nelems * get_global_id(0); i < num_evals; i += nelems * get_global_size(0))
   {
      //for (int k = 0; k < nelems; ++k)
      //   my_u[__getIndex(k)] = T + (100.*(i+k)) / num_evals;

      __ValueType p0 = p;

#if   (__ValueSize == 1)
      __ValueType T0 = 0;
#elif (__ValueSize == 2)
      __ValueType T0 = {0,1};
#elif (__ValueSize == 4)
      __ValueType T0 = {0,1,2,3};
#elif (__ValueSize == 8)
      __ValueType T0 = {0,1,2,3,4,5,6,7};
#elif (__ValueSize == 16)
      __ValueType T0 = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
#endif

      T0 = T + (1000. * (i + T0)) / num_evals;

      for (int k = 0; k < neq; ++k)
         my_u[__getIndex(k)] = (__ValueType)u[ k ];

      //FUNC_TYPE(ckhms) (T0, my_udot, ck);
      //my_udot[__getIndex(0)] = FUNC_TYPE(ckrhoy) (p0, T0, my_u, ck);
      //FUNC_TYPE(ckwyp) (p0, T0, my_u, my_udot, ck, my_rwk);
      FUNC_TYPE(ckrhs) (p0, T0, my_u, my_udot, ck, my_rwk);

      for (int k = 0; k < neq; ++k)
      {
#if (__ValueSize == 1)
                               udot[ neq*(i) + k ] = my_udot[__getIndex(k)];
#else
                               udot[ neq*(i+ 0) + k ] = my_udot[__getIndex(k)].s0;
         if ((i+ 1)<num_evals) udot[ neq*(i+ 1) + k ] = my_udot[__getIndex(k)].s1;
#if (__ValueSize > 2)
         if ((i+ 2)<num_evals) udot[ neq*(i+ 2) + k ] = my_udot[__getIndex(k)].s2;
         if ((i+ 3)<num_evals) udot[ neq*(i+ 3) + k ] = my_udot[__getIndex(k)].s3;
#if (__ValueSize > 4)
         if ((i+ 4)<num_evals) udot[ neq*(i+ 4) + k ] = my_udot[__getIndex(k)].s4;
         if ((i+ 5)<num_evals) udot[ neq*(i+ 5) + k ] = my_udot[__getIndex(k)].s5;
         if ((i+ 6)<num_evals) udot[ neq*(i+ 6) + k ] = my_udot[__getIndex(k)].s6;
         if ((i+ 7)<num_evals) udot[ neq*(i+ 7) + k ] = my_udot[__getIndex(k)].s7;
#if (__ValueSize > 8)
         if ((i+ 8)<num_evals) udot[ neq*(i+ 8) + k ] = my_udot[__getIndex(k)].s8;
         if ((i+ 9)<num_evals) udot[ neq*(i+ 9) + k ] = my_udot[__getIndex(k)].s9;
         if ((i+10)<num_evals) udot[ neq*(i+10) + k ] = my_udot[__getIndex(k)].sA;
         if ((i+11)<num_evals) udot[ neq*(i+11) + k ] = my_udot[__getIndex(k)].sB;
         if ((i+12)<num_evals) udot[ neq*(i+12) + k ] = my_udot[__getIndex(k)].sC;
         if ((i+13)<num_evals) udot[ neq*(i+13) + k ] = my_udot[__getIndex(k)].sD;
         if ((i+14)<num_evals) udot[ neq*(i+14) + k ] = my_udot[__getIndex(k)].sE;
         if ((i+15)<num_evals) udot[ neq*(i+15) + k ] = my_udot[__getIndex(k)].sF;
#endif
#endif
#endif
#endif
      }
   }
}
#endif

#if 1
#warning 'Skipping ck_driver_vec_array'
#else

void __kernel
//__attribute__((vec_type_hint(__ValueType)))
__attribute__((reqd_work_group_size(__blockSize, 1, 1)))
ck_driver_vec_array
          (__global const double *p,
           __global const double *T,
           __global const double *u,
           __global       double *udot,
           __ckdata_attr const ckdata_t *ck,
           __global       __ValueType *rwk,
           const int num_evals)
{
   const int tid = get_global_id(0);
// const int gid = get_group_id(0);
// const int blockSize = get_local_size(0); // == __blockSize
// const int num_threads = get_global_size(0);

   const int kk = __ckobj__.n_species;
   const int neq = kk+1;
   const int lenrwk = ck_lenrwk(ck);

   // Thread-local pointers ...
#if defined(__Alignment) && (__Alignment > 0)
   rwk = (__global __ValueType *) align_pointer(rwk, sizeof(__ValueType));
#endif
   __global __ValueType *my_rwk = rwk +
                ((tid / __arrayStride) * (__getIndex(lenrwk+2*neq)) + (tid % __arrayStride));
   __global __ValueType *my_u = my_rwk + (__getIndex(lenrwk));
   __global __ValueType *my_udot = my_u + (__getIndex(neq));

   const int nelems = vec_step(__ValueType);

   for (int i = nelems * get_global_id(0); i < num_evals; i += nelems * get_global_size(0))
   {
      __ValueType my_p, my_T;
      //__ValueType my_p = (__ValueType) p[i];
      //__ValueType my_T = (__ValueType) T[i];
#if (__ValueSize == 1)
         my_p = p[i]; my_T = T[i];
#else
                               { my_p.s0 = p[i   ]; my_T.s0 = T[i   ]; }
         if ((i+ 1)<num_evals) { my_p.s1 = p[i+ 1]; my_T.s1 = T[i+ 1]; }
#if (__ValueSize > 2)
         if ((i+ 2)<num_evals) { my_p.s2 = p[i+ 2]; my_T.s2 = T[i+ 2]; }
         if ((i+ 3)<num_evals) { my_p.s3 = p[i+ 3]; my_T.s3 = T[i+ 3]; }
#if (__ValueSize > 4)
         if ((i+ 4)<num_evals) { my_p.s4 = p[i+ 4]; my_T.s4 = T[i+ 4]; }
         if ((i+ 5)<num_evals) { my_p.s5 = p[i+ 5]; my_T.s5 = T[i+ 5]; }
         if ((i+ 6)<num_evals) { my_p.s6 = p[i+ 6]; my_T.s6 = T[i+ 6]; }
         if ((i+ 7)<num_evals) { my_p.s7 = p[i+ 7]; my_T.s7 = T[i+ 7]; }
#if (__ValueSize > 8)
         if ((i+ 8)<num_evals) { my_p.s8 = p[i+ 8]; my_T.s8 = T[i+ 8]; }
         if ((i+ 9)<num_evals) { my_p.s9 = p[i+ 9]; my_T.s9 = T[i+ 9]; }
         if ((i+10)<num_evals) { my_p.sA = p[i+10]; my_T.sA = T[i+10]; }
         if ((i+11)<num_evals) { my_p.sB = p[i+11]; my_T.sB = T[i+11]; }
         if ((i+12)<num_evals) { my_p.sC = p[i+12]; my_T.sC = T[i+12]; }
         if ((i+13)<num_evals) { my_p.sD = p[i+13]; my_T.sD = T[i+13]; }
         if ((i+14)<num_evals) { my_p.sE = p[i+14]; my_T.sE = T[i+14]; }
         if ((i+15)<num_evals) { my_p.sF = p[i+15]; my_T.sF = T[i+15]; }
#endif
#endif
#endif
#endif

      for (int k = 0; k < kk; ++k)
         my_u[__getIndex(k)] = (__ValueType)u[ i*kk + k ];

      //FUNC_TYPE(ckrhs) (p[i], T[i], my_u, my_udot, ck, my_rwk);
      FUNC_TYPE(ckrhs) (my_p, my_T, my_u, my_udot, ck, my_rwk);

      for (int k = 0; k < neq; ++k)
      {
#if (__ValueSize == 1)
                               udot[ neq*(i) + k ] = my_udot[__getIndex(k)];
#else
                               udot[ neq*(i+ 0) + k ] = my_udot[__getIndex(k)].s0;
         if ((i+ 1)<num_evals) udot[ neq*(i+ 1) + k ] = my_udot[__getIndex(k)].s1;
#if (__ValueSize > 2)
         if ((i+ 2)<num_evals) udot[ neq*(i+ 2) + k ] = my_udot[__getIndex(k)].s2;
         if ((i+ 3)<num_evals) udot[ neq*(i+ 3) + k ] = my_udot[__getIndex(k)].s3;
#if (__ValueSize > 4)
         if ((i+ 4)<num_evals) udot[ neq*(i+ 4) + k ] = my_udot[__getIndex(k)].s4;
         if ((i+ 5)<num_evals) udot[ neq*(i+ 5) + k ] = my_udot[__getIndex(k)].s5;
         if ((i+ 6)<num_evals) udot[ neq*(i+ 6) + k ] = my_udot[__getIndex(k)].s6;
         if ((i+ 7)<num_evals) udot[ neq*(i+ 7) + k ] = my_udot[__getIndex(k)].s7;
#if (__ValueSize > 8)
         if ((i+ 8)<num_evals) udot[ neq*(i+ 8) + k ] = my_udot[__getIndex(k)].s8;
         if ((i+ 9)<num_evals) udot[ neq*(i+ 9) + k ] = my_udot[__getIndex(k)].s9;
         if ((i+10)<num_evals) udot[ neq*(i+10) + k ] = my_udot[__getIndex(k)].sA;
         if ((i+11)<num_evals) udot[ neq*(i+11) + k ] = my_udot[__getIndex(k)].sB;
         if ((i+12)<num_evals) udot[ neq*(i+12) + k ] = my_udot[__getIndex(k)].sC;
         if ((i+13)<num_evals) udot[ neq*(i+13) + k ] = my_udot[__getIndex(k)].sD;
         if ((i+14)<num_evals) udot[ neq*(i+14) + k ] = my_udot[__getIndex(k)].sE;
         if ((i+15)<num_evals) udot[ neq*(i+15) + k ] = my_udot[__getIndex(k)].sF;
#endif
#endif
#endif
#endif
      }
   }
}
#endif

typedef struct //_rk_callback_s
{
   __ValueType p;
   __global __ValueType *rwk;

   __ckdata_attr const ckdata_t *ck;
}
FUNC_SIZE(cklib_callback_t);

int FUNC_TYPE(cklib_callback) (const int neq, const __ValueType time, __global __ValueType y[], __global __ValueType ydot[], __private void *vptr)
{
   FUNC_SIZE(cklib_callback_t) *user_data = (FUNC_SIZE(cklib_callback_t) *) vptr;

   //FUNC_TYPE(ckrhs) (user_data->p, y[neq-1], y, ydot, user_data->ck, user_data->rwk);
   FUNC_TYPE(ckrhs) (user_data->p, y[__getIndex(neq-1)], y, ydot, user_data->ck, user_data->rwk);

   return 0;
}

typedef struct //_rk_callback_s
{
   double p;
   __global double *rwk;

   __ckdata_attr const ckdata_t *ck;
}
cklib_callback_t;
int cklib_callback (const int neq, const double time, __global double y[], __global double ydot[], __private void *vptr)
{
   cklib_callback_t *user_data = (cklib_callback_t *) vptr;

   ckrhs (user_data->p, y[__getIndex(neq-1)], y, ydot, user_data->ck, user_data->rwk);

   return 0;
}

// Single-step function
int FUNC_TYPE(rkf45) (const int neq, const __ValueType h, __global __ValueType* y, __global __ValueType* y_out, __global __ValueType* rwk, __private void *user_data)
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
   __global __ValueType* f1   = rwk ;
   __global __ValueType* f2   = rwk +   neq ;
   __global __ValueType* f3   = rwk + 2*neq ;
   __global __ValueType* f4   = rwk + 3*neq ;
   __global __ValueType* f5   = rwk + 4*neq ;
   __global __ValueType* f6   = rwk + 5*neq ;
   __global __ValueType* ytmp = rwk + 6*neq ;

   // 1)
   FUNC_TYPE(cklib_callback)(neq, 0.0, y, f1, user_data);

   for (int k = 0; k < neq; k++)
   {
      //f1[k] = h * ydot[k];
      f1[__getIndex(k)] *= h;
      ytmp[__getIndex(k)] = y[__getIndex(k)] + c21 * f1[__getIndex(k)];
   }

   // 2)
   FUNC_TYPE(cklib_callback)(neq, 0.0, ytmp, f2, user_data);

   for (int k = 0; k < neq; k++)
   {
      //f2[k] = h * ydot[k];
      f2[__getIndex(k)] *= h;
      ytmp[__getIndex(k)] = y[__getIndex(k)] + c31 * f1[__getIndex(k)] + c32 * f2[__getIndex(k)];
   }

   // 3)
   FUNC_TYPE(cklib_callback)(neq, 0.0, ytmp, f3, user_data);

   for (int k = 0; k < neq; k++) {
      //f3[k] = h * ydot[k];
      f3[__getIndex(k)] *= h;
      ytmp[__getIndex(k)] = y[__getIndex(k)] + c41 * f1[__getIndex(k)] + c42 * f2[__getIndex(k)] + c43 * f3[__getIndex(k)];
   }

   // 4)
   FUNC_TYPE(cklib_callback)(neq, 0.0, ytmp, f4, user_data);

   for (int k = 0; k < neq; k++) {
      //f4[k] = h * ydot[k];
      f4[__getIndex(k)] *= h;
      ytmp[__getIndex(k)] = y[__getIndex(k)] + c51 * f1[__getIndex(k)] + c52 * f2[__getIndex(k)] + c53 * f3[__getIndex(k)] + c54 * f4[__getIndex(k)];
   }

   // 5)
   FUNC_TYPE(cklib_callback)(neq, 0.0, ytmp, f5, user_data);

   for (int k = 0; k < neq; k++) {
      //f5[k] = h * ydot[k];
      f5[__getIndex(k)] *= h;
      ytmp[__getIndex(k)] = y[__getIndex(k)] + c61*f1[__getIndex(k)] + c62*f2[__getIndex(k)] + c63*f3[__getIndex(k)] + c64*f4[__getIndex(k)] + c65*f5[__getIndex(k)];
   }

   // 6)
   FUNC_TYPE(cklib_callback)(neq, 0.0, ytmp, f6, user_data);

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
   const int neq = rk->neq;
   __ValueType sum = 0;
   for (int k = 0; k < neq; k++)
   {
      __ValueType ewt = (rk->s_rtol * fabs(y[__getIndex(k)])) + rk->s_atol;
      __ValueType prod = x[__getIndex(k)] / ewt;
      sum += (prod*prod);
   }

   return sqrt(sum / (__ValueType)neq);
}

#ifndef __any
  #if (__ValueSize == 1)
    #define __any(__val) (__val)
  #else
    #define __any(__val) (any(__val))
  #endif
#endif
#ifndef __all
  #if (__ValueSize == 1)
    #define __all(__val) (__val)
  #else
    #define __all(__val) (all(__val))
  #endif
#endif
#ifndef __select
  #if (__ValueSize == 1)
    #define __select(__is_false, __is_true, __cmp) ( (__cmp) ? (__is_true) : (__is_false) )
  #else
    #define __select(__is_false, __is_true, __cmp) (select((__is_false), (__is_true), (__cmp)))
  #endif
#endif
#ifndef __not
  #define __not(__val) ( !(__val) )
#endif

// Scalar helper functions for the pivot operation -- need a Vector version here.
#if (__ValueSize == 1)
  #define __read_from(__src, __lane, __dest) { (__dest) = (__src); }
  #define __write_to(__src, __lane, __dest) { (__dest) = (__src); }
#elif (__ValueSize == 2)
  #define __read_from(__src, __lane, __dest) \
  { \
     if      ((__lane) ==  0) (__dest) = (__src).s0; \
     else if ((__lane) ==  1) (__dest) = (__src).s1; \
  }
  #define __write_to(__src, __lane, __dest) \
  { \
     if      ((__lane) ==  0) (__dest).s0 = (__src); \
     else if ((__lane) ==  1) (__dest).s1 = (__src); \
  }
#elif (__ValueSize == 3)
  #define __read_from(__src, __lane, __dest) \
  { \
     if      ((__lane) ==  0) (__dest) = (__src).s0; \
     else if ((__lane) ==  1) (__dest) = (__src).s1; \
     else if ((__lane) ==  2) (__dest) = (__src).s2; \
  }
  #define __write_to(__src, __lane, __dest) \
  { \
     if      ((__lane) ==  0) (__dest).s0 = (__src); \
     else if ((__lane) ==  1) (__dest).s1 = (__src); \
     else if ((__lane) ==  2) (__dest).s2 = (__src); \
  }
#elif (__ValueSize == 4)
  #define __read_from(__src, __lane, __dest) \
  { \
     if      ((__lane) ==  0) (__dest) = (__src).s0; \
     else if ((__lane) ==  1) (__dest) = (__src).s1; \
     else if ((__lane) ==  2) (__dest) = (__src).s2; \
     else if ((__lane) ==  3) (__dest) = (__src).s3; \
  }
  #define __write_to(__src, __lane, __dest) \
  { \
     if      ((__lane) ==  0) (__dest).s0 = (__src); \
     else if ((__lane) ==  1) (__dest).s1 = (__src); \
     else if ((__lane) ==  2) (__dest).s2 = (__src); \
     else if ((__lane) ==  3) (__dest).s3 = (__src); \
  }
#elif (__ValueSize == 8)
  #define __read_from(__src, __lane, __dest) \
  { \
     if      ((__lane) ==  0) (__dest) = (__src).s0; \
     else if ((__lane) ==  1) (__dest) = (__src).s1; \
     else if ((__lane) ==  2) (__dest) = (__src).s2; \
     else if ((__lane) ==  3) (__dest) = (__src).s3; \
     else if ((__lane) ==  4) (__dest) = (__src).s4; \
     else if ((__lane) ==  5) (__dest) = (__src).s5; \
     else if ((__lane) ==  6) (__dest) = (__src).s6; \
     else if ((__lane) ==  7) (__dest) = (__src).s7; \
  }
  #define __write_to(__src, __lane, __dest) \
  { \
     if      ((__lane) ==  0) (__dest).s0 = (__src); \
     else if ((__lane) ==  1) (__dest).s1 = (__src); \
     else if ((__lane) ==  2) (__dest).s2 = (__src); \
     else if ((__lane) ==  3) (__dest).s3 = (__src); \
     else if ((__lane) ==  4) (__dest).s4 = (__src); \
     else if ((__lane) ==  5) (__dest).s5 = (__src); \
     else if ((__lane) ==  6) (__dest).s6 = (__src); \
     else if ((__lane) ==  7) (__dest).s7 = (__src); \
  }
#elif (__ValueSize == 16)
  #define __read_from(__src, __lane, __dest) \
  { \
     if      ((__lane) ==  0) (__dest) = (__src).s0; \
     else if ((__lane) ==  1) (__dest) = (__src).s1; \
     else if ((__lane) ==  2) (__dest) = (__src).s2; \
     else if ((__lane) ==  3) (__dest) = (__src).s3; \
     else if ((__lane) ==  4) (__dest) = (__src).s4; \
     else if ((__lane) ==  5) (__dest) = (__src).s5; \
     else if ((__lane) ==  6) (__dest) = (__src).s6; \
     else if ((__lane) ==  7) (__dest) = (__src).s7; \
     else if ((__lane) ==  8) (__dest) = (__src).s8; \
     else if ((__lane) ==  9) (__dest) = (__src).s9; \
     else if ((__lane) == 10) (__dest) = (__src).sA; \
     else if ((__lane) == 11) (__dest) = (__src).sB; \
     else if ((__lane) == 12) (__dest) = (__src).sC; \
     else if ((__lane) == 13) (__dest) = (__src).sD; \
     else if ((__lane) == 14) (__dest) = (__src).sE; \
     else if ((__lane) == 15) (__dest) = (__src).sF; \
  }
  #define __write_to(__src, __lane, __dest) \
  { \
     if      ((__lane) ==  0) (__dest).s0 = (__src); \
     else if ((__lane) ==  1) (__dest).s1 = (__src); \
     else if ((__lane) ==  2) (__dest).s2 = (__src); \
     else if ((__lane) ==  3) (__dest).s3 = (__src); \
     else if ((__lane) ==  4) (__dest).s4 = (__src); \
     else if ((__lane) ==  5) (__dest).s5 = (__src); \
     else if ((__lane) ==  6) (__dest).s6 = (__src); \
     else if ((__lane) ==  7) (__dest).s7 = (__src); \
     else if ((__lane) ==  8) (__dest).s8 = (__src); \
     else if ((__lane) ==  9) (__dest).s9 = (__src); \
     else if ((__lane) == 10) (__dest).sA = (__src); \
     else if ((__lane) == 11) (__dest).sB = (__src); \
     else if ((__lane) == 12) (__dest).sC = (__src); \
     else if ((__lane) == 13) (__dest).sD = (__src); \
     else if ((__lane) == 14) (__dest).sE = (__src); \
     else if ((__lane) == 15) (__dest).sF = (__src); \
  }
#endif

#if (__ValueSize == 1)
  #define __vload(__offset, __ptr) ( *(__ptr) )
#else
  #define __vload(__offset, __ptr) ( PASTE(vload,__ValueSize)((__offset), (__ptr)) )
#endif


int FUNC_TYPE(rk_hin) (__global const rk_t *rk, const __ValueType t, __ValueType *h0, __global __ValueType* y, __global __ValueType *rwk, __private void *user_data)
//int FUNC_TYPE(rk_hin) (const int neq, const double h_min, const double h_max, const __ValueType t, __ValueType *h0, __global __ValueType* y, __global __ValueType *rwk, __private void *user_data)
{
   const int neq = rk->neq;

   __global __ValueType *ydot  = rwk;
   __global __ValueType *y1    = ydot + neq;
   __global __ValueType *ydot1 = y1 + neq;

   double hlb = rk->h_min;
   double hub = rk->h_max;
   //double hlb = h_min;
   //double hub = h_max;

   // Alread done ...
   __MaskType done = isgreaterequal(*h0, rk->h_min);
   //__MaskType done = isgreaterequal(*h0, h_min);

   __ValueType hg = sqrt(hlb*hub);

   if (hub < hlb)
   {
      *h0 = __select(hg, *h0, done);

      return RK_SUCCESS;
   }

   //if (hub < hlb)
   //{
   //   *h0 = hg;
   //   return RK_SUCCESS;
   //}

   // Start iteration to find solution to ... {WRMS norm of (h0^2 y'' / 2)} = 1

   __MaskType hnew_is_ok = 0;
   __ValueType hnew = hg;
   const int miters = 10;
   int iter = 0;
   int ierr = RK_SUCCESS;

   // compute ydot at t=t0
   FUNC_TYPE(cklib_callback)(neq, 0.0, y, ydot, user_data);

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
      FUNC_TYPE(cklib_callback) (neq, 0.0, y1, ydot1, user_data);

      // Compute WRMS norm of y''
      #ifdef __INTEL_COMPILER
      #pragma ivdep
      #endif
      for (int k = 0; k < neq; k++)
         y1[__getIndex(k)] = (ydot1[__getIndex(k)] - ydot[__getIndex(k)]) / hg;

      __ValueType yddnrm = FUNC_TYPE(rk_wnorm) (rk, y1, y);

      //std::cout << "iter " << iter << " hg " << hg << " y'' " << yddnrm << std::endl;
      //std::cout << "ydot " << ydot[neq-1] << std::endl;

      // should we accept this?
      //if (hnew_is_ok || iter == miters)
      //{
      //   hnew = hg;
      //   //if (iter == miters) fprintf(stderr, "ERROR_HIN_MAX_ITERS\n");
      //   ierr = (hnew_is_ok) ? RK_SUCCESS : RK_HIN_MAX_ITERS;
      //   break;
      //}
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
      //if ( (hrat > 0.5) && (hrat < 2.0) )
      //   hnew_is_ok = 1;
      hnew_is_ok = isgreater(hrat, 0.5) & isless(hrat, 2.0);

      // If y'' is still bad after a few iterations, just accept h and give up.
      //if ( (iter > 1) && (hrat > 2.0) ) {
      //   hnew = hg;
      //   hnew_is_ok = 1;
      //}
      if (iter > 1)
      {
         hnew_is_ok = isgreater(hrat, 2.0);
         hnew = __select (hnew, hg, hnew_is_ok);
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
typedef struct
{
   int niters;
   __MaskType nsteps;
}
FUNC_SIZE(rk_counters_t);

int FUNC_TYPE(rk_solve) (__global const rk_t *rk, __ValueType *tcur, __ValueType *hcur, __private FUNC_SIZE(rk_counters_t) *counters, __global __ValueType y[], __global __ValueType rwk[], __private void *user_data)
{
   const int neq = rk->neq;

   int ierr = RK_SUCCESS;

      //printf("h = %e %e %e\n", *hnext, rk->h_min, rk->h_max);
   // Estimate the initial step size ...
   {
      __MaskType test = isless(*hcur, rk->h_min);
      if (__any(test))
      {
         ierr = FUNC_TYPE(rk_hin) (rk, *tcur, hcur, y, rwk, user_data);
         //ierr = FUNC_TYPE(rk_hin) (rk->neq, rk->h_min, rk->h_max, *tcur, hcur, y, rwk, user_data);
         //if (ierr != RK_SUCCESS)
         //   return ierr;
      }
/*#if (__ValueSize == 1)
      printf("hin = %e %e %e\n", *hcur, rk->h_min, rk->h_max);
#else
      printf("hin = %v"STRINGIFY(__ValueSize)"e %e %e\n", *hcur, rk->h_min, rk->h_max);
#endif*/
   }

   #define t (*tcur)
   #define h (*hcur)
   #define iter (counters->niters)
   #define nst (counters->nsteps)

   nst = 0;
   iter = 0;

   __MaskType done = isless(fabs(t - rk->t_stop), rk->t_round);

   while (__any(__not(done)))
   {
      __global __ValueType *ytmp = rwk + neq*7;

      // Take a trial step over h_cur ...
      FUNC_TYPE(rkf45) (neq, h, y, ytmp, rwk, user_data);

      __ValueType herr = fmax(1.0e-20, FUNC_TYPE(rk_wnorm) (rk, rwk, y));

      // Is there error acceptable?
      __MaskType accept = islessequal(herr, 1.0);
      accept |= islessequal(h, rk->h_min);
      accept &= __not(done);

      // update solution ...
      if (__any(accept))
      {
         t   = __select (t,   t + h  , accept);
         nst = __select (nst, nst + 1, accept);

         for (int k = 0; k < neq; k++)
            y[__getIndex(k)] = __select(y[__getIndex(k)], ytmp[__getIndex(k)], accept);

         done = isless( fabs(t - rk->t_stop), rk->t_round);
      }

      __ValueType fact = sqrt( sqrt(1.0 / herr) ) * (0.840896415);

      // Restrict the rate of change in dt
      fact = fmax(fact, 1.0 / rk->adaption_limit);
      fact = fmin(fact,       rk->adaption_limit);

#if 0
      if (iter % 100 == 0)
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
      h = fmin(h, rk->h_max);
      h = fmax(h, rk->h_min);

      // Stretch the final step if we're really close and we didn't just fail ...
      h = __select(h, rk->t_stop - t, accept & isless(fabs((t + h) - rk->t_stop), rk->h_min));

      // Don't overshoot the final time ...
      h = __select(h, rk->t_stop - t, __not(done) & isgreater((t + h),  rk->t_stop));

      ++iter;
      if (rk->max_iters && iter > rk->max_iters) {
         ierr = RK_TOO_MUCH_WORK;
         //printf("(iter > max_iters)\n");
         break;
      }
   }

   return ierr;

   #undef t
   #undef h
   #undef iter
   #undef nst
}

//#if 1
#ifdef __EnableQueue
#warning 'Skipping rk_driver kernel'
#else

void __kernel
__attribute__((reqd_work_group_size(__blockSize, 1, 1)))
rk_driver (const double p,
            __global const double *u_in,
            __global       double *u_out,
            __ckdata_attr const ckdata_t *ck,
            __global      const rk_t *rk,
            //__global       int *iwk,
            __global       double *rwk,
            __global rk_counters_t *rk_counters,
            const int numProblems)
{
   const int tid = get_global_id(0);

   const int kk = __ckobj__.n_species;
   const int neq = kk+1;
   const int lenrwk_ck = ck_lenrwk(ck);
   const int lenrwk_rk = rk_lenrwk(rk);
//   const int leniwk_rk = rk_leniwk(rk);

   // Thread-local pointers ...
   __global double *my_rwk = rwk +
                ((tid / __arrayStride) * (__getIndex(lenrwk_ck + lenrwk_rk + neq)) + (tid % __arrayStride));
   __global double *my_rwk_rk = my_rwk + (__getIndex(lenrwk_ck));
   __global double *my_u = my_rwk_rk + (__getIndex(lenrwk_rk));
//   __global int *my_iwk_rk = iwk +
//                ((tid / __arrayStride) * (__getIndex(leniwk_rk)) + (tid % __arrayStride));

   __private cklib_callback_t my_callback = { .p = p, .rwk = my_rwk, .ck = ck };
   __private rk_counters_t my_counters;

   for (int i = get_global_id(0); i < numProblems; i += get_global_size(0))
   {
      for (int k = 0; k < neq; ++k)
         my_u[__getIndex(k)] = u_in[i*neq+ k ];

      //for (int k = 0; k < neq; ++k)
      //   printf("u0[%d] = %e\n", k, my_u[__getIndex(k)]);
         //printf("T0[%d] = %e\n", i, my_u[__getIndex(kk)]);

      double t = 0, h = 0;//1e-6;

      int rkerr = rk_solve (rk, &t, &h, &my_counters, my_u, my_rwk_rk, (void *)0, &my_callback);

      //for (int k = 0; k < neq; ++k)
      //   printf("u1[%d] = %e %d %d\n", k, my_u[__getIndex(k)], my_counters.nsteps, my_counters.niters);
         //printf("T1[%d] = %e %d %d\n", i, my_u[__getIndex(kk)], my_counters.nsteps, my_counters.niters);

      for (int k = 0; k < neq; ++k)
      {
         u_out[ neq*(i    ) + k ] = my_u[__getIndex(k)];
      }

      rk_counters[i].niters = my_counters.niters;
      rk_counters[i].nsteps = my_counters.nsteps;
      if (rkerr != RK_SUCCESS)
         rk_counters[i].niters = rkerr;
   }
}
#endif

#ifndef __EnableQueue
#warning 'Skipping rk_driver_queue kernel'
#else

void __kernel
__attribute__((reqd_work_group_size(__blockSize, 1, 1)))
rk_driver_queue (const double p,
            __global const double *u_in,
            __global       double *u_out,
            __ckdata_attr const ckdata_t *ck,
            __global      const rk_t *rk,
            //__global       int *iwk,
            __global       double *rwk,
            __global rk_counters_t *rk_counters,
            const int numProblems,
            __global int *problemCounter)
{
   const int tid = get_global_id(0);

   const int kk = __ckobj__.n_species;
   const int neq = kk+1;
   const int lenrwk_ck = ck_lenrwk(ck);
   const int lenrwk_rk = rk_lenrwk(rk);
//   const int leniwk_rk = rk_leniwk(rk);

   // Thread-local pointers ...
   __global double *my_rwk = rwk +
                ((tid / __arrayStride) * (__getIndex(lenrwk_ck + lenrwk_rk + neq)) + (tid % __arrayStride));
   __global double *my_rwk_rk = my_rwk + (__getIndex(lenrwk_ck));
   __global double *my_u = my_rwk_rk + (__getIndex(lenrwk_rk));
//   __global int *my_iwk_rk = iwk +
//                ((tid / __arrayStride) * (__getIndex(leniwk_rk)) + (tid % __arrayStride));

   __private cklib_callback_t my_callback = { .p = p, .rwk = my_rwk, .ck = ck };
   __private rk_counters_t my_counters;

   __private int problem_idx;

   // Initial problem set and global counter.
   problem_idx = get_global_id(0);

   if (get_local_id(0) == 0)
      atomic_add( problemCounter, get_local_size(0));

   barrier(CLK_GLOBAL_MEM_FENCE);

   //while ((problem_idx = atomic_inc(problemCounter)) < numProblems)
   while (problem_idx < numProblems)
   {
      for (int k = 0; k < neq; ++k)
         my_u[__getIndex(k)] = u_in[problem_idx*neq+ k ];

      //for (int k = 0; k < neq; ++k)
      //   printf("u0[%d] = %e\n", k, my_u[__getIndex(k)]);
         //printf("T0[%d] = %e\n", i, my_u[__getIndex(kk)]);

      double t = 0, h = 0;//1e-6;

      int rkerr = rk_solve (rk, &t, &h, &my_counters, my_u, my_rwk_rk, (void *)0, &my_callback);

      //for (int k = 0; k < neq; ++k)
      //   printf("u1[%d] = %e %d %d\n", k, my_u[__getIndex(k)], my_counters.nsteps, my_counters.niters);
         //printf("T1[%d] = %e %d %d\n", i, my_u[__getIndex(kk)], my_counters.nsteps, my_counters.niters);

      for (int k = 0; k < neq; ++k)
      {
         u_out[ neq*(problem_idx    ) + k ] = my_u[__getIndex(k)];
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

//#if 1
#ifdef __EnableQueue
#warning 'Skipping rk_driver_vec kernel'
#else

void __kernel
//__attribute__((vec_type_hint(__ValueType)))
__attribute__((reqd_work_group_size(__blockSize, 1, 1)))
rk_driver_vec (const double p,
               __global const double *u_in,
               __global       double *u_out,
               __ckdata_attr const ckdata_t *ck,
               __global      const rk_t *rk,
               __global       __ValueType *rwk,
               __global rk_counters_t *rk_counters,
               const int numProblems)
{
   const int tid = get_global_id(0);

   const int kk = __ckobj__.n_species;
   const int neq = kk+1;
   const int lenrwk_ck = ck_lenrwk(ck);
   const int lenrwk_rk = rk_lenrwk(rk);

   // Thread-local pointers ...
   __global __ValueType *my_rwk = rwk +
                ((tid / __arrayStride) * (__getIndex(lenrwk_ck + lenrwk_rk + neq)) + (tid % __arrayStride));
   __global __ValueType *my_rwk_rk = my_rwk + (__getIndex(lenrwk_ck));
   __global __ValueType *my_u = my_rwk_rk + (__getIndex(lenrwk_rk));

   const int nelems = vec_step(__ValueType);

   //__private FUNC_SIZE(cklib_callback_t) my_callback = { .p = p, .rwk = my_rwk, .ck = ck };
   __private FUNC_SIZE(cklib_callback_t) my_callback; {my_callback.p = p; my_callback.rwk = my_rwk; my_callback.ck = ck;}
   __private FUNC_SIZE(rk_counters_t) my_counters;

   for (int i = nelems * get_global_id(0); i < numProblems; i += nelems * get_global_size(0))
   {
      //for (int k = 0; k < neq; ++k)
      //   my_u[__getIndex(k)] = u_in[i*neq+ k ];
      for (int k = 0; k < neq; ++k)
      {
         for (int lane = 0; lane < __ValueSize; ++lane)
         {
            const int problem_id = min(i + lane, numProblems-1);
            //if (problem_id < numProblems)
               __write_to(u_in[problem_id * neq + k], lane, my_u[__getIndex(k)]);
         }
      }

      __ValueType t = 0, h = 0;//1e-6;

      FUNC_TYPE(rk_solve) (rk, &t, &h, &my_counters, my_u, my_rwk_rk, &my_callback);

      for (int k = 0; k < neq; ++k)
      {
#if (__ValueSize == 1)
                                    u_out[ neq*(i    ) + k ] = my_u[__getIndex(k)];
#else
                                    u_out[ neq*(i+  0) + k ] = my_u[__getIndex(k)].s0;
         if ((i+  1) < numProblems) u_out[ neq*(i+  1) + k ] = my_u[__getIndex(k)].s1;
#if (__ValueSize  > 2)
         if ((i+  2) < numProblems) u_out[ neq*(i+  2) + k ] = my_u[__getIndex(k)].s2;
         if ((i+  3) < numProblems) u_out[ neq*(i+  3) + k ] = my_u[__getIndex(k)].s3;
#if (__ValueSize  > 4)
         if ((i+  4) < numProblems) u_out[ neq*(i+  4) + k ] = my_u[__getIndex(k)].s4;
         if ((i+  5) < numProblems) u_out[ neq*(i+  5) + k ] = my_u[__getIndex(k)].s5;
         if ((i+  6) < numProblems) u_out[ neq*(i+  6) + k ] = my_u[__getIndex(k)].s6;
         if ((i+  7) < numProblems) u_out[ neq*(i+  7) + k ] = my_u[__getIndex(k)].s7;
#if (__ValueSize  > 8)
         if ((i+  8) < numProblems) u_out[ neq*(i+  8) + k ] = my_u[__getIndex(k)].s8;
         if ((i+  9) < numProblems) u_out[ neq*(i+  9) + k ] = my_u[__getIndex(k)].s9;
         if ((i+ 10) < numProblems) u_out[ neq*(i+ 10) + k ] = my_u[__getIndex(k)].sA;
         if ((i+ 11) < numProblems) u_out[ neq*(i+ 11) + k ] = my_u[__getIndex(k)].sB;
         if ((i+ 12) < numProblems) u_out[ neq*(i+ 12) + k ] = my_u[__getIndex(k)].sC;
         if ((i+ 13) < numProblems) u_out[ neq*(i+ 13) + k ] = my_u[__getIndex(k)].sD;
         if ((i+ 14) < numProblems) u_out[ neq*(i+ 14) + k ] = my_u[__getIndex(k)].sE;
         if ((i+ 15) < numProblems) u_out[ neq*(i+ 15) + k ] = my_u[__getIndex(k)].sF;
#endif
#endif
#endif
#endif
      }

      // Each has the same value ...
      for (int ii = 0; ii < nelems; ++ii)
         rk_counters[ (i+ii) ].niters = my_counters.niters;

      {
#if (__ValueSize == 1)
                                    rk_counters[ (i    ) ].nsteps = my_counters.nsteps;
#else
                                    rk_counters[ (i+  0) ].nsteps = my_counters.nsteps.s0;
         if ((i+  1) < numProblems) rk_counters[ (i+  1) ].nsteps = my_counters.nsteps.s1;
#if (__ValueSize  > 2)
         if ((i+  2) < numProblems) rk_counters[ (i+  2) ].nsteps = my_counters.nsteps.s2;
         if ((i+  3) < numProblems) rk_counters[ (i+  3) ].nsteps = my_counters.nsteps.s3;
#if (__ValueSize  > 4)
         if ((i+  4) < numProblems) rk_counters[ (i+  4) ].nsteps = my_counters.nsteps.s4;
         if ((i+  5) < numProblems) rk_counters[ (i+  5) ].nsteps = my_counters.nsteps.s5;
         if ((i+  6) < numProblems) rk_counters[ (i+  6) ].nsteps = my_counters.nsteps.s6;
         if ((i+  7) < numProblems) rk_counters[ (i+  7) ].nsteps = my_counters.nsteps.s7;
#if (__ValueSize  > 8)
         if ((i+  8) < numProblems) rk_counters[ (i+  8) ].nsteps = my_counters.nsteps.s8;
         if ((i+  9) < numProblems) rk_counters[ (i+  9) ].nsteps = my_counters.nsteps.s9;
         if ((i+ 10) < numProblems) rk_counters[ (i+ 10) ].nsteps = my_counters.nsteps.sA;
         if ((i+ 11) < numProblems) rk_counters[ (i+ 11) ].nsteps = my_counters.nsteps.sB;
         if ((i+ 12) < numProblems) rk_counters[ (i+ 12) ].nsteps = my_counters.nsteps.sC;
         if ((i+ 13) < numProblems) rk_counters[ (i+ 13) ].nsteps = my_counters.nsteps.sD;
         if ((i+ 14) < numProblems) rk_counters[ (i+ 14) ].nsteps = my_counters.nsteps.sE;
         if ((i+ 15) < numProblems) rk_counters[ (i+ 15) ].nsteps = my_counters.nsteps.sF;
#endif
#endif
#endif
#endif
      }
   }
}
#endif

//#if 0
#ifndef __EnableQueue
#warning 'Skipping rk_driver_vec_queue kernel'
#else

void __kernel
__attribute__((reqd_work_group_size(__blockSize, 1, 1)))
rk_driver_vec_queue (const double p,
               __global const double *u_in,
               __global       double *u_out,
               __ckdata_attr const ckdata_t *ck,
               __global      const rk_t *rk,
               __global       __ValueType *rwk,
               __global rk_counters_t *rk_counters,
               const int numProblems,
               __global int *problemCounter)
{
   const int tid = get_global_id(0);

   const int kk = __ckobj__.n_species;
   const int neq = kk+1;
   const int lenrwk_ck = ck_lenrwk(ck);
   const int lenrwk_rk = rk_lenrwk(rk);

   // Thread-local pointers ...
   __global __ValueType *my_rwk = rwk +
                ((tid / __arrayStride) * (__getIndex(lenrwk_ck + lenrwk_rk + neq)) + (tid % __arrayStride));
   __global __ValueType *my_rwk_rk = my_rwk + (__getIndex(lenrwk_ck));
   __global __ValueType *my_u = my_rwk_rk + (__getIndex(lenrwk_rk));

   const int nelems = vec_step(__ValueType);

   //__private FUNC_SIZE(cklib_callback_t) my_callback = { .p = p, .rwk = my_rwk, .ck = ck };
   __private FUNC_SIZE(cklib_callback_t) my_callback; {my_callback.p = p; my_callback.rwk = my_rwk; my_callback.ck = ck;}
   __private FUNC_SIZE(rk_counters_t) my_counters;

   __private int problem_idx;

   while ((problem_idx = atomic_add(problemCounter, __ValueSize)) < numProblems)
   {
      //for (int k = 0; k < neq; ++k)
      //   my_u[__getIndex(k)] = u_in[i*neq+ k ];
      for (int k = 0; k < neq; ++k)
      {
         for (int lane = 0; lane < nelems; ++lane)
         {
            const int i = min(problem_idx + lane, numProblems-1);
            //if (problem_id < numProblems)
               __write_to(u_in[i * neq + k], lane, my_u[__getIndex(k)]);
         }
      }

      __ValueType t = 0, h = 0;//1e-6;

      FUNC_TYPE(rk_solve) (rk, &t, &h, &my_counters, my_u, my_rwk_rk, &my_callback);

      for (int lane = 0; lane < nelems; ++lane)
      {
         const int i = problem_idx + lane;
         if (i < numProblems)
         {
            for (int k = 0; k < neq; ++k)
               __read_from(my_u[__getIndex(k)], lane, u_out[neq * i + k]);

            __read_from(my_counters.nsteps, lane, rk_counters[i].nsteps);
         }

         // Each lane has the same value ...
         rk_counters[i].niters = my_counters.niters;
      }
   }
}
#endif

//#if 1
#ifdef __EnableQueue
#warning 'Skipping ROS ros_driver kernel'
#else
void __kernel
//__attribute__((vec_type_hint(double)))
__attribute__((reqd_work_group_size(__blockSize, 1, 1)))
ros_driver (const double p,
            __global const double *u_in,
            __global       double *u_out,
            __ckdata_attr const ckdata_t *ck,
            __global      const ros_t *ros,
            __global       int *iwk,
            __global       double *rwk,
            __global ros_counters_t *ros_counters,
            const int numProblems)
{
   const int tid = get_global_id(0);

   const int kk = __ckobj__.n_species;
   const int neq = kk+1;
   const int lenrwk_ck = ck_lenrwk(ck);
   const int lenrwk_ros = ros_lenrwk(ros);
   const int leniwk_ros = ros_leniwk(ros);

   // Thread-local pointers ...
   __global double *my_rwk = rwk +
                ((tid / __arrayStride) * (__getIndex(lenrwk_ck + lenrwk_ros + neq)) + (tid % __arrayStride));
   __global double *my_rwk_ros = my_rwk + (__getIndex(lenrwk_ck));
   __global double *my_u = my_rwk_ros + (__getIndex(lenrwk_ros));
   __global int *my_iwk_ros = iwk +
                ((tid / __arrayStride) * (__getIndex(leniwk_ros)) + (tid % __arrayStride));

   __private cklib_callback_t my_callback = { .p = p, .rwk = my_rwk, .ck = ck };
   __private ros_counters_t my_counters;

   for (int i = get_global_id(0); i < numProblems; i += get_global_size(0))
   {
      for (int k = 0; k < neq; ++k)
         my_u[__getIndex(k)] = u_in[i*neq+ k ];

      //for (int k = 0; k < neq; ++k)
      //   printf("u[%d] = %e\n", k, my_u[__getIndex(k)]);

      double t = 0, h = 0;//1e-6;

      ros_solve (ros, &t, &h, &my_counters, my_u, my_iwk_ros, my_rwk_ros, (void *)0, (void *)0, &my_callback);

      for (int k = 0; k < neq; ++k)
      {
         u_out[ neq*(i    ) + k ] = my_u[__getIndex(k)];
      }

      ros_counters[i].niters = my_counters.niters;
      ros_counters[i].nst = my_counters.nst;
   }
}
#endif

//#if 0
#ifndef __EnableQueue
#warning 'Skipping ROS ros_driver_queue kernel'
#else
void __kernel
__attribute__((reqd_work_group_size(__blockSize, 1, 1)))
ros_driver_queue (const double p,
            __global const double *u_in,
            __global       double *u_out,
            __ckdata_attr const ckdata_t *ck,
            __global      const ros_t *ros,
            __global       int *iwk,
            __global       double *rwk,
            __global ros_counters_t *ros_counters,
            const int numProblems,
            __global int *problemCounter)
{
   const int tid = get_global_id(0);

   const int kk = __ckobj__.n_species;
   const int neq = kk+1;
   const int lenrwk_ck = ck_lenrwk(ck);
   const int lenrwk_ros = ros_lenrwk(ros);
   const int leniwk_ros = ros_leniwk(ros);

   // Thread-local pointers ...
   __global double *my_rwk = rwk +
                ((tid / __arrayStride) * (__getIndex(lenrwk_ck + lenrwk_ros + neq)) + (tid % __arrayStride));
   __global double *my_rwk_ros = my_rwk + (__getIndex(lenrwk_ck));
   __global double *my_u = my_rwk_ros + (__getIndex(lenrwk_ros));
   __global int *my_iwk_ros = iwk +
                ((tid / __arrayStride) * (__getIndex(leniwk_ros)) + (tid % __arrayStride));

   __private cklib_callback_t my_callback = { .p = p, .rwk = my_rwk, .ck = ck };
   __private ros_counters_t my_counters;

   //__local int group_index;
   __private int local_index;

   //if (get_local_id(0) == 0)
   //   group_index = atomic_add( problemCounter, get_local_size(0));

   //barrier(CLK_LOCAL_MEM_FENCE);

   //while (group_index < numProblems)
   while ( (local_index = atomic_inc(problemCounter)) < numProblems)
   {
      //if (get_local_id(0) == 0)
      //   printf("group_index, group_id=%d\n", group_index, get_group_id(0));

      const int i = local_index;
      //int i = group_index + get_local_id(0);
      //if (i >= numProblems)
      //   i = numProblems-1;

      for (int k = 0; k < neq; ++k)
         my_u[__getIndex(k)] = u_in[i*neq+ k ];

      //for (int k = 0; k < neq; ++k)
      //   printf("u[%d] = %e\n", k, my_u[__getIndex(k)]);

      double t = 0, h = 0;//1e-6;

      ros_solve (ros, &t, &h, &my_counters, my_u, my_iwk_ros, my_rwk_ros, (void *)0, (void *)0, &my_callback);

      for (int k = 0; k < neq; ++k)
      {
         u_out[ neq*(i    ) + k ] = my_u[__getIndex(k)];
      }

      ros_counters[i].niters = my_counters.niters;
      ros_counters[i].nst = my_counters.nst;

      // Update the global problem counter queue.
      //barrier(CLK_LOCAL_MEM_FENCE);

      //if (get_local_id(0) == 0)
      //   group_index = atomic_add( problemCounter, get_local_size(0));

      //barrier(CLK_LOCAL_MEM_FENCE);
   }
}
#endif

// ROS internal routines ...
inline __ValueType FUNC_TYPE(ros_getewt) (__global const ros_t *ros, const int k, __global const __ValueType *y)
{
   const __ValueType ewtk = (ros->s_rtol * fabs(y[__getIndex(k)])) + ros->s_atol;
   return (1.0/ewtk);
}
inline __ValueType FUNC_TYPE(ros_wnorm) (__global const ros_t *ros, __global const __ValueType *x, __global const __ValueType *y)
{
   const int neq = ros->neq;
   __ValueType sum = 0;
   for (int k = 0; k < neq; k++)
   {
      //const __ValueType ewtk = FUNC_TYPE(ros_getewt)(ros, k, y);
      //__ValueType prod = x[__getIndex(k)] * ewtk;
      const __ValueType ewt = (ros->s_rtol * fabs(y[__getIndex(k)])) + ros->s_atol;
      __ValueType prod = x[__getIndex(k)] / ewt;
      sum += (prod*prod);
   }

   return sqrt(sum / (double)neq);
}
inline void FUNC_TYPE(ros_dzero) (const int len, __global __ValueType x[])
{
   for (int k = 0; k < len; ++k)
      x[__getIndex(k)] = 0.0;
}
inline void FUNC_TYPE(ros_dcopy) (const int len, const __global __ValueType src[], __global __ValueType dst[])
{
   for (int k = 0; k < len; ++k)
      dst[__getIndex(k)] = src[__getIndex(k)];
}
/*inline void dcopy_if (const int len, const MaskType &mask, const __global __ValueType src[], __global __ValueType dst[])
{
   for (int k = 0; k < len; ++k)
      dst[k] = if_then_else (mask, src[k], dst[k]);
}*/

//inline void FUNC_TYPE(ros_daxpy) (const int len, const double alpha, const __global __ValueType x[], __global __ValueType y[])
inline void FUNC_TYPE(ros_daxpy1) (const int len, const double alpha, const __global __ValueType x[], __global __ValueType y[])
{
   // Alpha is scalar type ... and can be easily checked.
   if (alpha == 1.0)
   {
      for (int k = 0; k < len; ++k)
         y[__getIndex(k)] += x[__getIndex(k)];
   }
   else if (alpha == -1.0)
   {
      for (int k = 0; k < len; ++k)
         y[__getIndex(k)] -= x[__getIndex(k)];
   }
   else if (alpha != 0.0)
   {
      for (int k = 0; k < len; ++k)
         y[__getIndex(k)] += alpha * x[__getIndex(k)];
   }
}
inline void FUNC_TYPE(ros_daxpy) (const int len, const __ValueType alpha, const __global __ValueType x[], __global __ValueType y[])
{
   // Alpha is vector type ... tedious to switch.
   for (int k = 0; k < len; ++k)
      y[__getIndex(k)] += alpha * x[__getIndex(k)];
}

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

int FUNC_TYPE(ros_ludec) (const int n, __global __ValueType *A, __global __IntType *ipiv)
{
   int ierr = ROS_SUCCESS;

   const int nelems = vec_step(__ValueType);

   int all_pivk[__ValueSize];

   /* k-th elimination step number */
   for (int k = 0; k < n; ++k)
   {
     __global __ValueType *A_k = A + __getIndex(k*n); // pointer to the column

     /* find pivot row number */
     for (int el = 0; el < nelems; ++el)
     {
        int pivk = k;
        double Akp;
        __read_from( A_k[__getIndex(pivk)], el, Akp);
        for (int i = k+1; i < n; ++i)
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
           ierr = (k+1);
           //printf("Singular value %d %d\n", k, el);
           break;
        }

        /* swap a(k,1:N) and a(piv,1:N) if necessary */
        if (pivk != k)
        {
           __global __ValueType *A_i = A; // pointer to the first column
           for (int i = 0; i < n; ++i, A_i += __getIndex(n))
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
     for (int i = k+1; i < n; ++i)
       A_k[__getIndex(i)] *= mult;

     /* row_i = row_i - [a(i,k)/a(k,k)] row_k, i=k+1, ..., m-1 */
     /* row k is the pivot row after swapping with row l.      */
     /* The computation is done one column at a time,          */
     /* column j=k+1, ..., n-1.                                */

     for (int j = k+1; j < n; ++j)
     {
       __global __ValueType *A_j = A + __getIndex(j*n);
       const __ValueType a_kj = A_j[__getIndex(k)];

       /* a(i,j) = a(i,j) - [a(i,k)/a(k,k)]*a(k,j)  */
       /* a_kj = a(k,j), col_k[i] = - a(i,k)/a(k,k) */
       //if (any(a_kj != 0.0)) {
         for (int i = k+1; i < n; ++i) {
           A_j[__getIndex(i)] -= a_kj * A_k[__getIndex(i)];
         }
       //}
     }
   }

   return ierr;
   //if (ierr)
   //{
   //  fprintf(stderr,"Singular pivot j=%d\n", ierr-1);
   //  exit(-1);
   //}
}
void FUNC_TYPE(ros_lusol) (const int n, __global __ValueType *A, __global __IntType *ipiv, __global __ValueType *b)
{
   /* Permute b, based on pivot information in p */
   for (int k = 0; k < n; ++k)
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
   for (int k = 0; k < n-1; ++k)
   {
      __global __ValueType *A_k = A + __getIndex(k*n);
      const __ValueType bk = b[__getIndex(k)];
      for (int i = k+1; i < n; ++i)
         b[__getIndex(i)] -= A_k[__getIndex(i)] * bk;
   }
   /* Solve Ux = y, store solution x in b */
   for (int k = n-1; k > 0; --k)
   {
      __global __ValueType *A_k = A + __getIndex(k*n);
      b[__getIndex(k)] /= A_k[__getIndex(k)];
      const __ValueType bk = b[__getIndex(k)];
      for (int i = 0; i < k; ++i)
         b[__getIndex(i)] -= A_k[__getIndex(i)] * bk;
   }
   b[__getIndex(0)] /= A[__getIndex(0)];
}
void FUNC_TYPE(ros_fdjac) (__global const ros_t *ros, const __ValueType tcur, const __ValueType hcur, __global __ValueType *y, __global __ValueType *fy, __global __ValueType *Jy, __private void *user_data)
{
   const int neq = ros->neq;

   // Norm of fy(t) ...
   __ValueType fnorm = FUNC_TYPE(ros_wnorm)( ros, fy, y );

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
      const __ValueType ewtj = FUNC_TYPE(ros_getewt)(ros, j, y);
      const __ValueType dely = fmax( sround * fabs(ysav), r0 / ewtj );
      y[__getIndex(j)] += dely;

      __global __ValueType *jcol = &Jy[__getIndex(j*neq)];

      //func (neq, tcur, y, jcol, user_data);
      FUNC_TYPE(cklib_callback) (neq, tcur, y, jcol, user_data);

      const __ValueType delyi = 1. / dely;
      for (int i = 0; i < neq; ++i)
         jcol[__getIndex(i)] = (jcol[__getIndex(i)] - fy[__getIndex(i)]) * delyi;

      y[__getIndex(j)] = ysav;
   }
}
int FUNC_TYPE(ros_hin) (__global const ros_t *ros, const __ValueType t, __ValueType *h0, __global __ValueType* y, __global __ValueType *rwk, __private void *user_data)
{
   const int neq = ros->neq;

   __global __ValueType *ydot  = rwk;
   __global __ValueType *y1    = ydot + neq;
   __global __ValueType *ydot1 = y1 + neq;

   double hlb = ros->h_min;
   double hub = ros->h_max;
   //double hlb = h_min;
   //double hub = h_max;

   // Alread done ...
   __MaskType done = isgreaterequal(*h0, ros->h_min);
   //__MaskType done = isgreaterequal(*h0, h_min);

   __ValueType hg = sqrt(hlb*hub);

   if (hub < hlb)
   {
      *h0 = __select(hg, *h0, done);

      return RK_SUCCESS;
   }

   //if (hub < hlb)
   //{
   //   *h0 = hg;
   //   return RK_SUCCESS;
   //}

   // Start iteration to find solution to ... {WRMS norm of (h0^2 y'' / 2)} = 1

   __MaskType hnew_is_ok = 0;
   __ValueType hnew = hg;
   const int miters = 10;
   int iter = 0;
   int ierr = RK_SUCCESS;

   // compute ydot at t=t0
   FUNC_TYPE(cklib_callback)(neq, 0.0, y, ydot, user_data);

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
      FUNC_TYPE(cklib_callback) (neq, 0.0, y1, ydot1, user_data);

      // Compute WRMS norm of y''
      #ifdef __INTEL_COMPILER
      #pragma ivdep
      #endif
      for (int k = 0; k < neq; k++)
         y1[__getIndex(k)] = (ydot1[__getIndex(k)] - ydot[__getIndex(k)]) / hg;

      __ValueType yddnrm = FUNC_TYPE(ros_wnorm) (ros, y1, y);

      //std::cout << "iter " << iter << " hg " << hg << " y'' " << yddnrm << std::endl;
      //std::cout << "ydot " << ydot[neq-1] << std::endl;

      // should we accept this?
      //if (hnew_is_ok || iter == miters)
      //{
      //   hnew = hg;
      //   //if (iter == miters) fprintf(stderr, "ERROR_HIN_MAX_ITERS\n");
      //   ierr = (hnew_is_ok) ? RK_SUCCESS : RK_HIN_MAX_ITERS;
      //   break;
      //}
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
      //if ( (hrat > 0.5) && (hrat < 2.0) )
      //   hnew_is_ok = 1;
      hnew_is_ok = isgreater(hrat, 0.5) & isless(hrat, 2.0);

      // If y'' is still bad after a few iterations, just accept h and give up.
      //if ( (iter > 1) && (hrat > 2.0) ) {
      //   hnew = hg;
      //   hnew_is_ok = 1;
      //}
      if (iter > 1)
      {
         hnew_is_ok = isgreater(hrat, 2.0);
         hnew = __select (hnew, hg, hnew_is_ok);
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

int FUNC_TYPE(ros_solve) (__global const ros_t *ros, __ValueType *tcur, __ValueType *hcur, __private FUNC_SIZE(rk_counters_t) *counters, __global __ValueType y[], __global __IntType iwk[], __global __ValueType rwk[], __private void *user_data)
{
   int ierr = ROS_SUCCESS;

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
         ierr = FUNC_TYPE(ros_hin) (ros, t, &(h), y, rwk, user_data);
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
   __global __ValueType *ynew = fy + __getIndex(neq);
   __global __ValueType *Jy   = ynew + __getIndex(neq);
   __global __ValueType *ktmp = Jy + __getIndex(neq*neq);
   __global __ValueType *yerr = ynew;
   //__global double *ewt  = &Jy[neq*neq];

   __MaskType done = isless( fabs(t - ros->t_stop), ros->t_round); 
   //while (fabs(t - ros->t_stop) > ros->t_round)
   while (__any(__not(done)))
   {
      // Set the error weight array.
      //ros_setewt (ros, y, ewt);

      // Compute the RHS and Jacobian matrix.
      //func (neq, t, y, fy, user_data);
      FUNC_TYPE(cklib_callback) (neq, t, y, fy, user_data);
      //nfe++;

      //if (jac == NULL)
      {
         FUNC_TYPE(ros_fdjac) (ros, t, h, y, fy, Jy, user_data);
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
#if (__ValueSize == 1)
      ros_ludec(neq, Jy, iwk); // scalar variant
#else
      FUNC_TYPE(ros_ludec)(neq, Jy, iwk); // simd variant
#endif
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
            FUNC_TYPE(ros_dcopy) (neq, y, ynew);

            for (int j = 0; j < s; ++j)
            {
               const double Asj = A(s,j);
               //if (Asj != 0.0)
               {
                  //printf("Asj = %f %d %d\n", Asj, s, j);
                  __global __ValueType *k_j = &ktmp[__getIndex(j*neq)];

                  FUNC_TYPE(ros_daxpy1) (neq, Asj, k_j, ynew);
               }
            }

            //func (neq, t, ynew, fy, user_data);
            FUNC_TYPE(cklib_callback) (neq, t, ynew, fy, user_data);
            //nfe++;

            //printf("newF=%d\n", s);
            //for (int k = 0; k < neq; ++k)
            //   printf("ynew[%d] = %e %e\n", k, ynew[k], fy[k]);
         }

         //printf("stage=%d\n", s);
         //for (int k = 0; k < neq; ++k)
         //   printf("fy[%d] = %e\n", k, fy[k]);

         // Build the sub-space vector K
         __global __ValueType *k_s = &ktmp[__getIndex(s*neq)];
         FUNC_TYPE(ros_dcopy) (neq, fy, k_s);

         for (int j = 0; j < s; j++)
         {
            //if (C(s,j) != 0.0)
            {
               const __ValueType hCsj = C(s,j) / h;
               //printf("C/h = %f %d %d\n", hCsj, s, j);

               __global __ValueType *k_j = &ktmp[__getIndex(j*neq)];
               FUNC_TYPE(ros_daxpy) (neq, hCsj, k_j, k_s);
            }
         }

         //printf("k before=%d\n", s);
         //for (int k = 0; k < neq; ++k)
         //   printf("k[%d] = %e\n", k, ks[k]);

         // Solve the current stage ..
#if (__ValueSize == 1)
         ros_lusol (neq, Jy, iwk, k_s); // scalar version
#else
         FUNC_TYPE(ros_lusol) (neq, Jy, iwk, k_s);
#endif

         //printf("k after=%d\n", s);
         //for (int k = 0; k < neq; ++k)
         //   printf("k[%d] = %e\n", k, ks[k]);
      }

      // Compute the error estimation of the trial solution
      FUNC_TYPE(ros_dzero) (neq, yerr);

      for (int j = 0; j < ros->numStages; ++j)
      {
         //if (ros->E[j] != 0.0)
         {
            __global __ValueType *k_j = &ktmp[__getIndex(j*neq)];
            FUNC_TYPE(ros_daxpy1) (neq, ros->E[j], k_j, yerr);
         }
      }

      __ValueType herr = fmax(1.0e-20, FUNC_TYPE(ros_wnorm) (ros, yerr, y));

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
         FUNC_TYPE(ros_dcopy) (neq, y, ynew);
         for (int j = 0; j < ros->numStages; ++j)
         {
            //if (ros->M[j] != 0.0)
            {
               __global __ValueType *k_j = &ktmp[__getIndex(j*neq)];
               FUNC_TYPE(ros_daxpy1) (neq, ros->M[j], k_j, ynew);
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

//#if 0
#ifdef __EnableQueue
#warning 'Skipping ros_driver_vec CL build'
#else

void __kernel
//__attribute__((vec_type_hint(__ValueType)))
__attribute__((reqd_work_group_size(__blockSize, 1, 1)))
ros_driver_vec (const double p,
                __global const double *u_in,
                __global       double *u_out,
                __ckdata_attr const ckdata_t *ck,
                __global      const ros_t *ros,
                __global      __IntType *iwk,
                __global      __ValueType *rwk,
                __global ros_counters_t *ros_counters,
                const int numProblems)
{
   const int tid = get_global_id(0);

   const int kk = __ckobj__.n_species;
   const int neq = kk+1;
   const int lenrwk_ck = ck_lenrwk(ck);
   const int lenrwk_ros = ros_lenrwk(ros);
   const int leniwk_ros = ros_leniwk(ros);

   // Thread-local pointers ...
   __global __ValueType *my_rwk = rwk +
                ((tid / __arrayStride) * (__getIndex(lenrwk_ck + lenrwk_ros + neq)) + (tid % __arrayStride));
   __global __ValueType *my_rwk_ros = my_rwk + (__getIndex(lenrwk_ck));
   __global __ValueType *my_u = my_rwk_ros + (__getIndex(lenrwk_ros));
   __global __IntType *my_iwk_ros = iwk +
                ((tid / __arrayStride) * (__getIndex(leniwk_ros)) + (tid % __arrayStride));

   const int nelems = vec_step(__ValueType);

   //__private FUNC_SIZE(cklib_callback_t) my_callback = { .p = p, .rwk = my_rwk, .ck = ck };
   __private FUNC_SIZE(cklib_callback_t) my_callback; {my_callback.p = p; my_callback.rwk = my_rwk; my_callback.ck = ck;}
   //__private FUNC_SIZE(ros_counters_t) my_counters;
   __private FUNC_SIZE(rk_counters_t) my_counters;

   for (int i = nelems * get_global_id(0); i < numProblems; i += nelems * get_global_size(0))
   {
      //for (int k = 0; k < neq; ++k)
      //   my_u[__getIndex(k)] = u_in[i*neq+ k ];
      for (int k = 0; k < neq; ++k)
      {
         for (int lane = 0; lane < __ValueSize; ++lane)
         {
            const int problem_id = min(i + lane, numProblems-1);
            //if (problem_id < numProblems)
               __write_to(u_in[problem_id * neq + k], lane, my_u[__getIndex(k)]);
         }
      }

      //printf("T[%d] = %f\n", i, my_u[__getIndex(kk)]);

      __ValueType t = 0, h = 0;//1e-6;

      FUNC_TYPE(ros_solve) (ros, &t, &h, &my_counters, my_u, my_iwk_ros, my_rwk_ros, &my_callback);

      //printf("T[%d] = %f %d %d\n", i, my_u[__getIndex(kk)], my_counters.nsteps, my_counters.niters);

      for (int lane = 0; lane < __ValueSize; ++lane)
      {
         const int problem_id = i + lane;
         if (problem_id < numProblems)
         {
            for (int k = 0; k < neq; ++k)
               __read_from(my_u[__getIndex(k)], lane, u_out[neq * problem_id + k]);

            __read_from(my_counters.nsteps, lane, ros_counters[problem_id].nst);
         }
      }

#if 0
      for (int k = 0; k < neq; ++k)
      {
#if (__ValueSize == 1)
                                    u_out[ neq*(i    ) + k ] = my_u[__getIndex(k)];
#else
                                    u_out[ neq*(i+  0) + k ] = my_u[__getIndex(k)].s0;
         if ((i+  1) < numProblems) u_out[ neq*(i+  1) + k ] = my_u[__getIndex(k)].s1;
#if (__ValueSize  > 2)
         if ((i+  2) < numProblems) u_out[ neq*(i+  2) + k ] = my_u[__getIndex(k)].s2;
         if ((i+  3) < numProblems) u_out[ neq*(i+  3) + k ] = my_u[__getIndex(k)].s3;
#if (__ValueSize  > 4)
         if ((i+  4) < numProblems) u_out[ neq*(i+  4) + k ] = my_u[__getIndex(k)].s4;
         if ((i+  5) < numProblems) u_out[ neq*(i+  5) + k ] = my_u[__getIndex(k)].s5;
         if ((i+  6) < numProblems) u_out[ neq*(i+  6) + k ] = my_u[__getIndex(k)].s6;
         if ((i+  7) < numProblems) u_out[ neq*(i+  7) + k ] = my_u[__getIndex(k)].s7;
#if (__ValueSize  > 8)
         if ((i+  8) < numProblems) u_out[ neq*(i+  8) + k ] = my_u[__getIndex(k)].s8;
         if ((i+  9) < numProblems) u_out[ neq*(i+  9) + k ] = my_u[__getIndex(k)].s9;
         if ((i+ 10) < numProblems) u_out[ neq*(i+ 10) + k ] = my_u[__getIndex(k)].sA;
         if ((i+ 11) < numProblems) u_out[ neq*(i+ 11) + k ] = my_u[__getIndex(k)].sB;
         if ((i+ 12) < numProblems) u_out[ neq*(i+ 12) + k ] = my_u[__getIndex(k)].sC;
         if ((i+ 13) < numProblems) u_out[ neq*(i+ 13) + k ] = my_u[__getIndex(k)].sD;
         if ((i+ 14) < numProblems) u_out[ neq*(i+ 14) + k ] = my_u[__getIndex(k)].sE;
         if ((i+ 15) < numProblems) u_out[ neq*(i+ 15) + k ] = my_u[__getIndex(k)].sF;
#endif
#endif
#endif
#endif
      }
#endif

      // Each has the same value ...
      for (int ii = 0; ii < nelems; ++ii)
         if ((i+ii) < numProblems)
            ros_counters[ (i+ii) ].niters = my_counters.niters;

#if 0
      {
#if (__ValueSize == 1)
                                    ros_counters[ (i    ) ].nst = my_counters.nsteps;
#else
                                    ros_counters[ (i+  0) ].nst = my_counters.nsteps.s0;
         if ((i+  1) < numProblems) ros_counters[ (i+  1) ].nst = my_counters.nsteps.s1;
#if (__ValueSize  > 2)
         if ((i+  2) < numProblems) ros_counters[ (i+  2) ].nst = my_counters.nsteps.s2;
         if ((i+  3) < numProblems) ros_counters[ (i+  3) ].nst = my_counters.nsteps.s3;
#if (__ValueSize  > 4)
         if ((i+  4) < numProblems) ros_counters[ (i+  4) ].nst = my_counters.nsteps.s4;
         if ((i+  5) < numProblems) ros_counters[ (i+  5) ].nst = my_counters.nsteps.s5;
         if ((i+  6) < numProblems) ros_counters[ (i+  6) ].nst = my_counters.nsteps.s6;
         if ((i+  7) < numProblems) ros_counters[ (i+  7) ].nst = my_counters.nsteps.s7;
#if (__ValueSize  > 8)
         if ((i+  8) < numProblems) ros_counters[ (i+  8) ].nst = my_counters.nsteps.s8;
         if ((i+  9) < numProblems) ros_counters[ (i+  9) ].nst = my_counters.nsteps.s9;
         if ((i+ 10) < numProblems) ros_counters[ (i+ 10) ].nst = my_counters.nsteps.sA;
         if ((i+ 11) < numProblems) ros_counters[ (i+ 11) ].nst = my_counters.nsteps.sB;
         if ((i+ 12) < numProblems) ros_counters[ (i+ 12) ].nst = my_counters.nsteps.sC;
         if ((i+ 13) < numProblems) ros_counters[ (i+ 13) ].nst = my_counters.nsteps.sD;
         if ((i+ 14) < numProblems) ros_counters[ (i+ 14) ].nst = my_counters.nsteps.sE;
         if ((i+ 15) < numProblems) ros_counters[ (i+ 15) ].nst = my_counters.nsteps.sF;
#endif
#endif
#endif
#endif
      }
#endif
   }
}
#endif

//#if 0
#ifndef __EnableQueue
#warning 'Skipping ROS ros_driver_vec_queue kernel'
#else
void __kernel
__attribute__((vec_type_hint(__ValueType)))
__attribute__((reqd_work_group_size(__blockSize, 1, 1)))
ros_driver_vec_queue (const double p,
                __global const double *u_in,
                __global       double *u_out,
                __ckdata_attr const ckdata_t *ck,
                __global      const ros_t *ros,
                __global      __IntType *iwk,
                __global      __ValueType *rwk,
                __global ros_counters_t *ros_counters,
                const int numProblems,
                __global int *problemCounter)
{
   const int tid = get_global_id(0);

   const int kk = __ckobj__.n_species;
   const int neq = kk+1;
   const int lenrwk_ck = ck_lenrwk(ck);
   const int lenrwk_ros = ros_lenrwk(ros);
   const int leniwk_ros = ros_leniwk(ros);

   // Thread-local pointers ...
   __global __ValueType *my_rwk = rwk +
                ((tid / __arrayStride) * (__getIndex(lenrwk_ck + lenrwk_ros + neq)) + (tid % __arrayStride));
   __global __ValueType *my_rwk_ros = my_rwk + (__getIndex(lenrwk_ck));
   __global __ValueType *my_u = my_rwk_ros + (__getIndex(lenrwk_ros));
   __global __IntType *my_iwk_ros = iwk +
                ((tid / __arrayStride) * (__getIndex(leniwk_ros)) + (tid % __arrayStride));

   __private FUNC_SIZE(cklib_callback_t) my_callback; {my_callback.p = p; my_callback.rwk = my_rwk; my_callback.ck = ck;}
   //__private FUNC_SIZE(ros_counters_t) my_counters;
   __private FUNC_SIZE(rk_counters_t) my_counters;

   __private int problem_idx;

   while ((problem_idx = atomic_add(problemCounter, __ValueSize)) < numProblems)
   {
      for (int k = 0; k < neq; ++k)
      {
         for (int lane = 0; lane < __ValueSize; ++lane)
         {
            const int i = min(problem_idx + lane, numProblems-1);
            __write_to(u_in[i * neq + k], lane, my_u[__getIndex(k)]);
         }
      }

      __ValueType t = 0, h = 0;

      FUNC_TYPE(ros_solve) (ros, &t, &h, &my_counters, my_u, my_iwk_ros, my_rwk_ros, &my_callback);

      for (int lane = 0; lane < __ValueSize; ++lane)
      {
         const int i = problem_idx + lane;
         if (i < numProblems)
         {
            for (int k = 0; k < neq; ++k)
               __read_from(my_u[__getIndex(k)], lane, u_out[neq * i + k]);

            __read_from(my_counters.nsteps, lane, ros_counters[i].nst);
         }
      }

      // Each has the same value ...
      for (int i = 0; i < __ValueSize; ++i)
         if ((i+problem_idx) < numProblems)
            ros_counters[ (i+problem_idx) ].niters = my_counters.niters;
   }
}

#endif
