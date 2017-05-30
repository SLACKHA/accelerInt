#ifndef __ck_hpp
#define __ck_hpp

#include <stdio.h>
#include <stdlib.h>

#include <string.h>
//#include <math.h>
#include <cmath>

#include <sstream>
#include <string>
#include <vector>

//#define NDEBUG
#include <assert.h>

// Disable printf format %d warnings
#ifdef __INTEL_COMPILER
#pragma warning(disable:181)
#endif

#include <Vector.h>

#include <tr1/type_traits>
#include <limits>

namespace CK
{

//const double RU = 8.31447215e7;	// erg / (mol * K)
const double RU  = 8.314e7;	// erg / (mol * K)
const double RUC = 1.987;	// cal / (mol * K)
const double PA  = 1.013250e6;	// dynes / cm^2

//const double small = 1.0e-110;
//const double big   = 1.0e+110;
const double small = exp10(-300);
const double big   = exp10( 300);
const double exparg= log(big);

enum { sp_strlen = 16 };
enum { th_max_terms = 7 };
enum { rx_max_order = 3 }; // 3 reactants/products
enum { rx_max_falloff = 8 };

// ... binary switches
enum { rx_flag_nil		= 0,		// Arrehius and equilibrium ... normal
       rx_flag_irrev		= (1 << 1),	// Irreversible rxn
       rx_flag_rparams		= (1 << 2),	// Reversible with explicit reverse rate params
       rx_flag_thdbdy		= (1 << 3),	// 3-body efficiencies
       rx_flag_falloff		= (1 << 4),	// Pressure dependencies ... and types
       rx_flag_falloff_sri	= (1 << 5),
       rx_flag_falloff_troe	= (1 << 6),
       rx_flag_falloff_sri5	= (1 << 7),
       rx_flag_falloff_troe4	= (1 << 7) };

struct CKData
{
   // Species info

   int n_species;

   //VectorType<std::string> sp_name;
   std::vector<std::string> sp_name;

   VectorType<double>	sp_mwt;

   // Thermo info
   VectorType<double>	th_tmid;
   VectorType<double>	th_alo;
   VectorType<double>	th_ahi;

   // Reaction info

   int n_reactions;

   VectorType<double>	rx_A;
   VectorType<double>	rx_b;
   VectorType<double>	rx_E; // normalized by R already ...

   VectorType<int>	rx_nu;
   VectorType<int>	rx_nuk;
   VectorType<int>	rx_sumnu;

   // Reversible reactions with explicit reversible parameters
   int n_reversible_reactions;

   VectorType<int>	rx_rev_idx;
   VectorType<double>	rx_rev_A;
   VectorType<double>	rx_rev_b;
   VectorType<double>	rx_rev_E;

   // Irreversible reactions ...
   int n_irreversible_reactions;

   VectorType<int>	rx_irrev_idx;

   // 3rd-body efficiencies for pressure dependent reactions ...
   int n_thdbdy;

   VectorType<int>	rx_thdbdy_idx;
   VectorType<int>	rx_thdbdy_offset;
   VectorType<int>	rx_thdbdy_spidx;
   VectorType<double>	rx_thdbdy_alpha;

   // Fall-off reactions ...
   int n_falloff;

   VectorType<int>	rx_falloff_idx;
   VectorType<int>	rx_falloff_spidx;
   VectorType<double>	rx_falloff_params;

   VectorType<int>	rx_info;

   int lenrwk, leniwk;
   VectorType<int>	iwk;
   VectorType<double>	rwk;

   int initialized;

   // Constructor ...
   CKData(void) : initialized(0),
                  n_species(0),
                  n_reactions(0), 
                  n_reversible_reactions(0),
                  n_irreversible_reactions(0),
                  n_thdbdy(0),
                  n_falloff(0),
                  lenrwk(0), leniwk(0)
   {}
};

enum { do_profile = 0 };

static double ckwyp_time   = 0;
static double ckratt_time  = 0;
static double ckratc_time  = 0;
static double thdbdy_time  = 0;
static double falloff_time = 0;
static int num_temp_solves = 0;
static int num_temp_iters  = 0;

static struct _profiles
{
   ~_profiles()
   {
      if (do_profile)
      {
      printf("RU    = %e\n", RU);
      printf("RUC   = %e\n", RUC);
      printf("PA    = %e\n", PA);
      printf("small = %e\n", small);
      printf("big   = %e\n", big);
      printf("num_temp_solves=%d, num_temp_iters=%d\n", num_temp_solves, num_temp_iters);
      printf("ckwyp_time = %f %f\n", 1000*(ckwyp_time), 1000*(ckwyp_time - ckratt_time - ckratc_time));
      printf("ckratt_time = %f\n", 1000*(ckratt_time));
      printf("ckratc_time = %f, %f\n", 1000*(ckratc_time), 1000*(ckratc_time-thdbdy_time-falloff_time));
      printf("thdbdy_time = %f\n", 1000*(thdbdy_time));
      printf("falloff_time = %f\n", 1000*(falloff_time));
      }
   }
} profiles;

// Internal utility functions ...

// Remove leading/trailing white space from string
std::string trim (char *str)
{
   // Trim leading non-letters
   //while (!isalnum(*str)) str++;
   while (isspace(*str)) str++;

   // trim trailing non-letters
   char *end = str + strlen(str) - 1;
   //while (end > str && !isalnum(*end)) end--;
   while (end > str && isspace(*end)) end--;

   return std::string(str, end+1);
}
// Square an argument ...
template <typename T>
inline T sqr (const T& a) { return (a*a); }

namespace details
{

#ifdef FAST_MATH
#warning 'enabled FAST MATH pow() functions'
inline bool is_odd (unsigned q) { return bool(q % 2); }
// p^q where q is a positive integral
inline double __powu (double p, unsigned q)
{
   if      (q == 0) return 1.0;
   else if (q == 1) return p;
   else if (q == 2) return p*p;
   else if (q == 3) return p*p*p;
   else if (q == 4) return p*p*p*p;
   else
   {
      // q^p -> (q^(p/2))^2 ... recursively takes log(q) ops
      double r(1);
      while (q)
      {
         //if (q % 2) // odd power ...
         if (is_odd(q)) // odd power ...
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
inline double __powi (double p, int q)
{
   if (p == 0.0)
   {
      if (not(q))
         return 1.0;
      else if (q < 0)
         return std::numeric_limits<double>::infinity();
      else
         return 0.0;
   }

   if      (q > 0) return __powu(p,q);
   else if (q < 0) return __powu(1.0/p,unsigned(-q));
   else            return 1.0;
}
#else
//inline double __powu (double p, unsigned q) { return pow(p,q); }
//inline double __powi (double p, int q) { return pow(p,q); }
inline double __powu (double p, unsigned q) { return __builtin_powi(p,q); }
inline double __powi (double p, int q) { return __builtin_powi(p,q); }
#endif

} // end namespace details

template <typename T> inline double pow(const double &a, const T &b) { return std::pow(a,b); }
template <> inline double pow<int>(const double &a, const int &b) { return details::__powi(a,b); }
template <> inline double pow<unsigned int>(const double &a, const unsigned int &b) { return details::__powu(a,b); }

int ckinit (CKData &ck, FILE *ckfile)
{
   const int verbose = 0;

   // Species info ...
   int &kk = ck.n_species;

   fread(&kk, sizeof(int), 1, ckfile);
   printf("n_species = %d\n", kk);

   ck.sp_name.resize(kk);
   ck.sp_mwt.resize(kk);

   ck.th_tmid.resize(kk);
   ck.th_alo.resize(th_max_terms * kk);
   ck.th_ahi.resize(th_max_terms * kk);

   for (int k = 0; k < kk; ++k)
   {
      std::string name;
      name.resize(sp_strlen);
      fread(const_cast<char*>(name.c_str()), sizeof(char), sp_strlen, ckfile);
      ck.sp_name[k] = trim(const_cast<char*>(name.c_str()));

      double mwt;
      fread(&mwt, sizeof(double), 1, ckfile);
      ck.sp_mwt[k] = mwt;

      double tmid;
      fread(&tmid, sizeof(double), 1, ckfile);
      ck.th_tmid[k] = tmid;

      double alo[th_max_terms];
      fread(alo, sizeof(double), th_max_terms, ckfile);
      std::copy(alo, alo + th_max_terms, &ck.th_alo[k*th_max_terms]);

      double ahi[th_max_terms];
      fread(ahi, sizeof(double), th_max_terms, ckfile);
      std::copy(ahi, ahi + th_max_terms, &ck.th_ahi[k*th_max_terms]);

      printf("%3d: name = %s, mwt = %f, tmid = %f\n", k, ck.sp_name[k].c_str(), ck.sp_mwt[k], ck.th_tmid[k]);
   }

   // Reaction info ...
   int &ii = ck.n_reactions;
   fread(&ii, sizeof(int), 1, ckfile);
   printf("n_reactions = %d\n", ii);

   ck.rx_A.resize(ii);
   ck.rx_b.resize(ii);
   ck.rx_E.resize(ii);

   ck.rx_nu.resize(ii*rx_max_order*2);
   ck.rx_nuk.resize(ii*rx_max_order*2);
   ck.rx_sumnu.resize(ii);

   ck.rx_info.resize(ii);

   for (int i = 0; i < ii; ++i)
   {
      fread(&ck.rx_A[i], sizeof(double), 1, ckfile);
      fread(&ck.rx_b[i], sizeof(double), 1, ckfile);
      fread(&ck.rx_E[i], sizeof(double), 1, ckfile);
      if (verbose) printf("%3d: %e %e %e\n", i, ck.rx_A[i],ck.rx_b[i],ck.rx_E[i]);

      int nu[rx_max_order*2], nuk[rx_max_order*2];
      fread(nu, sizeof(int), rx_max_order*2, ckfile);
      fread(nuk, sizeof(int), rx_max_order*2, ckfile);

      int sumnu = 0;
      for (int n = 0; n < rx_max_order*2; ++n)
      {
         ck.rx_nu[i*rx_max_order*2 +n] = nu[n];
         ck.rx_nuk[i*rx_max_order*2 +n] = nuk[n];
         sumnu += nu[n];
         //printf("%3d: nu, nuk=%d %d\n", i, nu[n], nuk[n]);
      }
      ck.rx_sumnu[i] = sumnu;

      // Initialize the rxn info flag
      ck.rx_info[i] = rx_flag_nil;
   }

   // ... Reversible reaction with explicit parameters ...
   int &n_rev = ck.n_reversible_reactions;
   fread(&n_rev, sizeof(int), 1, ckfile);
   printf("n_reversible_reactions = %d\n", n_rev);

   if (n_rev > 0)
   {
      ck.rx_rev_idx.resize(n_rev);
      ck.rx_rev_A.resize(n_rev);
      ck.rx_rev_b.resize(n_rev);
      ck.rx_rev_E.resize(n_rev);

      for (int n = 0; n < n_rev; ++n)
      {
         fread(&ck.rx_rev_idx[n], sizeof(int), 1, ckfile);
         fread(&ck.rx_rev_A[n], sizeof(double), 1, ckfile);
         fread(&ck.rx_rev_b[n], sizeof(double), 1, ckfile);
         fread(&ck.rx_rev_E[n], sizeof(double), 1, ckfile);

         //printf("%3d: [%3d], rev_A = %e, rev_b = %f, rev_E = %e; \n", n, ck.rx_rev_idx[n], ck.rx_rev_A[n], ck.rx_rev_b[n], ck.rx_rev_E[n]);

         int k = ck.rx_rev_idx[n];
         ck.rx_info[k] |= rx_flag_rparams;
      }
   }

   // ... Irreversible reactions ...
   int &n_irrev = ck.n_irreversible_reactions;
   fread(&n_irrev, sizeof(int), 1, ckfile);
   printf("n_irreversible_reactions = %d\n", n_irrev);

   if (n_irrev > 0)
   {
      ck.rx_irrev_idx.resize(n_irrev);

      fread(&ck.rx_irrev_idx[0], sizeof(int), n_irrev, ckfile);

      for (int n = 0; n < n_irrev; ++n)
      {
         int k = ck.rx_irrev_idx[n];
         ck.rx_info[k] |= rx_flag_irrev;
         if (verbose) printf("%3d: is irreversible\n", k);
      }
   }

   // ... 3rd-body efficiencies for pressure dependence ...
   int &n_thdbdy = ck.n_thdbdy;
   fread(&n_thdbdy, sizeof(int), 1, ckfile);
   printf("n_thdbdy = %d\n", n_thdbdy);

   if (n_thdbdy > 0)
   {
      ck.rx_thdbdy_idx.resize(n_thdbdy);
      ck.rx_thdbdy_offset.resize(n_thdbdy+1);

      std::vector<int> _spidx;
      std::vector<double> _alpha;
      //ck.rx_thdbdy_spidx.clear();
      //ck.rx_thdbdy_alpha.clear();

      int n_thdbdy_coefs = 0;
      for (int n = 0; n < n_thdbdy; ++n)
      {
         fread(&ck.rx_thdbdy_idx[n], sizeof(int), 1, ckfile);

         int n_sp;
         fread(&n_sp, sizeof(int), 1, ckfile);

         ck.rx_thdbdy_offset[n] = n_thdbdy_coefs;
         n_thdbdy_coefs += n_sp;

         for (int i = 0; i < n_sp; ++i)
         {
            int spidx;
            double alpha;

            fread(&spidx, sizeof(int), 1, ckfile);
            fread(&alpha, sizeof(double), 1, ckfile);

            //ck.rx_thdbdy_spidx.push_back(spidx);
            //ck.rx_thdbdy_alpha.push_back(alpha);
            _spidx.push_back(spidx);
            _alpha.push_back(alpha);
         }

         int k = ck.rx_thdbdy_idx[n];
         ck.rx_info[k] |= rx_flag_thdbdy;
         if (verbose) printf("%d, rxn %d is third-body\n", n, k);

         //printf("%d, ck.rx_thdbdy_idx=%d, ck.rx_thdbdy_offset=%d, nsp=%d\n", n, ck.rx_thdbdy_idx[n], ck.rx_thdbdy_offset[n], n_sp);
      }
      ck.rx_thdbdy_offset[n_thdbdy] = n_thdbdy_coefs;
      //printf("n_coefs=%d\n", n_thdbdy_coefs);

      ck.rx_thdbdy_spidx.resize(n_thdbdy_coefs);
      ck.rx_thdbdy_alpha.resize(n_thdbdy_coefs);
      for (int i = 0; i < n_thdbdy_coefs; ++i)
      {
         ck.rx_thdbdy_spidx[i] = _spidx[i];
         ck.rx_thdbdy_alpha[i] = _alpha[i];
      }
   }

   // ... Fall-off pressure dependencies ...
   int &n_falloff = ck.n_falloff;
   fread(&n_falloff, sizeof(int), 1, ckfile);
   printf("n_falloff = %d\n", n_falloff);

   if (n_falloff > 0)
   {
      ck.rx_falloff_idx.resize(n_falloff);
      ck.rx_falloff_spidx.resize(n_falloff);
      ck.rx_falloff_params.resize(n_falloff*rx_max_falloff);

      for (int n = 0; n < n_falloff; ++n)
      {
         fread(&ck.rx_falloff_idx[n], sizeof(int), 1, ckfile);
         fread(&ck.rx_falloff_spidx[n], sizeof(int), 1, ckfile);
         fread(&ck.rx_falloff_params[n*rx_max_falloff], sizeof(double), rx_max_falloff, ckfile);

         int type = 0;
         fread(&type, sizeof(int), 1, ckfile);

         int k = ck.rx_falloff_idx[n];
         ck.rx_info[k] |= rx_flag_falloff;
         if (type == 1 or type == 2) {
            printf("SRI fall-off rxn not ready\n");
            exit(-1);
            ck.rx_info[k] |= rx_flag_falloff_sri;
            if (type == 2)
               ck.rx_info[k] |= rx_flag_falloff_sri5;
         }
         else if (type == 3 or type == 4) {
            ck.rx_info[k] |= rx_flag_falloff_troe;
            if (type == 4)
               ck.rx_info[k] |= rx_flag_falloff_troe4;
         }

         if (verbose) printf("falloff: %3d, %3d, %3d, %3d, %d\n", n, ck.rx_falloff_idx[n], ck.rx_falloff_spidx[n], type, ck.rx_info[k]);
      }
   }

   if (verbose)
      for (int i = 0; i < ii; ++i)
      {
         int *nu = &ck.rx_nu[i*rx_max_order*2];
         int *nuk = &ck.rx_nuk[i*rx_max_order*2];

         std::ostringstream rxn;

         for (int n = 0; n < 3; ++n)
            if (nu[n])
            {
               int k = nuk[n];
               if (abs(nu[n]) > 1)
                  rxn << abs(nu[n]) << " ";
               rxn << ck.sp_name[k];
               if (n != 2 && nu[n+1])
                  rxn << " + ";
            }

         if (ck.rx_info[i] & rx_flag_irrev)
            rxn << "  =  ";
         else
            rxn << " <=> ";

         for (int n = 3; n < 6; ++n)
            if (nu[n])
            {
               int k = nuk[n];
               if (abs(nu[n]) > 1)
                  rxn << abs(nu[n]) << " ";
               rxn << ck.sp_name[k];
               if (n != 5 && nu[n+1])
                  rxn << " + ";
            }

         if (ck.rx_info[i] & rx_flag_thdbdy)
            if (ck.rx_info[i] & rx_flag_falloff)
               rxn << " (+M)";
            else
               rxn << " + M";

/*         std::string type;
         if (ck.rx_info[i] & rx_flag_rparams) type += std::string(":Rev");

         if (ck.rx_info[i] & rx_flag_thdbdy)
            if (ck.rx_info[i] & rx_flag_falloff)
                type += std::string(":Falloff");
            else
               type += std::string(":3rd");*/

         printf("%3d: A = %9.3e, b = %6.3f, Ea/R = %12.4f; %s\n", i, ck.rx_A[i], ck.rx_b[i], ck.rx_E[i], rxn.str().c_str());
      }

#ifndef _OPENMP
   // Work space ...
   ck.leniwk = 0;
   ck.lenrwk = kk + 4*ii;
   ck.rwk.resize(ck.lenrwk);
#endif

   ck.initialized = 1;

   return 0;
}

#if 0
//inline double compute_H_RT (const int k, const double& restrict T, const CKData& restrict ck)
inline double compute_H_RT (const int k, const double& T, const CKData& ck)
{
   // H / RT = Sum_(i=1)^(5){ a_i / i * T^(i-1) } + a_6 / T

   const int offset = k * th_max_terms;

   const double *a = (T > ck.th_tmid[k]) ? &ck.th_ahi[offset] : &ck.th_alo[offset];
#if 1
   return a[0] + a[5] / T + T * (a[1] / 2.0 + T * (a[2] / 3.0 + T * (a[3] / 4.0 + T * a[4] / 5.0)));
#else
   const double T1 = T / 2;
   const double T2 = T*T / 3;
   const double T3 = T*T*T / 4;
   const double T4 = T*T*T*T / 5;
   return a[0] + T1 * a[1] + T2 * a[2] + T3 * a[3] + T4 * a[4] + a[5] / T;
#endif
}
#else

namespace details
{

   using std::tr1::false_type;
   using std::tr1::true_type;
   using std::tr1::is_scalar;

   // Vector format ...
   template <typename VectorType>
   inline VectorType compute_H_RT (const int k, const VectorType& T, const CKData& ck, false_type)
   {
      const int offset = k * CK::th_max_terms;

      const double *ahi = &ck.th_ahi[offset];
      const double *alo = &ck.th_alo[offset];

      VectorType H_RT_lo;
      H_RT_lo  = T *            (alo[4]/5.0);
      H_RT_lo  = T * (H_RT_lo + (alo[3]/4.0));
      H_RT_lo  = T * (H_RT_lo + (alo[2]/3.0));
      H_RT_lo  = T * (H_RT_lo + (alo[1]/2.0));
      H_RT_lo += (alo[5] / T);
      H_RT_lo += alo[0];

      VectorType H_RT_hi;
      H_RT_hi  = T *            (ahi[4]/5.0);
      H_RT_hi  = T * (H_RT_hi + (ahi[3]/4.0));
      H_RT_hi  = T * (H_RT_hi + (ahi[2]/3.0));
      H_RT_hi  = T * (H_RT_hi + (ahi[1]/2.0));
      H_RT_hi += (ahi[5] / T);
      H_RT_hi += ahi[0];

      return if_then_else(T > ck.th_tmid[k], H_RT_hi, H_RT_lo);
   }

   // Scalar format ...
   template <typename ValueType>
   inline ValueType compute_H_RT (const int k, const ValueType& T, const CKData& ck, true_type)
   {
      const int offset = k * CK::th_max_terms;

      const double *a = (T > ck.th_tmid[k]) ? &ck.th_ahi[offset] : &ck.th_alo[offset];

      //return a[0] + T * (a[1] + T * (a[2] + T * (a[3] + T * a[4])));
      return a[0] + a[5] / T + T * (a[1] / 2.0 + T * (a[2] / 3.0 + T * (a[3] / 4.0 + T * a[4] / 5.0)));
   }

} // namespace details
template <typename ValueType>
inline ValueType compute_H_RT (const int k, const ValueType& T, const CKData& ck)
{
   return details::compute_H_RT (k, T, ck, typename details::is_scalar<ValueType>::type());
}

#endif
#if 0
inline double compute_Cp_R (const int k, const double& T, const CKData& ck)
{
   // Cp / R = Sum_(i=1)^(5){ a_i * T^(i-1) }

   const int offset = k * th_max_terms;

   const double *a = (T > ck.th_tmid[k]) ? &ck.th_ahi[offset] : &ck.th_alo[offset];
#if 1
   return a[0] + T * (a[1] + T * (a[2] + T * (a[3] + T * a[4])));
#else
   const double T2 = T*T;
   const double T3 = T*T*T;
   const double T4 = T*T*T*T;
   return a[0] + T * a[1] + T2 * a[2] + T3 * a[3] + T4 * a[4];
#endif
}
#else
namespace details
{
   // Vector format ...
   template <typename VectorType>
   inline VectorType compute_Cp_R (const int k, const VectorType& T, const CKData& ck, false_type)
   {
      const int offset = k * CK::th_max_terms;

      const double *ahi = &ck.th_ahi[offset];
      const double *alo = &ck.th_alo[offset];

      VectorType Cp_R_lo;
      Cp_R_lo = T *         alo[4];
      Cp_R_lo = T * (Cp_R_lo + alo[3]);
      Cp_R_lo = T * (Cp_R_lo + alo[2]);
      Cp_R_lo = T * (Cp_R_lo + alo[1]);
      Cp_R_lo += alo[0];

      VectorType Cp_R_hi;
      Cp_R_hi = T *         ahi[4];
      Cp_R_hi = T * (Cp_R_hi + ahi[3]);
      Cp_R_hi = T * (Cp_R_hi + ahi[2]);
      Cp_R_hi = T * (Cp_R_hi + ahi[1]);
      Cp_R_hi += ahi[0];

      return if_then_else(T > ck.th_tmid[k], Cp_R_hi, Cp_R_lo);
   }
   // Scalar format ...
   template <typename ValueType>
   inline ValueType compute_Cp_R (const int k, const ValueType& T, const CKData& ck, true_type)
   {
      // Cp / R = Sum_(i=1)^(5){ a_i * T^(i-1) }

      const int offset = k * CK::th_max_terms;

      const double *a = (T > ck.th_tmid[k]) ? &ck.th_ahi[offset] : &ck.th_alo[offset];

      return a[0] + T * (a[1] + T * (a[2] + T * (a[3] + T * a[4])));
   }

} // namespace details
template <typename ValueType>
inline ValueType compute_Cp_R (const int k, const ValueType& T, const CKData& ck)
{
   return details::compute_Cp_R (k, T, ck, typename details::is_scalar<ValueType>::type());
}
#endif

// Mean molecular weight given mole fractions ... g / mol
double ckmmwx (double x[], const CKData &ck)
{
   // <W> = Sum_k { x_k * w_k }
   double mean_mwt(0);
   for (int k = 0; k < ck.n_species; ++k)
      mean_mwt += x[k] * ck.sp_mwt[k];

   return mean_mwt;
}
// Mean molecular weight given mass fractions ... g / mol
template <typename ValueType>
ValueType ckmmwy (ValueType y[], const CKData &ck)
{  
   // <W> = 1 / Sum_k { y_k / w_k }
   ValueType sumyow(0.0);
   for (int k = 0; k < ck.n_species; ++k)
      sumyow += (y[k] / ck.sp_mwt[k]);

   return (ValueType(1) / sumyow);
}
// Return pointer to molecular weights ... g / mol
double* ckwt (const CKData &ck)
{
   return (double*)(&ck.sp_mwt[0]);
}

// Species enthalpies in mass units given temperature ... erg / g
//void ckhms (const double &T, double h[], const CKData &ck)
template <typename ValueType>
void ckhms (const ValueType &T, ValueType h[], const CKData &ck)
{
   //const double RUT = RU * T;

   #pragma ivdep
   for (int k = 0; k < ck.n_species; ++k)
   {
      h[k] = compute_H_RT(k, T, ck);

      //h[k] *= (RUT / ck.sp_mwt[k]);

      h[k] *= (RU / ck.sp_mwt[k]);
      h[k] *= T;
   }
}
// Species internal energy in mass units given temperature ... erg / g
//void ckums (const double &T, double u[], const CKData &ck)
template <typename ValueType>
void ckums (const ValueType &T, ValueType u[], const CKData &ck)
{
   //const double RUT = RU * T;

   #pragma ivdep
   for (int k = 0; k < ck.n_species; ++k)
   {
      // U = H - RT
      u[k] = compute_H_RT(k, T, ck) - 1.0;

      //u[k] *= (RUT / ck.sp_mwt[k]);

      u[k] *= (RU / ck.sp_mwt[k]);
      u[k] *= T;
   }
}
// Species Cp in mass units given temperature ... erg / (g * k)
//void ckcpms (const double &T, double cp[], const CKData &ck)
template <typename ValueType>
void ckcpms (const ValueType &T, ValueType cp[], const CKData &ck)
{
   #pragma ivdep
   for (int k = 0; k < ck.n_species; ++k)
   {
      cp[k] = compute_Cp_R(k, T, ck);

      cp[k] *= (RU / ck.sp_mwt[k]);
   }
}
// Species Cv in mass units given temperature ... erg / (g * k)
//void ckcvms (const double &T, double cv[], const CKData &ck)
template <typename ValueType>
void ckcvms (const ValueType &T, ValueType cv[], const CKData &ck)
{
   #pragma ivdep
   for (int k = 0; k < ck.n_species; ++k)
   {
      //cv[k] = compute_Cp_R(k, T, ck) - 1.0;
      //cv[k] *= (RU / ck.sp_mwt[k]);

      cv[k]  = compute_Cp_R(k, T, ck);
      cv[k] -= 1.0;
      cv[k] *= (RU / ck.sp_mwt[k]);
   }
}
// Mixture enthalpy in mass units given mass fractions and temperature ... erg / g
template <typename ValueType>
ValueType ckhbms (const ValueType &T, ValueType y[], const CKData &ck)
{
   //const ValueType RUT = RU * T;
   ValueType h_mix(0);

   #pragma ivdep
   for (int k = 0; k < ck.n_species; ++k)
   {
      ValueType h_k = compute_H_RT(k, T, ck);
      //h_k *= (RUT / ck.sp_mwt[k]);
      h_k /= ck.sp_mwt[k];
      //h_mix += (y[k] * h_k);
      h_k *= y[k];
      h_mix += h_k;
   }

   //return h_mix;
   //return h_mix * RUT;
   h_mix *= (T * RU);
   return h_mix;
}
// Mixture internal energy in mass units given mass fractions and temperature ... erg / g
double ckubms (const double &T, double y[], const CKData &ck)
{
   double u_mix(0);

   #pragma ivdep
   for (int k = 0; k < ck.n_species; ++k)
   {
      const double u_k = compute_H_RT(k, T, ck) - 1.0;

      u_mix += (u_k * y[k] / ck.sp_mwt[k]);
   }

   return (u_mix * RU * T);
}
// Mixture Cp in mass units given mass fractions and temperature ... erg / (g * k)
template <typename ValueType>
ValueType ckcpbs (const ValueType &T, ValueType y[], const CKData &ck)
{
   ValueType cp_mix(0.);

   #pragma ivdep
   for (int k = 0; k < ck.n_species; ++k)
   {
      ValueType cp_k = compute_Cp_R(k, T, ck);
      cp_k *= (RU / ck.sp_mwt[k]);
      cp_k *= y[k];
      cp_mix += cp_k;
      //cp_mix += (y[k] * cp_k);
   }

   return cp_mix;
}
// Mixture Cv in mass units given mass fractions and temperature ... erg / (g * k)
//double ckcvbs (const double &T, double y[], const CKData &ck)
template <typename ValueType>
ValueType ckcvbs (const ValueType &T, ValueType y[], const CKData &ck)
{
   ValueType cv_mix(0.);

   #pragma ivdep
   for (int k = 0; k < ck.n_species; ++k)
   {
      //const double cv_k = compute_Cp_R(k, T, ck) - 1.0;
      //cv_mix += (cv_k * y[k] / ck.sp_mwt[k]);

      ValueType cv_k = compute_Cp_R(k, T, ck);
      cv_k -= 1.0;
      cv_k *= (RU / ck.sp_mwt[k]);
      cv_k *= y[k];

      cv_mix += cv_k;
   }

   //return (cv_mix * RU);
   return cv_mix;
}
// Mixture Cp in molar units given mole fractions and temperature ... erg / (g * k)
double ckcpbl (const double &T, double x[], const CKData &ck)
{ 
   double cp_mix(0);

   #pragma ivdep
   for (int k = 0; k < ck.n_species; ++k)
   {
      const double cp_k = compute_Cp_R(k, T, ck); 

      cp_mix += (cp_k * x[k]);
   }

   return (cp_mix * RU);
}
namespace details
{
   template <typename ValueType>
   inline void cksmh (const ValueType& T, ValueType smh[], const CKData& ck, true_type)
   {
      const ValueType logTm1 = log(T) - ValueType(1);
      const ValueType invT   = ValueType(1) / T;
      const ValueType T1     = T / ValueType(2);
      const ValueType T2     = T*T / ValueType(6);
      const ValueType T3     = T*T*T / ValueType(12);
      const ValueType T4     = T*T*T*T / ValueType(20);

      #pragma ivdep
      for (int k = 0; k < ck.n_species; ++k)
      {
         const int offset = k * th_max_terms;

         const ValueType *a = (T > ck.th_tmid[k]) ? &ck.th_ahi[offset] : &ck.th_alo[offset];

         //smh[k] = a[0] * logTm1 + T * (a[1]/2.0 + T * (a[2]/6.0 + T * (a[3]/12.0 + T * a[4]/20.0))) - a[5] * invT + a[6];
         smh[k] = a[0] * logTm1 + T1 * a[1] + T2 * a[2] + T3 * a[3] + T4 * a[4] - a[5] * invT + a[6];
/*         double sum(0);
         sum += logTm1 * a[0];
         sum += T1 * a[1];
         sum += T2 * a[2];
         sum += T3 * a[3];
         sum += T4 * a[4];
         smh[k] = sum + a[6] - a[5] / T;*/
      }
   }

   template <typename VectorType>
   inline void cksmh (const VectorType& T, VectorType smh[], const CKData& ck, false_type)
   {
      typedef typename VectorType::value_type ValueType;

      const VectorType logTm1 = log(T) - ValueType(1);
      const VectorType invT   = 1.0 / T;
//      const ValueType T1     = T / ValueType(2);
//      const ValueType T2     = T*T / ValueType(6);
//      const ValueType T3     = T*T*T / ValueType(12);
//      const ValueType T4     = T*T*T*T / ValueType(20);

      #pragma ivdep
      for (int k = 0; k < ck.n_species; ++k)
      {
         const int offset = k * th_max_terms;

         const double *ahi = &ck.th_ahi[offset];
         const double *alo = &ck.th_alo[offset];

         VectorType smh_hi;
         smh_hi  = T *           (ahi[4] / 20.0);
         smh_hi  = T * (smh_hi + (ahi[3] / 12.0));
         smh_hi  = T * (smh_hi + (ahi[2] /  6.0));
         smh_hi  = T * (smh_hi + (ahi[1] /  2.0));
         smh_hi += (logTm1 * ahi[0]);
         smh_hi -= (invT * ahi[5]);
         smh_hi += ahi[6];

         VectorType smh_lo;
         smh_lo  = T *           (alo[4] / 20.0);
         smh_lo  = T * (smh_lo + (alo[3] / 12.0));
         smh_lo  = T * (smh_lo + (alo[2] /  6.0));
         smh_lo  = T * (smh_lo + (alo[1] /  2.0));
         smh_lo += (logTm1 * alo[0]);
         smh_lo -= (invT * alo[5]);
         smh_lo += alo[6];

         smh[k] = if_then_else(T > ck.th_tmid[k], smh_hi, smh_lo);
      }
   }
}
// Species S/R - H/RT ... special function.
template <typename ValueType>
inline void cksmh (const ValueType& T, ValueType smh[], const CKData& ck)
{
   details::cksmh (T, smh, ck, typename details::is_scalar<ValueType>::type());
}
// Mixture density given pressure, temperature and mass fractions ... g / cm^3
//double ckrhoy (const double &p, const double &T, double y[], const CKData &ck)
template <typename ValueType>
ValueType ckrhoy (const ValueType &p, const ValueType &T, ValueType y[], const CKData &ck)
{
   //const double mean_mwt = ckmmwy(y, ck);
   //const ValueType mean_mwt = ckmmwy(y, ck);
   ValueType mean_mwt = ckmmwy(y, ck);

   // rho = p / (<R> * T) = p / (RU / <W> * T)

   //return p / (T * RU / mean_mwt);
   mean_mwt *= p;
   mean_mwt /= T;
   mean_mwt /= RU;
   return mean_mwt;
}
// Mixture pressure given mixture density, temperature and mass fractions ... dyne / cm^2
double ckpy (const double &rho, const double &T, double y[], const CKData &ck)
{
   const double mean_mwt = ckmmwy(y, ck);

   // p = rho * (RU / <W>) * T

   //return rho * T * RU / mean_mwt;
   return (rho * T / mean_mwt) * RU;
}
// Convert mole fraction to mass fractions
void ckxty (double x[], double y[], const CKData &ck)
{
   // y_k = x_k * w_k / <W>
   const double mean_mwt_inv = 1. / ckmmwx(x,ck);

   #pragma ivdep
   for (int k = 0; k < ck.n_species; ++k)
      y[k] = x[k] * (ck.sp_mwt[k] * mean_mwt_inv);
}
// Convert mass fraction to molar concentration given p and T
//void ckytcp (const double& p, const double &T, double y[], double c[], const CKData &ck)
//void ckytcp (const double& p, const double &T, double* restrict y, double* restrict c, const CKData &ck)
template <typename ValueType>
void ckytcp (const ValueType& p, const ValueType &T, ValueType y[], ValueType c[], const CKData &ck)
{
#if 0
   double sumyow(0);
   for (int k = 0; k < ck.n_species; ++k)
      sumyow += y[k] / ck.sp_mwt[k];

   sumyow = sumyow * T * RU;

   for (int k = 0; k < ck.n_species; ++k)
      c[k] = p * y[k] / (sumyow * ck.sp_mwt[k]);
#else
   // [c]_k = rho * y_k / w_k
   //const double rho = ckrhoy (p, T, y, ck);
   const ValueType rho = ckrhoy (p, T, y, ck);

   #pragma ivdep
   for (int k = 0; k < ck.n_species; ++k)
   {
      //c[k] = rho * y[k] / ck.sp_mwt[k];
      c[k] = rho * y[k];
      c[k] /= ck.sp_mwt[k];
   }
#endif
}

namespace details
{

// Compute molar forward/reverse reaction rates given temperature and molar concentrations
// ... utility function ...
void ckratt (const double &T, double smh[], double eqk[], double rkf[], double rkr[], const CKData &ck, true_type)
{
   const int kk = ck.n_species;
   const int ii = ck.n_reactions;

   const double logT = log(T);
   const double invT = 1.0 / T;
   const double pfac = PA / (RU * T); // (dyne / cm^2) / (erg / mol / K) / (K)

   // I. Temperature-dependent rates ...

   // S/R - H/RT ... only needed for equilibrium.
   cksmh (T, smh, ck);

   assert (rx_max_order == 3);

   #pragma ivdep
   for (int i = 0; i < ii; ++i)
   {
      // Basic Arrhenius rates: A * exp( logT * b - E_R / T)
      rkf[i] = ck.rx_A[i] * exp(ck.rx_b[i] * logT - ck.rx_E[i] * invT);
      //rkf[i] = ck.rx_A[i] * exp(ck.rx_b[i] * logT - ck.rx_E[i] / T);
   }

   assert( ck.n_reversible_reactions == 0 );

/*   // Irreversible reaction ...
   for (int i = 0; i < ck.n_irreversible_reactions; ++i)
   {
      int j = ck.rx_irrev_idx[i];
      rkr[j] = 0.0;
      eqk[j] = 0.0;
   }*/

   #pragma ivdep
   for (int i = 0; i < ii; ++i)
   {
#if 1
      //assert (not(ck.rx_info[i] & rx_flag_rparams)); // reversible parameters aren't supported yet.

      if (ck.rx_info[i] & rx_flag_irrev)
      {
         // Irreversible reaction ...
         rkr[i] = 0.0;
         eqk[i] = 0.0;
      }
      else
#endif
      {
         // Equilibrium calc for reversible rate ...

         // Sum_k { nu_k * (S/R - H/RT)_k }
         const int *nuk = &ck.rx_nuk[i*rx_max_order*2];
         const int *nu  = &ck.rx_nu[i*rx_max_order*2];

         //double sumsmh(0);
         //sumsmh += (nu[0] * smh[nuk[0]]);
         double sumsmh = (nu[0] * smh[nuk[0]]);
         if (nuk[1] not_eq -1) sumsmh += (nu[1] * smh[nuk[1]]);
         if (nuk[2] not_eq -1) sumsmh += (nu[2] * smh[nuk[2]]);
         sumsmh += (nu[3] * smh[nuk[3]]);
         if (nuk[4] not_eq -1) sumsmh += (nu[4] * smh[nuk[4]]);
         if (nuk[5] not_eq -1) sumsmh += (nu[5] * smh[nuk[5]]);
         /*if (nuk[0] not_eq -1 and nu[0] not_eq 0) sumsmh += (nu[0] * smh[nuk[0]]);
         if (nuk[1] not_eq -1 and nu[1] not_eq 0) sumsmh += (nu[1] * smh[nuk[1]]);
         if (nuk[2] not_eq -1 and nu[2] not_eq 0) sumsmh += (nu[2] * smh[nuk[2]]);
         if (nuk[3] not_eq -1 and nu[3] not_eq 0) sumsmh += (nu[3] * smh[nuk[3]]);
         if (nuk[4] not_eq -1 and nu[4] not_eq 0) sumsmh += (nu[4] * smh[nuk[4]]);
         if (nuk[5] not_eq -1 and nu[5] not_eq 0) sumsmh += (nu[5] * smh[nuk[5]]);*/
         /*for (int j = 0; j < rx_max_order*2; ++j)
         {
            const int k = ck.rx_nuk[i*rx_max_order*2+j];
            const int nu = ck.rx_nu[i*rx_max_order*2+j];
            if (k not_eq -1 and nu not_eq 0)
               sumsmh += (nu * smh[k]);
         }*/

         eqk[i] = exp(fmin(sumsmh, exparg));

         if (ck.rx_sumnu[i] not_eq 0)
            eqk[i] *= pow(pfac,ck.rx_sumnu[i]);
            //eqk[i] *= __powi(pfac,ck.rx_sumnu[i]);

         if (not(ck.rx_info[i] & rx_flag_irrev))
            rkr[i] = rkf[i] / fmax(eqk[i],small);
      }

      //printf("%3d: %e, %e, %f\n", i, rkf[i], rkr[i], log(eqk[i]));
   }

}
template <typename ValueType>
void ckratt (const ValueType &T, ValueType smh[], ValueType eqk[], ValueType rkf[], ValueType rkr[], const CKData &ck, false_type)
{
   const int kk = ck.n_species;
   const int ii = ck.n_reactions;

   const ValueType logT = log(T);
   const ValueType invT = 1.0 / T;
   const ValueType pfac = (PA / RU) / T; // (dyne / cm^2) / (erg / mol / K) / (K)

   // I. Temperature-dependent rates ...

   // S/R - H/RT ... only needed for equilibrium.
   cksmh (T, smh, ck);

   assert (rx_max_order == 3);

   #pragma ivdep
   for (int i = 0; i < ii; ++i)
   {
      // Basic Arrhenius rates: A * exp( logT * b - E_R / T)
      rkf[i] = ck.rx_A[i] * exp(logT * ck.rx_b[i] - invT * ck.rx_E[i]);
   }

   assert( ck.n_reversible_reactions == 0 );

/*   // Irreversible reaction ...
   for (int i = 0; i < ck.n_irreversible_reactions; ++i)
   {
      int j = ck.rx_irrev_idx[i];
      rkr[j] = 0.0;
      eqk[j] = 0.0;
   }*/

   #pragma ivdep
   for (int i = 0; i < ii; ++i)
   {
#if 1
      if (ck.rx_info[i] & rx_flag_irrev)
      {
         // Irreversible reaction ...
         rkr[i] = 0.0;
         eqk[i] = 0.0;
      }
      else
#endif
      {
         // Equilibrium calc for reversible rate ...

         // Sum_k { nu_k * (S/R - H/RT)_k }
         const int *nuk = &ck.rx_nuk[i*rx_max_order*2];
         const int *nu  = &ck.rx_nu[i*rx_max_order*2];

         ValueType             sumsmh  = (smh[nuk[0]] * (double)nu[0]);
         if (nuk[1] not_eq -1) sumsmh += (smh[nuk[1]] * (double)nu[1]);
         if (nuk[2] not_eq -1) sumsmh += (smh[nuk[2]] * (double)nu[2]);
                               sumsmh += (smh[nuk[3]] * (double)nu[3]);
         if (nuk[4] not_eq -1) sumsmh += (smh[nuk[4]] * (double)nu[4]);
         if (nuk[5] not_eq -1) sumsmh += (smh[nuk[5]] * (double)nu[5]);

         eqk[i] = exp(fmin(sumsmh, exparg));

         if (ck.rx_sumnu[i] not_eq 0)
            eqk[i] *= pow(pfac,ck.rx_sumnu[i]);

         if (not(ck.rx_info[i] & rx_flag_irrev))
            rkr[i] = rkf[i] / fmax(eqk[i],small);
      }
   }
}
void ckratc (const double &T, const double c[], double ctb[], double rkf[], double rkr[], const CKData &ck, true_type)
{
   const int kk = ck.n_species;
   const int ii = ck.n_reactions;

   const double logT = log(T);
// const double invT = 1.0 / T;
// const double pfac = PA / (RU * T); // (dyne / cm^2) / (erg / mol / K) / (K)

   // II. Concentration-dependent rates ...

   //std::fill(ctb, ctb + ii, 1.0);
   for (int i = 0; i < ii; ++i)
      ctb[i] = 1.0;

   // Third-body reactions ...
   if (ck.n_thdbdy > 0)
   {
      double _time0 = WallClock();

      double ctot(0);
      for (int k = 0; k < kk; ++k)
         ctot += c[k];

      //printf("ctot = %e\n", ctot);

      for (int n = 0; n < ck.n_thdbdy; ++n)
      {
         const int rxn_idx = ck.rx_thdbdy_idx[n];

         ctb[rxn_idx] = ctot;

         // Add in the specific efficiencies ...

         for (int m = ck.rx_thdbdy_offset[n]; m < ck.rx_thdbdy_offset[n+1]; ++m)
         {
            const int k = ck.rx_thdbdy_spidx[m];
            ctb[rxn_idx] += (ck.rx_thdbdy_alpha[m] - 1.0) * c[k];
         }
         //printf("%3d, %3d, %e\n", n, rxn_idx, ctb[rxn_idx]);
      }

      if (do_profile) thdbdy_time += (WallClock() - _time0);
   }

   // Fall-off pressure dependencies ...
   if (ck.n_falloff > 0)
   {
      double _time0 = WallClock();

      #pragma ivdep
      for (int n = 0; n < ck.n_falloff; ++n)
      {
         const int rxn_idx = ck.rx_falloff_idx[n];

         // Concentration of the third-body ... could be a specific species, too.
         double cthb;
         if (ck.rx_falloff_spidx[n] != -1)
         {
            cthb = ctb[rxn_idx];
            ctb[rxn_idx] = 1.0;
         }
         else
            cthb = c[ ck.rx_falloff_spidx[n] ];

         const double *fpar = &ck.rx_falloff_params[n*rx_max_falloff];

         // Low-pressure limit rate ...
         double rklow = fpar[0] * exp(fpar[1] * logT - fpar[2] / T);

         // Reduced pressure ...
         double pr    = rklow * cthb / rkf[rxn_idx];

         // Correction ... k_infty (pr / (1+pr)) * F()
         double p_cor;

         // Different F()'s ...
         //if (ck.rx_info[rxn_idx] & rx_flag_falloff_sri)
         //{
         //   printf("SRI fall-off rxn not ready\n");
         //   exit(-1);
         //}
         //else if (ck.rx_info[rxn_idx] & rx_flag_falloff_troe)
         if (ck.rx_info[rxn_idx] & rx_flag_falloff_troe)
         {
            // 3-parameter Troe form ...
            double Fcent = (1.0 - fpar[3]) * exp(-T / fpar[4]) + fpar[3] * exp(-T / fpar[5]);

            // Additional 4th (T**) parameter ...
            if (ck.rx_info[rxn_idx] & rx_flag_falloff_troe4)
               Fcent += exp(-fpar[6] / T);
               //Fcent += exp(-fpar[6] * invT);

            double log_Fc = log10( fmax(Fcent,small) );
            double eta    = 0.75 - 1.27 * log_Fc;
            double log_pr = log10( fmax(pr,small) );
            double plus_c = log_pr - (0.4 + 0.67 * log_Fc);
            double log_F  = log_Fc / (1.0 + sqr(plus_c / (eta - 0.14 * plus_c)));
            double Fc     = exp10(log_F);

            p_cor = Fc * (pr / (1.0 + pr));
         }
         else // Lindermann form
         {
            p_cor = pr / (1.0 + pr);
         }

         rkf[rxn_idx] *= p_cor;
         rkr[rxn_idx] *= p_cor;
         //printf("%3d, %3d, %e, %e\n", n, rxn_idx, ck.rx_info[rxn_idx], p_cor, cthb);
      }

      if (do_profile) falloff_time += (WallClock() - _time0);
   } // fall-off's

   // II. Stoichiometry rates ...

   assert (rx_max_order == 3);

#ifdef __MIC__
   #warning 'ivdep for Stoichiometric rates'
   #pragma ivdep
#endif
   for (int i = 0; i < ii; ++i)
   {
      const int *nu  = &ck.rx_nu[i*rx_max_order*2];
      const int *nuk = &ck.rx_nuk[i*rx_max_order*2];

      rkf[i] = rkf[i] * ctb[i] * pow( c[nuk[0]],-nu[0]);
      rkr[i] = rkr[i] * ctb[i] * pow( c[nuk[3]], nu[3]);
      //rkf[i] = rkf[i] * ctb[i] * __powu( c[nuk[0]],-nu[0]);
      //rkr[i] = rkr[i] * ctb[i] * __powu( c[nuk[3]], nu[3]);
      if (nuk[1] not_eq -1)
      {
         //                      rkf[i] = rkf[i] * __powu( c[nuk[1]],-nu[1]);
         //if (nuk[2] not_eq -1) rkf[i] = rkf[i] * __powu( c[nuk[2]],-nu[2]);
                               rkf[i] = rkf[i] * pow( c[nuk[1]],-nu[1]);
         if (nuk[2] not_eq -1) rkf[i] = rkf[i] * pow( c[nuk[2]],-nu[2]);
      }

      if (nuk[4] not_eq -1)
      {
         //                      rkr[i] = rkr[i] * __powu( c[nuk[4]], nu[4]);
         //if (nuk[5] not_eq -1) rkr[i] = rkr[i] * __powu( c[nuk[5]], nu[5]);
                               rkr[i] = rkr[i] * pow( c[nuk[4]], nu[4]);
         if (nuk[5] not_eq -1) rkr[i] = rkr[i] * pow( c[nuk[5]], nu[5]);
      }

      //printf("%3d, %e, %e, %e\n", i, rkf[i], rkr[i], rkf[i]-rkr[i]);
   }
}
template <typename ValueType>
void ckratc (const ValueType &T, const ValueType c[], ValueType ctb[], ValueType rkf[], ValueType rkr[], const CKData &ck, false_type)
{
   const int kk = ck.n_species;
   const int ii = ck.n_reactions;

   const ValueType logT = log(T);

   // II. Concentration-dependent rates ...

   for (int i = 0; i < ii; ++i)
      ctb[i] = 1.0;

   // Third-body reactions ...
   if (ck.n_thdbdy > 0)
   {
      double _time0 = WallClock();

      ValueType ctot(0.0);
      for (int k = 0; k < kk; ++k)
         ctot += c[k];

      for (int n = 0; n < ck.n_thdbdy; ++n)
      {
         const int rxn_idx = ck.rx_thdbdy_idx[n];

         ctb[rxn_idx] = ctot;

         // Add in the specific efficiencies ...

         for (int m = ck.rx_thdbdy_offset[n]; m < ck.rx_thdbdy_offset[n+1]; ++m)
         {
            const int k = ck.rx_thdbdy_spidx[m];
            ctb[rxn_idx] += c[k] * (ck.rx_thdbdy_alpha[m] - 1.0);
         }
      }

      if (do_profile) thdbdy_time += (WallClock() - _time0);
   }

   // Fall-off pressure dependencies ...
   if (ck.n_falloff > 0)
   {
      double _time0 = WallClock();

      #pragma ivdep
      for (int n = 0; n < ck.n_falloff; ++n)
      {
         const int rxn_idx = ck.rx_falloff_idx[n];

         // Concentration of the third-body ... could be a specific species, too.
         ValueType cthb;
         if (ck.rx_falloff_spidx[n] != -1)
         {
            cthb = ctb[rxn_idx];
            ctb[rxn_idx] = 1.0;
         }
         else
            cthb = c[ ck.rx_falloff_spidx[n] ];

         const double *fpar = &ck.rx_falloff_params[n*rx_max_falloff];

         // Low-pressure limit rate ...
         ValueType rklow = fpar[0] * exp(logT * fpar[1] - fpar[2] / T);

         // Reduced pressure ...
         ValueType pr    = rklow * cthb / rkf[rxn_idx];

         // Correction ... k_infty (pr / (1+pr)) * F()
         ValueType p_cor;

         // Different F()'s ...
         //if (ck.rx_info[rxn_idx] & rx_flag_falloff_sri)
         //{
         //   printf("SRI fall-off rxn not ready\n");
         //   exit(-1);
         //}
         //else if (ck.rx_info[rxn_idx] & rx_flag_falloff_troe)
         if (ck.rx_info[rxn_idx] & rx_flag_falloff_troe)
         {
            // 3-parameter Troe form ...
            ValueType Fcent = (1.0 - fpar[3]) * exp(-T / fpar[4]) + fpar[3] * exp(-T / fpar[5]);

            // Additional 4th (T**) parameter ...
            if (ck.rx_info[rxn_idx] & rx_flag_falloff_troe4)
               Fcent += exp(-fpar[6] / T);
               //Fcent += exp(-fpar[6] * invT);

            ValueType log_Fc = log10( fmax(Fcent,small) );
            ValueType eta    = 0.75 - log_Fc * 1.27;
            ValueType log_pr = log10( fmax(pr,small) );
            ValueType plus_c = log_pr - (0.4 + 0.67 * log_Fc);
            ValueType log_F  = log_Fc / (1.0 + sqr(plus_c / (eta - 0.14 * plus_c)));
            ValueType Fc     = exp10(log_F);

            p_cor = Fc * (pr / (pr + 1.0));
         }
         else // Lindermann form
         {
            p_cor = pr / (pr + 1.0);
         }

         rkf[rxn_idx] *= p_cor;
         rkr[rxn_idx] *= p_cor;
         //printf("%3d, %3d, %e, %e\n", n, rxn_idx, ck.rx_info[rxn_idx], p_cor, cthb);
      }

      if (do_profile) falloff_time += (WallClock() - _time0);
   } // fall-off's

   // II. Stoichiometry rates ...

   assert (rx_max_order == 3);

   for (int i = 0; i < ii; ++i)
   {
      const int *nu  = &ck.rx_nu[i*rx_max_order*2];
      const int *nuk = &ck.rx_nuk[i*rx_max_order*2];

      rkf[i] *= ctb[i];
      rkr[i] *= ctb[i];

                               rkf[i] *= pow( c[nuk[0]],-nu[0]);
      if (nuk[1] not_eq -1) {  rkf[i] *= pow( c[nuk[1]],-nu[1]);
         if (nuk[2] not_eq -1) rkf[i] *= pow( c[nuk[2]],-nu[2]);
      }

                               rkr[i] *= pow( c[nuk[3]], nu[3]);
      if (nuk[4] not_eq -1) {  rkr[i] *= pow( c[nuk[4]], nu[4]);
         if (nuk[5] not_eq -1) rkr[i] *= pow( c[nuk[5]], nu[5]);
      }
   }
}


void ckwyp (const double& p, const double& T, double y[], double wdot[], const CKData& ck, int _lenrwk = 0, double *_rwk = NULL)
{
   double _time_start = WallClock();

   const int kk = ck.n_species;
   const int ii = ck.n_reactions;

   const int lenrwk = kk + 4*ii;
#if 0
#ifdef _OPENMP
#warning 'using thread-private allocation for ckwyp scratch'
   std::vector<double> _rwk(lenrwk);
   double *rwk = &_rwk[0];
#else
   assert(not(lenrwk > ck.rwk.size()));
   double *rwk = const_cast< double* >(&ck.rwk[0]);
#endif
#else
   std::vector<double> _vtmp;
   double *rwk;
   if (_rwk == NULL)
   {
      printf("allocated temp space in ckwyp\n");
      _vtmp.resize(lenrwk);
      rwk = &_vtmp[0];
   }
   else
   {
      assert(_lenrwk >= lenrwk);
      rwk = _rwk;
   }
#endif

//   std::vector<double> _c(kk);
//   std::vector<double> _rkf(ii);
//   std::vector<double> _rkr(ii);
//   std::vector<double> _ctb(ii);
//   std::vector<double> _eqk(ii);

//   double *smh = &_c[0];
//   double *ctb = &_ctb[0];
//   double *eqk = ctb;
//   double *rkf = &_rkf[0];
//   double *rkr = &_rkr[0];
   double *rkf = rwk;
   double *rkr = rkf + ii;
   double *ctb = rkr + ii;
   double *c   = ctb + ii;
   double *smh = c;
   double *eqk = ctb;

   // Compute temperature-dependent forward/reverse rates ... mol / cm^3 / s
   double _time0 = WallClock();
   details::ckratt (T, smh, eqk, rkf, rkr, ck, details::true_type());
   if (do_profile) ckratt_time += (WallClock() - _time0);

   // Convert to molar concentrations ... mol / cm^3
   ckytcp (p, T, y, c, ck);

   // Compute concentration-dependent forward/reverse rates ... mol / cm^3 / s
   double _time1 = WallClock();
   details::ckratc (T, c, ctb, rkf, rkr, ck, details::true_type());
   if (do_profile) ckratc_time += (WallClock() - _time1);

   // Compute species net production rates ... mol / cm^3 / s

   //std::fill(wdot, wdot + kk, 0.0);
   for (int k = 0; k < kk; ++k)
      wdot[k] = 0.0;

#if 0
   for (int n = 0; n < rx_max_order*2; ++n)
   {
      for (int i = 0; i < ii; ++i)
      {
         const double rop = rkf[i] - rkr[i];
         const int nu = ck.rx_nu[i*rx_max_order*2+n];
         const int k  = ck.rx_nuk[i*rx_max_order*2+n];
         if (k not_eq -1)
            wdot[k] += (rkf[i] - rkr[i])*nu;
      }
   }
#else
   for (int i = 0; i < ii; ++i)
   {
      const double rop = rkf[i] - rkr[i];

      const int *nu  = &ck.rx_nu[i*rx_max_order*2];
      const int *nuk = &ck.rx_nuk[i*rx_max_order*2];

      /*if (nuk[0] not_eq -1 and nu[0]) wdot[nuk[0]] += (nu[0] * rop);
      if (nuk[1] not_eq -1 and nu[1]) wdot[nuk[1]] += (nu[1] * rop);
      if (nuk[2] not_eq -1 and nu[2]) wdot[nuk[2]] += (nu[2] * rop);
      if (nuk[3] not_eq -1 and nu[3]) wdot[nuk[3]] += (nu[3] * rop);
      if (nuk[4] not_eq -1 and nu[4]) wdot[nuk[4]] += (nu[4] * rop);
      if (nuk[5] not_eq -1 and nu[5]) wdot[nuk[5]] += (nu[5] * rop);*/

                               wdot[nuk[0]] += (rop * nu[0]);
      if (nuk[1] not_eq -1) {  wdot[nuk[1]] += (rop * nu[1]);
         if (nuk[2] not_eq -1) wdot[nuk[2]] += (rop * nu[2]);
      }

                               wdot[nuk[3]] += (rop * nu[3]);
      if (nuk[4] not_eq -1) {  wdot[nuk[4]] += (rop * nu[4]);
         if (nuk[5] not_eq -1) wdot[nuk[5]] += (rop * nu[5]);
      }
   }
#endif

   if (do_profile) ckwyp_time += (WallClock() - _time_start);

#if 0
   {
      FILE *fp = fopen("ckrat.bin","w");
      assert(not(fp == NULL));
      fwrite(&ck.n_reactions, sizeof(int), 1, fp);
      fwrite(&rkf[0], sizeof(double), ck.n_reactions, fp);
      fwrite(&rkr[0], sizeof(double), ck.n_reactions, fp);
      fwrite(&wdot[0], sizeof(double), ck.n_species, fp);
      fclose(fp);
   }
#endif
}

} // end namespace details

template <typename ValueType>
void ckwyp (const ValueType& p, const ValueType& T, ValueType y[], ValueType wdot[], const CKData& ck, ValueType *tmp = NULL)
{
   double _time_start = WallClock();

   const int kk = ck.n_species;
   const int ii = ck.n_reactions;

   const int lenrwk = kk + 4*ii;
   std::vector<ValueType> v_tmp;
   ValueType *rwk;

   if (tmp == NULL)
   {
      printf("allocated temp space in ckwyp\n");
      v_tmp.resize(lenrwk);
      rwk = &v_tmp[0];
   }
   else
   {
      //assert(_lenrwk >= lenrwk);
      rwk = tmp;
   }

   ValueType *rkf = rwk;
   ValueType *rkr = rkf + ii;
   ValueType *ctb = rkr + ii;
   ValueType *c   = ctb + ii;
   ValueType *smh = c;
   ValueType *eqk = ctb;

   // Compute temperature-dependent forward/reverse rates ... mol / cm^3 / s
   double _time0 = WallClock();
   //details::ckratt (T, smh, eqk, rkf, rkr, ck, details::false_type());
   details::ckratt (T, smh, eqk, rkf, rkr, ck, typename details::is_scalar<ValueType>::type());
   if (do_profile) ckratt_time += (WallClock() - _time0);

   // Convert to molar concentrations ... mol / cm^3
   ckytcp (p, T, y, c, ck);

   // Compute concentration-dependent forward/reverse rates ... mol / cm^3 / s
   double _time1 = WallClock();
   details::ckratc (T, c, ctb, rkf, rkr, ck, typename details::is_scalar<ValueType>::type());
   if (do_profile) ckratc_time += (WallClock() - _time1);

   // Compute species net production rates ... mol / cm^3 / s

   for (int k = 0; k < kk; ++k)
      wdot[k] = 0.0;

#if 0
   for (int n = 0; n < rx_max_order*2; ++n)
   {
      for (int i = 0; i < ii; ++i)
      {
         const double rop = rkf[i] - rkr[i];
         const int nu = ck.rx_nu[i*rx_max_order*2+n];
         const int k  = ck.rx_nuk[i*rx_max_order*2+n];
         if (k not_eq -1)
            wdot[k] += (rkf[i] - rkr[i])*nu;
      }
   }
#else
   for (int i = 0; i < ii; ++i)
   {
      const ValueType rop = rkf[i] - rkr[i];

      const int *nu  = &ck.rx_nu[i*rx_max_order*2];
      const int *nuk = &ck.rx_nuk[i*rx_max_order*2];

      /*if (nuk[0] not_eq -1 and nu[0]) wdot[nuk[0]] += (nu[0] * rop);
      if (nuk[1] not_eq -1 and nu[1]) wdot[nuk[1]] += (nu[1] * rop);
      if (nuk[2] not_eq -1 and nu[2]) wdot[nuk[2]] += (nu[2] * rop);
      if (nuk[3] not_eq -1 and nu[3]) wdot[nuk[3]] += (nu[3] * rop);
      if (nuk[4] not_eq -1 and nu[4]) wdot[nuk[4]] += (nu[4] * rop);
      if (nuk[5] not_eq -1 and nu[5]) wdot[nuk[5]] += (nu[5] * rop);*/

                               wdot[nuk[0]] += (rop * (double)nu[0]);
      if (nuk[1] not_eq -1) {  wdot[nuk[1]] += (rop * (double)nu[1]);
         if (nuk[2] not_eq -1) wdot[nuk[2]] += (rop * (double)nu[2]);
      }

                               wdot[nuk[3]] += (rop * (double)nu[3]);
      if (nuk[4] not_eq -1) {  wdot[nuk[4]] += (rop * (double)nu[4]);
         if (nuk[5] not_eq -1) wdot[nuk[5]] += (rop * (double)nu[5]);
      }
   }
#endif

   if (do_profile) ckwyp_time += (WallClock() - _time_start);
}

template <bool solve_enthalpy>
void _solve_temp (const double &h, double &T, double y[], const CKData& ck, const double dTtol = 1.0e-04)
{
   //const double ttol = 1.0e-9; // tolerance for temp O(10^3)
   //const double htol = 1.0e-3; // absolute error tolerance for h O(10^10)
//   const double tol = 1.0e-10;
   const int max_iters = 20;

//   const double htol = 0; //h * tol;
//   const double ttol = tol; //* T;

   bool converged(0);

   ++num_temp_solves;

   // Get initial values at T ...
   double h_new  = (solve_enthalpy) ? ckhbms(T, y, ck) : ckubms(T, y, ck);
   double cp_new = (solve_enthalpy) ? ckcpbs(T, y, ck) : ckcvbs(T, y, ck);

   for (int niters = 0; niters < max_iters && not(converged); ++niters)
   {
      ++num_temp_iters;

      // current h (or u) and cp (or cv)
      double h_old  = h_new;
      double cp_old = cp_new;

      assert(cp_old > small);

      // correction ...
      double dT = (h - h_old) / cp_old;

      // update state ...
      T += dT;
      h_new  = (solve_enthalpy) ? ckhbms(T, y, ck) : ckubms(T, y, ck);
      cp_new = (solve_enthalpy) ? ckcpbs(T, y, ck) : ckcvbs(T, y, ck);

      // error test ...
      double h_err = fabs(h - h_new); // absolute error
      double h_nrm = h_err / fmax(h,cp_new*dTtol); // relative error

      if (fabs(dT) < dTtol or h_nrm < 1.0e-5*dTtol) converged = true;
      //}
      //else
      //   converged = true;

      //printf("n=%d, h_err=%e, dT=%e, T=%f\n", niters, h_err, dT, T);
   }

   if (not(converged))
   {
      fprintf(stderr,"newton iteration for temperatured failed\n");
      exit(-1);
   }
}

// Compute temperature given mixture enthalpy and mass fraction ...
void ckthy (const double &h, double &T, double y[], const CKData& ck)
{
   _solve_temp<true> (h, T, y, ck, 1.0e-4);
}
// Compute temperature given mixture internal energy and mass fraction ...
void cktuy (const double &u, double &T, double y[], const CKData& ck)
{
   _solve_temp<false> (u, T, y, ck, 1.0e-4);
}

} // namespace CK

#endif // ifdef
