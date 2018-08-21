#include <stdio.h>
#include <stdlib.h>

#include <cmath>
#include <float.h>
#include <string>
#include <vector>

#include <fstream>
#include <sstream>
#include <iostream>

#ifdef DEBUG
   #ifdef NDEBUG
      #undef NDEBUG
   #endif
#else
   #define NDEBUG
#endif
#include <assert.h>

#include <clock.h>

//#include <ck.c>
#include <cklib.h>
#include <rk.h>
#include <ros.h>
#include <sdirk.h>

// Don't solve the last species. Needed for PyJac w/ analytical Jacobian.
static bool skipLastSpecies = false;
static bool usePyJac = false;

#ifdef __ENABLE_PYJAC
# warning 'Enabled PyJac RHS'
namespace pyjac
{
#ifndef __restrict__
#define __restrict__
#endif
   extern "C" void dydt (const double, const double, const double * __restrict__, double * __restrict__);
   extern "C" void eval_jacob (const double, const double, const double * __restrict__, double * __restrict__);

   int N2_idx = -1;

} // namespace-pyjac
#endif

#if defined(__ENABLE_OPENCL) && (__ENABLE_OPENCL != 0)
   extern "C" {
      //void cl_driver (double p, double T, double *u, ckdata_t*, double *udot, int numProblems, rk_t *rk);
      void cl_ck_driver (double p, double T, double *u, ckdata_t*, double *udot, int numProblems);
      void cl_ck_driver_array (double *p, double *T, double *u, ckdata_t*, double *udot, int numProblems);
      void cl_rk_driver (double p, double *u_in, double *u_out, ckdata_t*, rk_t *rk, int numProblems);
      void cl_ros_driver (double p, double *u_in, double *u_out, ckdata_t*, ros_t *ros, int numProblems);
   }
#endif

#if defined(__ENABLE_TCHEM) && (__ENABLE_TCHEM != 0)
namespace TChem
{

extern "C"
{
#include <TC_interface.h>
#include <TC_defs.h>
}

bool isInit_ = false;
int nsp_ = 0;
std::vector<double> ytmp_, fytmp_, Jytmp_;
bool enableRHS_ = false;
bool enableJacobian_ = false;
int nrhs_ = 0, njac_ = 0;

void init(int nsp)
{
   if (isInit_ == false)
   {
      isInit_ = true;

      {
         char *env = getenv("ENABLE_TCHEM");
         if (env)
         {
            if (isdigit(*env))
               enableRHS_ = bool(atoi(env));
            else
               enableRHS_ = true;
         }
      }
      {
         char *env = getenv("ENABLE_TCHEM_JACOBIAN");
         if (env)
         {
            if (isdigit(*env))
               enableJacobian_ = bool(atoi(env));
            else
               enableJacobian_ = true;
         }
      }

      if (enableJacobian_ and not(enableRHS_))
      {
         fprintf(stderr,"TChem RHS was enabled since Jacobian was enabled.\n");
         enableRHS_ = true;
      }

      if (not(enableRHS_))
      {
         fprintf(stderr,"TChem disabled at run-time\n");
         return;
      }

      std::string cheminp = "chem.inp";
      std::string thermdat = "therm.dat";
      TC_initChem (const_cast<char*>(cheminp.c_str()), const_cast<char*>(thermdat.c_str()), 0, 0.2);

      nsp_ = nsp;

      int neq = nsp_+1;

      ytmp_.resize(neq);
      fytmp_.resize(neq);

      if (enableJacobian_)
         Jytmp_.resize(neq*neq);

      printf("TChem initialized with %d species, AnalyticalJacobian = %d\n", nsp_, enableJacobian_);

      /* Internal TChem variables */
      TC_abstol = 1.0e-9;
      TC_reltol = 1.0e-11;

      printf("TChem tolerance = %e %e\n", TC_abstol, TC_reltol);
   }
}
void rhs (const int neq, const double t, double *y, double *fy)
{
   //init(nsp);
   assert (isInit_ and enableRHS_);

   nrhs_++;

   //TC_setThermoPres (p);
   const int nsp = neq-1;

   ytmp_[0] = y[nsp];
   for (int i = 0; i < nsp; ++i)
   {
      //ytmp_[i+1] = fmax(y[i], 1.e-64);
      ytmp_[i+1] = y[i];
      //ytmp_[i+1] = fmin( fmax(y[i], 1.e-64), 1.0);
   }

   TC_getSrc (&ytmp_[0], neq, &fytmp_[0] );

   fy[nsp] = fytmp_[0];
   for (int i = 0; i < nsp; ++i)
      fy[i] = fytmp_[i+1];

   //return true;
}
int rhs (const int neq, const double t, double *y, double *fy, void *vdata)
{
   rhs (neq, t, y, fy);
   return 0;
}
//bool jac (const int nsp, const double p, const double T, double *y, double *Jac)
void jac (const int neq, const double t, double *y, double *Jy)
{
   //init(nsp);
   assert (isInit_ and enableJacobian_);

   njac_++;

   static bool has_been_called = false;
   if (not(has_been_called))
   {
      fprintf(stderr,"Inside TChem::jac\n");
      has_been_called = true;
   }

   //const int neq = nsp+1;
   const int nsp = neq-1;

   //TC_setThermoPres (p);

   //ytmp_[0] = T;
   ytmp_[0] = y[nsp];
   for (int i = 0; i < nsp; ++i)
   {
      ytmp_[i+1] = y[i];
      //ytmp_[i+1] = fmax(y[i], 1.e-64);
      //ytmp_[i+1] = std::fmin( std::fmax( y[i], 1e-30), 1.0 );
   }

   TC_getJacTYNanl (&ytmp_[0], nsp, &Jytmp_[0] );
   //TC_getJacTYN (&ytmp_[0], nsp, &Jytmp_[0], 0 /* Analytical? */ );

   // Rotate entries to match the Y_k,T ordering.

   #define Jy_out(_i,_j) ( Jy[(_i) + (_j)*neq] )
   #define Jy_in(_i,_j) ( Jytmp_[ (_i) + (_j)*neq] )

   for (int i = 0; i < nsp; ++i)
   {
      for (int j = 0; j < nsp; ++j)
         Jy_out(i,j) = Jy_in(i+1,j+1);

      Jy_out(i,nsp) = Jy_in(i+1,0);
   }

   for (int j = 0; j < nsp; ++j)
      Jy_out(nsp,j) = Jy_in(0,j+1);

   Jy_out(nsp,nsp) = Jy_in(0,0);

   //for (int j = 0; j < neq; ++j)
   //   for (int i = 0; i < neq; ++i)
   //      printf("tcjac: %d %d %e\n", i, j, Jy_out(i,j));
   //exit(1);

   #undef Jy_out
   #undef Jy_in

   //return true;
}
int jac (const int neq, const double t, double *y, double *Jy, void *vdata)
{
   jac(neq, t, y, Jy);
   return 0;
}

}
#endif

//#define USE_SUNDIALS
#ifdef USE_SUNDIALS

   #include <cvodes/cvodes.h>             /* prototypes for CVODE fcts., consts. */
   #include <nvector/nvector_serial.h>  /* serial N_Vector types, fcts., macros */
   #include <cvodes/cvodes_dense.h>       /* prototype for CVDense */
   #include <sundials/sundials_dense.h> /* definitions DlsMat DENSE_ELEM */
   #include <sundials/sundials_types.h> /* definition of type realtype */
   #ifdef USE_SUNDIALS_LAPACK
      #include <cvodes/cvodes_lapack.h>    /* prototype for CVLapackDense */
   #endif

   #define SUNDIALS_VERSION_MAJOR	(2)
   #define SUNDIALS_VERSION_MINOR	(5)
   #if defined(SUNDIALS_VERSION_MAJOR) && defined(SUNDIALS_VERSION_MINOR)
      #if (SUNDIALS_VERSION_MAJOR >= 2 && SUNDIALS_VERSION_MINOR > 4)
         #warning 'using long int for SUNDIALS integers'
         #define SUNDIALS_INTTYPE	long int
      #endif
   #endif
   #if defined(SUNDIALS_INTTYPE)
      typedef SUNDIALS_INTTYPE SUNDIALS_inttype;
   #else
      typedef int SUNDIALS_inttype;
   #endif
   #undef SUNDIALS_INTTYPE

#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#include <ck.hpp>

#ifdef USE_SUNDIALS
#include <cv_integrator.h>
#endif

void ckfunc (int neq, double time, double y[], double ydot[], void *v_user_data);

struct user_data_t
{
   const CK::CKData& ck;

   std::vector<double> rwk;
   std::vector<int> iwk;

   double pres;
   double temp;
   double rho;

   int iterate_for_temp;
   double h0;
   int constant_pres;

   double f_time;
   double jac_time;

   double temp_time;

   user_data_t(const CK::CKData& _ck) :
      ck(_ck),
      pres(-1), temp(-1), rho(-1),
      iterate_for_temp(0),
      constant_pres(1),
      h0(0),
      f_time(0), jac_time(0), temp_time(0)
   {}

   void operator() (int neq, double time, double y[], double ydot[])
   {
      //printf("user_data_t operator()\n");
      ckfunc(neq, time, y, ydot, this);
   }
   void operator() (int neq, double y[], double ydot[])
   {
      this->operator()(neq, 0.0, y, ydot);
   }
};

void ckfunc (int neq, double time, double y[], double ydot[], void *v_user_data)
{
   double time_start = WallClock();

   user_data_t *user_data = (user_data_t *) v_user_data;
   const CK::CKData &ck = user_data->ck;

   //printf("neq=%d, ck.n_species=%d\n", neq, ck.n_species);

   const int kk = (user_data->iterate_for_temp) ? neq : neq-1;
   //printf("kk=%d neq=%d, ck.n_species=%d\n", kk, neq, ck.n_species);
   assert(ck.n_species == kk);

   //if (user_data->rwk.size() < kk) user_data->rwk.resize(kk);
   int lenrwk = kk + kk + 4*ck.n_reactions;
   if (user_data->rwk.size() < lenrwk) user_data->rwk.resize(lenrwk);

   double *h = &(user_data->rwk[0]);

   //std::fill (ydot, ydot + kk, 0.0);
   for (int k = 0; k < neq; ++k)
      ydot[k] = 0.0;

   double pres, rho, temp;

   if (user_data->iterate_for_temp)
   {
      // compute temp from constant H or U
      if (user_data->constant_pres)
         CK::ckthy (user_data->h0, user_data->temp, y, ck);
      else
         CK::cktuy (user_data->h0, user_data->temp, y, ck);

      temp = user_data->temp;
   }
   else
   {
      temp = y[neq-1];
   }

   if (user_data->constant_pres)
   {
      /* Compute local density given pressure+temp+yk */
      pres = user_data->pres; // in CGS units dyne/cm^2
      rho = CK::ckrhoy(pres, temp, y, ck);
   }
   else
   {
      /* Compute local pressure given density+temp+yk */
      rho  = user_data->rho;
      pres = CK::ckpy(rho, temp, y, ck);
   }
   //printf("temp, rho, pres=%f, %e, %e\n", temp, rho, pres);

   /* Compute molar reaction rate. (mol / (s*cm^3) */
   //CK::ckwyp(pres, temp, y, ydot, ck, lenrwk-kk, h + kk);
   CK::ckwyp(pres, temp, y, ydot, ck, h + kk);

   /* Extract the molecular weights of the species ... this could just be a pointer. */
   double *mwt = CK::ckwt(ck);

   /* Convert from molar to mass units. */
   double rhoinv = 1.0 / rho;
   for (int k = 0; k < kk; k++)
      ydot[k] *= (mwt[k] * rhoinv);

   if (not(user_data->iterate_for_temp))
   {
      /* Compute mixture Cp or Cv (ergs / gm*K) */
      // ... and... 
      /* Compute temperature-dependent species energy (ergs / gm) */
      double cp_mix;
      if (user_data->constant_pres)
      {
         cp_mix = CK::ckcpbs(temp, y, ck);
         CK::ckhms(temp, h, ck);
      }
      else
      {
         cp_mix = CK::ckcvbs(temp, y, ck);
         CK::ckums(temp, h, ck);
      }

      double sum_ydot(0), delT(0);

      for (int k = 0; k < kk; k++)
      {
         //ydot[neq-1] += (h[k] * ydot[k]);
         delT += (h[k] * ydot[k]);
         sum_ydot += ydot[k];
      }

      //ydot[neq-1] = -ydot[neq-1] / cp_mix;
      ydot[neq-1] = -delT / cp_mix;

      //printf("sum_ydot=%le, Tdot=%le\n", sum_ydot, ydot[neq-1]);
   }

   user_data->f_time += (WallClock() - time_start);

   return;
}

static user_data_t *static_user_data = NULL;
void static_ckfunc (int& neq, double& time, double y[], double ydot[])
{
   assert(static_user_data != NULL);
   ckfunc (neq, time, y, ydot, static_user_data);
}

struct cklib_functor
{
   const ckdata_t *ck_;

   double pres_;

   int lenrwk_;
   double *rwk_;

   cklib_functor(const ckdata_t *ckptr) :
      ck_(ckptr),
      pres_(-1),
      lenrwk_(0), rwk_(NULL)
   {
      lenrwk_ = ck_lenrwk(ck_);
      rwk_ = (double *) malloc(sizeof(double)*lenrwk_);
   }

   ~cklib_functor()
   {
      if (rwk_)
         free(rwk_);
   }

   void operator() (const int &neq, double y[], double ydot[])
   {
      this->operator()(neq, 0.0, y, ydot);
   }
   void operator() (const int &neq, const double &time, double y[], double ydot[])
   {
      double time_start = WallClock();
      const int kk = this->ck_->n_species;

      if (usePyJac)
      {
#ifdef __ENABLE_PYJAC
         if ( skipLastSpecies == false )
         {
            fprintf(stderr,"Must use skipLastSpecies for pyJac\n");
            exit(-1);
         }

         if ( neq != kk )
         {
            fprintf(stderr,"Equation set is not ==kk\n");
            exit(-1);
         }

         VectorType<double> y_in_(neq), dy_out_(neq);
         y_in_[0] = y[neq-1];
         for (int k = 0; k < kk-1; ++k)
            y_in_[k+1] = y[k];

         pyjac::dydt( time, 0.1*this->pres_, &y_in_[0], &dy_out_[0] );

         ydot[neq-1] = dy_out_[0];
         for (int k = 0; k < kk-1; ++k)
            ydot[k] = dy_out_[k+1];

         //for (int k = 0; k < neq; ++k)
         //   printf("pyJac: ydot[%d] = %e\n", k, ydot[k]);

         //const int iN2 = pyjac::N2_idx;
         //if (iN2 < 0 or iN2 >= kk)
         //{
         //   fprintf(stderr,"pyjac: not able to find N2 %d\n", iN2);
         //   exit(-1);
         //}

         //std::vector<double> pyjac_in( neq );
         //std::vector<double> pyjac_out( neq );

         //pyjac_in[0] = y[kk]; // T
         //for (int k = 0; k < iN2; ++k)
         //   pyjac_in[k+1] = y[k];

         //pyjac_in[kk] = y[iN2];

         //for (int k = iN2+1; k < kk; ++k)
         //   pyjac_in[k+1-1] = y[k];

         //pyjac::dydt( time, 0.1*this->pres_, &pyjac_in[0], &pyjac_out[0] );

         //ydot[kk] = pyjac_out[0];
         //for (int k = 0; k < iN2; ++k)
         //   ydot[k] = pyjac_out[k+1];

         //ydot[iN2] = pyjac_out[kk];

         //for (int k = iN2+1; k < kk; ++k)
         //   ydot[k] = pyjac_out[k+1-1];

         //for (int k = 0; k < neq; ++k)
         //   printf("pyJac N2: ydot[%d] = %e\n", k, ydot[k]);

         return;
#else
         fprintf(stderr,"pyJac requested but not available at run-time\n");
         exit(-1);
#endif
      }

#if defined(__ENABLE_TCHEM) && (__ENABLE_TCHEM != 0)
      if (TChem::enableRHS_)
      {
         assert( skipLastSpecies == false ); // Not supported for TChem (yet).

         TChem::rhs (neq, time, y, ydot);
         return;
      }
#endif

      if ( skipLastSpecies )
      {
         const double T = y[neq-1];
         VectorType<double> z(kk+1), zdot(kk+1);

         double ysum = 0;
         for (int k = 0; k < kk-1; ++k)
         {
            ysum += y[k];
            z[k] = y[k];
         }
         double y_last = 1.0 - ysum;

         z[kk-1]  = y_last;

         ckrhs (this->pres_, T, &z[0], &zdot[0], this->ck_, rwk_);

         for (int k = 0; k < kk-1; ++k)
            ydot[k] = zdot[k];
         ydot[neq-1] = zdot[kk];
      }
      else
         ckrhs (this->pres_, y[neq-1], y, ydot, this->ck_, rwk_);

      //for (int k = 0; k < neq; ++k)
      //   printf("cklib: ydot[%d] = %e\n", k, ydot[k]);
      //exit(-1);

#if 0
      for (int k = 0; k < neq; ++k)
         ydot[k] = 0.0;

      const double& T = y[neq-1];
      const double& p( this->pres_ ); // in CGS units dyne/cm^2

      /* Compute local density given p/T/y_k */
      const double rho = ckrhoy(p, T, y, this->ck_);

      /* Compute molar reaction rate. (mol / (s*cm^3) */
      ckwyp (p, T, y, ydot, this->ck_);//, this->rwk_);

      /* Extract the molecular weights of the species ... this could just be a pointer. */
      double *mwt = ckwt(this->ck_);

      /* Compute mixture Cp or Cv (ergs / gm*K)			*/
      /* ... and... 						*/
      /* Compute temperature-dependent species energy (ergs / gm)	*/
      double cp_mix;
      //VectorType *h = &(user_data->rwk[0]);
      //VectorType *h = new VectorType [kk];
      //double *h = this->rwk_;
      double *h = ckrwk(this->ck_);//this->rwk_;

      //if (user_data->constant_pres)
      {
         cp_mix = ckcpbs(T, y, ck_);
         ckhms(T, h, ck_);
      }
      //else
      //{
      //   cp_mix = CK::ckcvbs(T, y, ck);
      //   CK::ckums(T, h, ck);
      //}

      for (int k = 0; k < kk; k++)
      {
         /* Convert from molar to mass units. */
         ydot[k] *= mwt[k];
         ydot[k] /= rho;

         //std::cout << k << " :" << ydot[k] << std::endl;
         //std::cout << k << " :" << h[k] << std::endl;

         /* Sum up the net enthalpy change. */
         h[k] *= ydot[k];
         ydot[neq-1] -= h[k];
      }

      ydot[neq-1] /= cp_mix;

      //std::cout << ydot[neq-1] << std::endl;

      //printf("sum_ydot=%le, Tdot=%le\n", sum_ydot, ydot[neq-1]);

      //this->f_time += (WallClock() - time_start);

      //delete [] h;
#endif
      return;
   }

   int jac (const int &neq, double t, double y[], double dfdy[])
   {
      const int kk = ck_->n_species;

      if (usePyJac)
      {
#ifdef __ENABLE_PYJAC
         if ( skipLastSpecies == false )
         {
            fprintf(stderr,"Must enable skipLastSpecies with pyJac %s %d\n", __FILE__, __LINE__);
            exit(-1);
         }
         if ( neq != kk )
         {
            fprintf(stderr,"Must solve KK-1 species with pyJac %d %s %d\n", neq, __FILE__, __LINE__);
            exit(-1);
         }

         VectorType<double> y_in_(neq), df_out_(neq*neq);

         const int kkm1 = kk-1;

         y_in_[0] = y[neq-1];
         for (int k = 0; k < kkm1; ++k)
            y_in_[k+1] = y[k];

         pyjac::eval_jacob( t, 0.1*this->pres_, &y_in_[0], &df_out_[0] );

#define  jac_out(i,j) dfdy[(i) + (j)*neq]

// pyJac uses C row-major ordering ... faster along the row.
#define  jac_in(i,j) df_out_[(i) + (j)*neq]
//#define  jac_in(i,j) jtmp[(i) + (j)*neq]

         // dYi/dYj
         for (int j = 0; j < kkm1; ++j)
            for (int i = 0; i < kkm1; ++i)
               jac_out(i,j) = jac_in(i+1,j+1);

         // dYi/dT
         for (int i = 0; i < kkm1; ++i)
            jac_out(i,kkm1) = jac_in(i+1,0);

         // dT/dYj
         for (int j = 0; j < kkm1; ++j)
            jac_out(kkm1,j) = jac_in(0,j+1);

         // dT/dT
         jac_out(kkm1,kkm1) = jac_in(0,0);

         //for (int j = 0; j < neq; ++j)
         //   for (int i = 0; i < neq; ++i)
         //      printf("pyjac: %d %d %e\n", i, j, jac_out(i,j));
         //exit(1);
#undef jac_in
#undef jac_out

#else
         fprintf(stderr,"pyJac Jacobian requested but not available at run-time.\n");
         exit(-1);
#endif
      }
#if defined(__ENABLE_TCHEM) && (__ENABLE_TCHEM != 0)
      else if (TChem::enableJacobian_)
         TChem::jac( neq, t, y, dfdy);
#endif
      else
      {
         fprintf(stderr,"Analytical Jacobian called but not enabled.\n");
         exit(2);
      }

      return 0;
   }
};
int cklib_functor_callback (const int neq, const double time, double y[], double ydot[], void *vptr)
{
   cklib_functor *ck_ = (cklib_functor *) vptr;

   (*ck_)(neq, time, y, ydot);

   return 0;
}
int cklib_callback (const int neq, const double time, double y[], double ydot[], void *vptr)
{
   return cklib_functor_callback (neq, time, y, ydot, vptr);
}

//template <typename ValueType, typename _T, class Func>
template <typename ValueType, typename _T, class Solver, class Func>
int cv_driver (int neq, ValueType u_in[], const _T& t_stop, Solver &solver, Func &func, const bool write_data)
{
   double time_start = WallClock();

   ValueType *u = u_in;

   /* Loop over all the problems ... */

   //const int nsteps = 512 * 13 * 2; //100;
   const int nsteps = (write_data == true) ? 1000 : 1;
   _T t0 = 0.0;
   _T dt = t_stop / double(nsteps);

   ValueType t(t0);

   int ierr = solver.init (t, t_stop, u, func);
   if (ierr)
   {
      std::cerr << "Error in CV::Integrator::init()" << std::endl;
      return 1;
   }
   //return 1;

   FILE *data_out = NULL;
   if (write_data)
   {
#ifdef _OPENMP
      fprintf(stderr,"Can not do data save in parallel\n");
      exit(-1);
#endif
      data_out = fopen("cv.out","w");
      if (data_out == NULL)
      {
         fprintf(stderr,"Error opening cv.out\n");
         exit(-1);
      }
      fwrite(&nsteps, sizeof(int), 1, data_out);
      fwrite(&dt, sizeof(double), 1, data_out);
   }

   int nst = 0;

   for (int i = 0; i < nsteps; ++i)
   {
      if (write_data && data_out)
         fwrite(u, sizeof(double), neq, data_out);

      _T t_next = fmin(t + dt, t_stop);
      if (i == nsteps-1) t_next = t_stop;
      double _t0 = WallClock();

      const int itask = CV_NORMAL;
      solver.solve(t, t_next, u, func, itask);

      if (nsteps > 1)
      {
         double _t1 = WallClock();
         //std::cout << "step: " << i << std::endl;
         //std::cout << "   t: " << t << std::endl;
         //std::cout << "   T: " << u[neq-1] << std::endl;
         //std::cout << " nst: " << solver.nst << std::endl;
         //std::cout << "time: " << 1000*(_t1-_t0) << std::endl;
         printf("%5d %g, %g, %d, %d, %g\n", i, t, u[neq-1], solver.nst, nst, 1000*(_t1-_t0));
      }

      nst += solver.nst;

      if (write_data && data_out)
         fwrite(u, sizeof(double), neq, data_out);
   }

   double time_stop = WallClock();

   //printf("solver time = %f (ms) %d\n", 1000*(WallClock() - time_start), solver.nst);
   //printf("func   time = %f (ms)\n", user_data.f_time*1000);
   int thread_id = 0, num_threads = 1;
#ifdef _OPENMP
   thread_id = omp_get_thread_num();
   num_threads = omp_get_num_threads();
//   std::cout << thread_id << "/" << num_threads << ": " 1000*(time_stop - time_start) << std::endl;
#endif
   //std::cout << thread_id << "/" << num_threads << " nst: " << solver.nst << ": " << 1000*(time_stop - time_start) << std::endl;

   if (write_data && data_out)
      fclose(data_out);

   return solver.nst;
}

int main (int argc, char* argv[])
{
   CK::CKData ck;

   FILE *ckfile = NULL;

   std::string ckname("ck.bin");

   typedef enum { None, RK, ROS, SDIRK, CV } solverTag_t;

   int num_problems = 1;
   //int use_rk  = 0;
   //int use_ros = 1;
   //int use_cv  = 0;
   solverTag_t solver_tag = None;
   double delta= 1;
   double t_stop = 0.001;
   int cl_iters = 1;
   int write_data = 0;
   int read_data = 0;
   int read_csv = 1;
   char *read_file = NULL;
   int omp_chunk_size = 1;
   bool nohost = false;

   {
      for (int index = 1; index < argc; ++index)
      {
         printf("argv[%d] = %s\n", index, argv[index]);

      //int index = 1;

      //while (index < argc)
      //{
         if (strcmp(argv[index], "-ck") == 0)
         {
            index++;
            assert (index < argc);
            //ckname = std::string(argv[index++]);
            ckname = std::string(argv[index]);
         }
         else if (strcmp(argv[index], "-np") == 0)
         {
            index++;
            assert (index < argc);
            if (isdigit(*argv[index]))
               num_problems = atoi(argv[index]);
            //index++;
         }
         else if (strcmp(argv[index], "-omp_chunk_size") == 0)
         {
            index++;
            assert (index < argc);
            if (isdigit(*argv[index]))
               omp_chunk_size = atoi(argv[index]);
         }
         else if (strcmp(argv[index], "-cl_iters") == 0)
         {
            index++;
            assert (index < argc);
            if (isdigit(*argv[index]))
               cl_iters = atoi(argv[index]);
            //index++;
         }
         else if (strcmp(argv[index], "-tstop") == 0)
         {
            index++;
            assert (index < argc);
            if (isdigit(*argv[index]))
               t_stop = atof(argv[index]);
         }
         else if (strcmp(argv[index], "-delta") == 0)
         {
            index++;
            assert (index < argc);
            if (isdigit(*argv[index]))
               delta = atof(argv[index]);
         }
         else if (strcmp(argv[index], "-write") == 0)
         {
            write_data = 1;
         }
         else if (strcmp(argv[index], "-read") == 0)
         {
            index++;
            assert (index < argc);
            read_file = argv[index];
            read_data = 1;
         }
         else if (strcmp(argv[index], "-binary") == 0)
         {
            read_csv = 0;
         }
         else if (strcmp(argv[index], "-csv") == 0)
         {
            read_csv = 1;
         }
         else if (strcmp(argv[index], "-pyjac") == 0)
         {
            usePyJac = true;
#ifndef __ENABLE_PYJAC
            fprintf(stderr,"PyJac enabled at run-time but not compiled.\n");
            return -1;
#endif
         }
         else if (strcmp(argv[index], "-nohost") == 0)
         {
            nohost = true;
         }
         else if (strcmp(argv[index], "-ros") == 0)
         {
            solver_tag = ROS;
         }
         else if (strcmp(argv[index], "-rk") == 0)
         {
            solver_tag = RK;
         }
         else if (strcmp(argv[index], "-sdirk") == 0)
         {
            solver_tag = SDIRK;
         }
         else if (strcmp(argv[index], "-cv") == 0)
         {
            solver_tag = CV;
         }
      }
   }

   if (read_data && write_data)
   {
      fprintf(stderr,"Can not read and write data at the same time\n");
      return 1;
   }
   if (write_data && num_problems > 1)
   {
      fprintf(stderr,"Can not write data with multiple problems\n");
      return 1;
   }

   ckfile = fopen(ckname.c_str(),"r");
   if (not(ckfile))
   {
      fprintf(stderr,"error opening ckfile %s", ckname.c_str());
      return 1;
   }
   else
   {
      CK::ckinit (ck, ckfile);

      fclose(ckfile);

      if (0)
      {
         CK::CKData ck_test;
         ckfile = fopen("ck_test.bin","r");
         if (not(ckfile))
         {
            fprintf(stderr,"error opening ckfile");
            return 1;
         }
         CK::ckinit (ck_test, ckfile);
         fclose(ckfile);

         if (0)
            for (int k = 0; k < ck.n_species; ++k)
            {
               printf("%3d: mwt=%e, tmid=%e\n", k,
                               ck.sp_mwt[k] - ck_test.sp_mwt[k],
                               ck.th_tmid[k] - ck_test.th_tmid[k]);
               for (int j = 0; j < CK::th_max_terms; ++j)
                  printf("alo[%d]=%e, ahi=%e\n", j,
                                ck.th_alo[k*CK::th_max_terms+j] - ck_test.th_alo[k*CK::th_max_terms+j],
                                ck.th_ahi[k*CK::th_max_terms+j] - ck_test.th_ahi[k*CK::th_max_terms+j]);
            }

         if (0)
            for (int i = 0; i < ck.n_reactions; ++i)
            {
               printf("%3d: A=%e, b=%e, Ea=%e\n", i,
                               ck.rx_A[i] - ck_test.rx_A[i],
                               ck.rx_b[i] - ck_test.rx_b[i],
                               ck.rx_E[i] - ck_test.rx_E[i]);
            }

         return 0;
      }
   }

   double T = 1001.0;
   double p = CK::PA;

   std::vector<double> x(ck.n_species);
   std::vector<double> y(ck.n_species);
   std::vector<double> c(ck.n_species);
   std::vector<double> h(ck.n_species);
   std::vector<double> cp(ck.n_species);

   //std::vector<double> u_out;
   VectorType<double> u_in;
   VectorType<double> u_out;

   int iH2=-1, iO2=-1, iN2=-1;
   for (int k = 0; k < ck.n_species; ++k)
      if      (ck.sp_name[k].compare("O2") == 0) iO2 = k;
      else if (ck.sp_name[k].compare("H2") == 0) iH2 = k;
      else if (ck.sp_name[k].compare("N2") == 0) iN2 = k;

   printf("iH2=%d, iO2=%d, iN2=%d\n", iH2, iO2, iN2);

#ifdef __ENABLE_PYJAC
   pyjac::N2_idx = iN2;
#endif

   x[iH2] = 2.0; x[iO2] = 1.0; x[iN2] = 4.0;
   //x[iH2] = 2.0; x[iO2] = 1.0; x[iN2] = 0.0;

   double x_sum(0);
   for (int k = 0; k < ck.n_species; ++k)
      x_sum += x[k];

   for (int k = 0; k < ck.n_species; ++k)
      x[k] /= x_sum;

   CK::ckxty (&x[0], &y[0], ck);
   CK::ckytcp (p, T, &y[0], &c[0], ck);

   double rho = CK::ckrhoy(p, T, &y[0], ck);

   printf("cp_mass=%e, cv_mass=%e, h_mass=%e, u_mass=%e, rho=%e\n", CK::ckcpbs(T,&y[0],ck), CK::ckcvbs(T,&y[0],ck), CK::ckhbms(T,&y[0],ck), CK::ckubms(T,&y[0],ck), rho);

#if defined(__ENABLE_TCHEM)
   TChem::init(ck.n_species);
   TChem::TC_setThermoPres (p/10.0);
#endif

   ckdata_t *ckptr = NULL;

   if (1)
   {
      std::vector<double> wdot(ck.n_species);
      std::vector<double> h(ck.n_species);

      CK::ckwyp (p, T, &y[0], &wdot[0], ck);

      double *mwt = CK::ckwt(ck);

      CK::ckhms(T, &h[0], ck);
      double cp_mix = CK::ckcpbs(T, &y[0], ck);

      double sum_ydot(0);
      double Tdot(0);
      printf("y, c, h, wdot, ydot\n");
      for (int k = 0; k < ck.n_species; ++k)
      {
         double ydot_k = wdot[k] * mwt[k] / rho;
         printf("%3d: %e, %e, %e, %e, %e\n", k, y[k], c[k], h[k], wdot[k], ydot_k);
         sum_ydot += ydot_k;
         Tdot += h[k] * ydot_k;
      }
      printf("sum(ydot) = %15.9e, Tdot = %15.9e\n", sum_ydot, -Tdot/cp_mix);

      std::vector<char *> sp_name_(ck.n_species);
      for (int k = 0; k < ck.n_species; ++k)
         sp_name_[k] = const_cast<char*>(ck.sp_name[k].c_str());

      std::vector<int> rx_falloff_type_(ck.n_falloff+1);
      for (int i = 0; i < ck.n_falloff; ++i)
      {
         int k = ck.rx_falloff_idx[i];
         int type = 0;
         if (!(ck.rx_info[k] & CK::rx_flag_falloff))
            printf("not a falloff %d %d\n", i, k);
         if (ck.rx_info[k] & CK::rx_flag_falloff_sri)
         {
            type = 1;
            if (ck.rx_info[k] & CK::rx_flag_falloff_sri5)
               type = 2;
         }
         else if (ck.rx_info[k] & CK::rx_flag_falloff_troe)
         {
            type = 3;
            if (ck.rx_info[k] & CK::rx_flag_falloff_troe4)
               type = 4;
         }
         rx_falloff_type_[i] = type;
         //printf("%d %d %d %d\n", i, k, ck.rx_info[k], type);
      }

      ckptr = ck_create (
         ck.n_species,
         &sp_name_[0], ck.sp_mwt.getPointer(), ck.th_tmid.getPointer(), ck.th_alo.getPointer(), ck.th_ahi.getPointer(),
         ck.n_reactions,
         ck.rx_A.getPointer(), ck.rx_b.getPointer(), ck.rx_E.getPointer(), ck.rx_nu.getPointer(), ck.rx_nuk.getPointer(),
         ck.n_reversible_reactions,
         ck.rx_rev_idx.getPointer(), ck.rx_rev_A.getPointer(), ck.rx_rev_b.getPointer(), ck.rx_rev_E.getPointer(),
         ck.n_irreversible_reactions,
         ck.rx_irrev_idx.getPointer(),
         ck.n_thdbdy,
         ck.rx_thdbdy_idx.getPointer(), ck.rx_thdbdy_offset.getPointer(), ck.rx_thdbdy_spidx.getPointer(), ck.rx_thdbdy_alpha.getPointer(),
         ck.n_falloff,
         ck.rx_falloff_idx.getPointer(), &rx_falloff_type_[0], ck.rx_falloff_spidx.getPointer(), ck.rx_falloff_params.getPointer());
   }

   double *yptr = &y[0];
   double ysum(0);

   const int kk = ck.n_species;

   double time_start = WallClock();

   CK::CKData *ck_ptr = &ck;

   const int vector_width = 1;
   //const double delta = .01; // % deviation on T0
   delta /= 100.; // % deviation on T0

   printf("alignment = %d\n", (*ck_ptr).rx_A.alignment);

#ifdef _OPENMP
   //num_problems *= omp_get_max_threads();
   int max_threads = omp_get_max_threads();
   if (max_threads > 1 and num_problems % max_threads and false)
   {
      int round_up = num_problems % max_threads;
      round_up = max_threads - round_up;
      printf("adjusting num_problems to be multiple of max_threads %d %d %d\n", max_threads, num_problems, round_up);
      num_problems += round_up;
   }
#endif

   if (usePyJac)
      std::cout << "Using PyJac RHS instead of internal functions." << std::endl;
#ifdef __ENABLE_PYJAC
   else
      std::cout << "PyJac disabled at run-time." << std::endl;
#endif

   std::cout << "num_problems = " << num_problems << " delta = " << delta << std::endl;

   double dT = 500. / num_problems;

   //const bool use_cv = false; //true;
   //const double t_stop = (use_cv) ? 0.001 : 0.0001;

   // Define the test problem set size.
   if (u_out.size() == 0 and num_problems > 0)
      u_out.resize(num_problems*(ck.n_species+1));
   if (u_in.size() == 0 and num_problems > 0)
      u_in.resize(num_problems*(ck.n_species+1));

   // Load problem data from a file.
   if (read_data)
   {
      printf("read_csv= %d\n", read_csv);
      const int neq = ck.n_species + 1;
      int np = -1;

      // ASCII CSV path.
      if (read_csv)
      {
         // First, load all of the file into a list of strings for each line.
         std::vector <std::vector <std::string> > file_data;
         std::ifstream infile( read_file );

         while (infile)
         {
            std::string s;
            if (!std::getline( infile, s )) break;

            //std::cout << "line: " << s << std::endl;

            std::istringstream ss( s );
            std::vector <std::string> record;

            while (ss)
            {
               std::string s;
               if (!std::getline( ss, s, ',' )) break;
               record.push_back( s );
               //std::cout << s<< std::endl;
               //size_t pos = 0;
               //while (s[pos] == ' ') pos++;
               //record.push_back( std::string(s, pos, std::string::npos) );
            }

            file_data.push_back( record );
         }
         if (!infile.eof())
         {
            std::cerr << "Fooey!\n";
            return -1;
         }

         np = atoi( file_data[0][0].c_str() );
         int nsp = atoi( file_data[0][1].c_str() );
         if (nsp != ck.n_species)
         {
            fprintf(stderr,"Input profile file error: # species not correct %d %d\n", nsp, ck.n_species);
            return -1;
         }

         printf("num_problems (in) = %d %d\n", num_problems, np);

         //if (np < num_problems)
         //   num_problems = np;

         //t_stop = dt;

         for (int i = 0; i < std::min(np, num_problems); ++i)
         {
            double x_, v_ = 0, T_, p_;
            VectorType<double> yk_(ck.n_species);

            x_ = strtod( file_data[i+1][0].c_str(), NULL );
            T_ = strtod( file_data[i+1][1].c_str(), NULL );
            p_ = strtod( file_data[i+1][2].c_str(), NULL );

            for (int k = 0; k < ck.n_species; ++k)
               yk_[k] = strtod( file_data[i+1][k+3].c_str(), NULL );

            memcpy( &u_in[i*neq], yk_.getPointer(), sizeof(double)*ck.n_species);
            u_in[i*neq+ck.n_species] = T_;
            //printf("%d %f\n", i, T_);
         }
      }
      // Binary path.
      else
      {
         FILE *data_in = fopen(read_file,"r");
         if (data_in == NULL)
         {
            fprintf(stderr,"error opening data input file %s\n", read_file);
            return 2;
         }

         fread(&np, sizeof(int), 1, data_in);

         printf("num_problems (in) = %d %d\n", num_problems, np);

         //if (np < num_problems)
         //   num_problems = np;

         double dt;
         fread(&dt, sizeof(double), 1, data_in);
         printf("dt = %e\n", dt);

         //for (int i = 0; i < num_problems; ++i)
         for (int i = 0; i < std::min(np, num_problems); ++i)
         {
            double x_, v_, T_, p_;
            VectorType<double> yk_(ck.n_species);

            fread(&x_, sizeof(double), 1, data_in);
            fread(&v_, sizeof(double), 1, data_in);
            fread(&T_, sizeof(double), 1, data_in);
            fread(&p_, sizeof(double), 1, data_in);
            fread(yk_.getPointer(), sizeof(double), ck.n_species, data_in);

            memcpy( &u_in[i*neq], yk_.getPointer(), sizeof(double)*ck.n_species);
            u_in[i*neq+ck.n_species] = T_;
            //printf("%d %f\n", i, T_);
         }

         fclose(data_in);
      }

      printf("t_stop = %e\n", t_stop);
      //printf("np (after) = %d\n", np);
      //return 0;

      /* Now, fill in the rest of the slots */
      if (np < num_problems)
      {
         printf("replicating inputs to fill %d problems\n", num_problems);
         for (int i = np; i < num_problems; ++i)
         {
            int j = i % np;
            //printf("i,j=%d,%d\n", i,j);
            //for (int k = 0; k < neq; ++k)
            //   u_in[i*neq+k] = u_in[j*neq+k];
            memcpy( &u_in[i*neq], &u_in[j*neq], sizeof(double)*neq);
            memcpy( &u_out[i*neq], &u_out[j*neq], sizeof(double)*neq);
         }
      }
   }
   else
   {
      // Else, make up a mock problem set.
      for (int i = 0; i < num_problems; ++i)
      {
         //double T0 = T * (1.0 + delta * drand48());
         double T0 = T * (1.0 + (i*delta));
         //std::cout << "T0 = " << T0 << std::endl;
         //printf("T[%d] = %f\n", i, T0);

         for (int k = 0; k < kk; ++k)
            u_in[i*(ck.n_species+1) + k] = y[k];

         u_in[i*(ck.n_species+1) + kk] = T0;
      }
   }

   int nst_ = 0, nit_ = 0;

   // Don't run on the host if requested.
   if (nohost == false)
   {

   VectorType<double> cv_err(kk+1), cv_ref(kk+1);
   for (int i = 0; i < kk+1; ++i)
      cv_err[i] = cv_ref[i] = 0;

   #pragma omp parallel reduction (+:ysum, nst_, nit_)
   {
      int thread_id = 0, num_threads = 1;
#ifdef _OPENMP
      thread_id = omp_get_thread_num();
      num_threads = omp_get_num_threads();
      #pragma omp master
      {
         std::cout << "num_threads = " << num_threads << std::endl;
      }
#endif
      user_data_t user_data(*ck_ptr);

      user_data.rwk.resize(kk);

      user_data.iterate_for_temp = 0;
      user_data.constant_pres = 1;

      const int neq = (skipLastSpecies == true) ? kk : kk+1;

      VectorType<double> u(neq);

      cklib_functor cklib_func(ckptr);

#ifdef USE_SUNDIALS
       bool use_analytical_jacobian = false;

#if defined(__ENABLE_PYJAC) || (defined(__ENABLE_TCHEM) && (__ENABLE_TCHEM != 0))
       if (usePyJac or TChem::enableJacobian_)
          use_analytical_jacobian = true;
#endif

       CV::Integrator< cklib_functor > cv_cklib_obj(neq, use_analytical_jacobian);
#endif

      cklib_func.pres_ = p;

      VectorType<double> _rwk;
      VectorType<int> _iwk;

      const int stride = 1;
      const int chunk_size = omp_chunk_size;

      #pragma omp for schedule(dynamic,chunk_size) //schedule(static,1)
      for (int problem_id = 0; problem_id < num_problems; problem_id += stride)
      {
         double t0 = WallClock();

         //for (int k = 0; k < kk; ++k)
         if (skipLastSpecies)
         {
            u[neq-1] = u_in[problem_id * (kk+1) + kk];
            for (int k = 0; k < kk-1; ++k)
               u[k] = u_in[problem_id * (kk+1) + k];
         }
         else
            for (int k = 0; k < neq; ++k)
               u[k] = u_in[problem_id * neq + k];

         double To = u[neq-1];

         //u[neq-1] = T0;

         user_data.pres   = p;
         //user_data.temp   = T0;
         //user_data.rho    = CK::ckrhoy(p, T0, &u[0], *ck_ptr);

         //if (use_cv)
         if (solver_tag == CV)
         {
#ifdef USE_SUNDIALS
            nst_ += cv_driver (neq, u.getPointer(), t_stop, cv_cklib_obj, cklib_func, write_data);

            int write_stride = std::max(1,num_problems / 16);
            if (problem_id % write_stride == 0)
               printf("%d: %d %d %f %f\n", problem_id, cv_cklib_obj.nst, cv_cklib_obj.nfe, u[neq-1], To);
#if (defined(__ENABLE_TCHEM) && (__ENABLE_TCHEM != 0))
            printf("nrhs_= %d %d\n", TChem::nrhs_, TChem::njac_);
#endif
#else
             std::cerr << "recompile with -DUSE_SUNDIALS" << std::endl;
#endif
         }
         else if (solver_tag == RK)
         {
            rk_t rk_;
            rk_counters_t counters_;

            double t_ = 0, h_ = 0;
            rk_create (&rk_, neq);

            rk_.max_iters = 1000;
            rk_.min_iters = 1;

            int _lrwk = rk_lenrwk (&rk_);
            //static VectorType<double> rk_rwk;
            //if (rk_rwk.size() != _lrwk) rk_rwk.resize(_lrwk);
            if (_rwk.size() != _lrwk) _rwk.resize(_lrwk);

            rk_init (&rk_, t_, t_stop);

            //rk_solve (&rk_, &t_, &h_, &u[0], rk_rwk, cklib_functor_callback, &cklib_func);
            int ierr_ = rk_solve (&rk_, &t_, &h_, &counters_, &u[0], &_rwk[0], NULL, &cklib_func);
            if (ierr_ != RK_SUCCESS)
            {
               fprintf(stderr,"%d: rk_solve error %d %d %d\n", problem_id, ierr_, counters_.niters, rk_.max_iters);
               exit(-1);
            }

            nst_ += counters_.nsteps;
            nit_ += counters_.niters;

            rk_destroy(&rk_);

            //free(rk_rwk);

            //printf("%d: %d %d\n", problem_id*vector_width+i, counters_.nsteps, counters_.niters);
            int write_stride = std::max(1,num_problems / 16);
            if (problem_id % write_stride == 0)
               printf("%d: %d %d %f %f\n", problem_id, counters_.nsteps, counters_.niters, u[neq-1], u_in[problem_id * neq + (neq-1)]);
         }
         else if (solver_tag == ROS)
         {
            ros_t ros_;
            ros_counters_t counters_;
            double t_ = 0, h_ = 0;
            ros_create (&ros_, neq, Ros4);

            int _lrwk = ros_lenrwk (&ros_);
            int _liwk = ros_leniwk (&ros_);
            //static VectorType<double> ros_rwk;
            if (_rwk.size() != _lrwk) _rwk.resize(_lrwk);
            //static VectorType<int> ros_iwk;
            if (_iwk.size() != _liwk) _iwk.resize(_liwk);

            ros_init (&ros_, t_, t_stop);

#if defined(__ENABLE_TCHEM) //&& (0)
            if (TChem::enableJacobian_)
            {
               //ros_solve (&ros_, &t_, &h_, &counters_, &u[0], ros_iwk, ros_rwk, cklib_callback, TChem::jac, &cklib_func);
               ros_solve (&ros_, &t_, &h_, &counters_, &u[0], &_iwk[0], &_rwk[0], TChem::rhs, TChem::jac, NULL);
            }
            else
#endif
            {
               //ros_solve (&ros_, &t_, &h_, &counters_, &u[0], ros_iwk, ros_rwk, cklib_callback, NULL, &cklib_func);
               ros_solve (&ros_, &t_, &h_, &counters_, &u[0], &_iwk[0], &_rwk[0], NULL, NULL, &cklib_func);
            }

            nst_ += counters_.nst;
            nit_ += counters_.niters;

            ros_destroy(&ros_);
            //free(ros_rwk);
            //free(ros_iwk);

            int write_stride = std::max(1,num_problems / 16);
            if (problem_id % write_stride == 0)
               printf("%d: %d %d %f %f\n", problem_id, counters_.nst, counters_.niters, u[neq-1], To);
         }
         else if (solver_tag == SDIRK)
         {
            sdirk_t sdirk_obj;
            sdirk_counters_t counters_;
            double t_ = 0, h_ = 0;
            sdirk_create (&sdirk_obj, neq, S4a);

            int _lrwk = sdirk_lenrwk (&sdirk_obj);
            int _liwk = sdirk_leniwk (&sdirk_obj);
            if (_rwk.size() != _lrwk) _rwk.resize(_lrwk);
            if (_iwk.size() != _liwk) _iwk.resize(_liwk);

            sdirk_init (&sdirk_obj, t_, t_stop);

#if defined(__ENABLE_TCHEM)
            if (TChem::enableJacobian_)
            {
               sdirk_solve (&sdirk_obj, &t_, &h_, &counters_, &u[0], &_iwk[0], &_rwk[0], TChem::rhs, TChem::jac, NULL);
               printf("nrhs_= %d %d\n", TChem::nrhs_, TChem::njac_);
            }
            else
#endif
            {
               sdirk_solve (&sdirk_obj, &t_, &h_, &counters_, &u[0], &_iwk[0], &_rwk[0], NULL, NULL, &cklib_func);
            }

            nst_ += counters_.nst;
            nit_ += counters_.niters;

            sdirk_destroy(&sdirk_obj);

            int write_stride = std::max(1,num_problems / 16);
            if (problem_id % write_stride == 0)
               printf("%d: %d %d %f %f\n", problem_id, counters_.nst, counters_.niters, u[neq-1], To);
         }

         ysum += u[neq-1];

         if (u_out.size() > 0 and read_data == 0)
         {
            const double Tout = skipLastSpecies ? u[kk-1] : u[kk];

            if (skipLastSpecies)
            {
               u_out[(kk)+problem_id*(kk+1)] = Tout;
               for (int k = 0; k < kk-1; ++k)
                  u_out[k+problem_id*(kk+1)] = u[k];
            }
            else
               for (int k = 0; k < kk+1; ++k)
                  u_out[k+problem_id*(kk+1)] = u[k];

            if (num_problems < 16)
               printf("u[%d] = %f %f %f\n", problem_id, Tout, u_in[kk+problem_id*neq], 1000*(WallClock() - t0));

         }

         if (read_data)// and (solver_tag == ROS or solver_tag == RK))
         {
            for (int k = 0; k < kk+1; ++k)
            {
               double diff = u[k] - u_out[k+problem_id*neq];
               cv_err[k] += diff*diff;
               cv_ref[k] += u_out[k+problem_id*neq]*u_out[k+problem_id*neq];
            }
         }

#if (0) && defined(_OPENMP)
         printf("%d/%d T0=%f, T=%f, %f\n", thread_id, num_threads, T0, u[neq-1], 1000*(WallClock() - t0));
#endif
      }
   }

   printf("ysum = %f\n", ysum);
   double calc_time = WallClock() - time_start;
   printf("time = %f %f %d\n", calc_time, calc_time / num_problems, num_problems);
   printf("nst = %d, nit = %d\n", nst_, nit_);

   if (read_data)// and (solver_tag == ROS or solver_tag == RK))
   {
      for (int k = 0; k < kk+1; ++k)
      {
         cv_err[k] = sqrt(cv_err[k]) / num_problems;
         cv_ref[k] = sqrt(cv_ref[k]) / num_problems;
         double ref_ = cv_ref[k];
         if (ref_ < 1e-20) ref_ = 1;
         if (cv_ref[k] > sqrt(DBL_EPSILON))
            printf("err[%d] = %e, ref = %e, rel = %e\n", k, cv_err[k], cv_ref[k], cv_err[k] / ref_);
      }
   }

   } // nohost

   for (int iter = 0; iter < cl_iters; iter++)
   {
      double T_ = 500.;

      int numProblems = 10000;
      {
         char *env = getenv("NP");
         if (env)
            if (isdigit(*env))
               numProblems = atoi(env);
      }

      const int neq = ckptr->n_species+1;

      if (numProblems)
      {
         const int lenrwk = (ck_lenrwk(ckptr) + 2*neq);
         double *udot_ref = (double *) malloc(sizeof(double)*neq*numProblems);

         const int kk = ckptr->n_species;
         VectorType<double> T_array, p_array, y_array;
         T_array.resize(numProblems);
         p_array.resize(numProblems);
         y_array.resize(numProblems*kk);

         for (int n = 0; n < numProblems; ++n)
         {
            double T0 = T_ + (1000.*n)/numProblems;
            T_array[n] = T0;
            p_array[n] = p;
            for (int i = 0; i < kk; ++i) y_array[n*kk+i] = y[i];
         }

         int n_threads = 1;
         double t0 = WallClock();
         if (1)
         {
         #pragma omp parallel// reduction(+:err, ref)
         {
            //double *udot = (double *) malloc(sizeof(double)*neq);
            double *rwk = (double *) malloc(sizeof(double)*lenrwk);

#ifdef _OPENMP
            #pragma omp master
            n_threads = omp_get_num_threads();
#endif

            #pragma omp for
            for (int n = 0; n < numProblems; ++n)
            {
               double *udot_ = udot_ref + (neq*n);
               const int idx = 0;

               if (T_array.size() > 0)
                  ckrhs (p_array[n], T_array[n], &y_array[n*kk], udot_, ckptr, rwk);
               else
               {
                  double T0 = T_ + (1000.*n)/numProblems;

                  ckrhs (p, T0, &y[0], udot_, ckptr, rwk);
               }
               //ckwyp (p, T0, u, udot_, ck, rwk);
               //ckytcp (p, T0, u, udot_, ck);
               //ckhms (T0, udot_, ck);
               //udot_[idx] = ckcpbs (T0, u, ck);
               //udot_[idx] = ckrhoy (p, T0, u, ck);
            }

            free(rwk);
         }

         double t1 = WallClock();
         printf("Ref time = %f T=%f n_threads=%d\n", (t1-t0), T_, n_threads);
         }

         // Compare to TChem
#if defined(__ENABLE_TCHEM) && (__ENABLE_TCHEM != 0)
         if (TChem::enableRHS_)
         {
            const int kk = ckptr->n_species;
            const int neq = kk+1;

            //TChem::init(kk);
            TChem::TC_setThermoPres (p/10.0);

            VectorType<double> ftmp(neq), ytmp(neq);
            VectorType<double> err(neq), ref(neq);

            const bool compute_error = 0;

            for (int i = 0; i < neq; ++i)
               err[i] = ref[i] = 0.0;

            for (int k = 0; k < kk; ++k)
               ytmp[k] = y[k];

            double t0_ = WallClock();

            for (int n = 0; n < numProblems; ++n)
            {
               double T0 = T_ + (1000.*n)/numProblems;

               ytmp[kk] = T0;

               TChem::rhs(neq, 0.0, ytmp.getPointer(), ftmp.getPointer());

               //if (n < 10)
               //   printf("Tdot[%d]=%e, Tchem=%e\n", n, udot_ref[neq*n+kk], ftmp[kk]);

               if (compute_error)
                  for (int i = 0; i < neq; ++i)
                  {
                     double diff = udot_ref[neq*n+i] - ftmp[i];
                     err[i] += diff*diff;
                     ref[i] += udot_ref[neq*n+i]*udot_ref[neq*n+i];
                  }
            }

            printf("TChem time = %f\n", WallClock()-t0_);

            if (compute_error)
               for (int i = 0; i < neq; ++i)
               {
                  err[i] = sqrt(err[i]/numProblems);
                  ref[i] = sqrt(ref[i]/numProblems);

                  double ref_ = (ref[i] > 1.0e-20) ? ref[i] : 1.0;
                  if (ref[i] > 1.0e-20)
                     printf("%d err = %e, ref = %e, rel = %e\n", i, err[i], ref[i], err[i] / ref_);
               }
         }
#endif

#if defined(__ENABLE_OPENCL) //&& (__ENABLE_OPENCL != 0)
#warning 'Enabled OpenCL ... calling cl_ck_driver'
         double *udot_ref_ptr = (iter == 0) ? udot_ref : NULL;

         if (T_array.size() > 0)
         {
            printf("calling cl_ck_driver_array %d\n", numProblems);
            cl_ck_driver_array (p_array.getPointer(), T_array.getPointer(), y_array.getPointer(), ckptr, udot_ref_ptr, numProblems);
         }
         else
         {
            printf("calling cl_ck_driver %d\n", numProblems);
            cl_ck_driver (p, T_, &y[0], ckptr, udot_ref_ptr, numProblems);
         }
#endif

         free(udot_ref);
      }

#if defined(__ENABLE_OPENCL) && (__ENABLE_OPENCL != 0)
      if (num_problems > 0)
      {
         const int kk = ckptr->n_species;
         /*VectorType<double> u_in(num_problems * neq);

         for (int i = 0; i < num_problems; ++i)
         {
            //double T0 = T * (1.0 + delta * drand48());
            double T0 = T * (1.0 + (i*delta));
            //std::cout << "T0 = " << T0 << std::endl;
            //printf("T[%d] = %f\n", i, T0);

            for (int k = 0; k < kk; ++k)
               u_in[i*neq + k] = y[k];

            u_in[i*neq + kk] = T0;
         }*/

         if (solver_tag == RK)
         {
            rk_t rk_;
            rk_create (&rk_, neq);
            rk_init (&rk_, 0.0, t_stop);

            cl_rk_driver (p, u_in.getPointer(), u_out.getPointer(), ckptr, &rk_, num_problems);

            rk_destroy(&rk_);
         }
         if (solver_tag == ROS)
         {
            ros_t ros_;
            ros_create (&ros_, neq, Ros4);
            ros_init (&ros_, 0.0, t_stop);

            cl_ros_driver (p, u_in.getPointer(), u_out.getPointer(), ckptr, &ros_, num_problems);

            ros_destroy(&ros_);
         }
      }
#endif
   }

   if (ckptr)
      ck_destroy(&ckptr);

   return 0;
}
