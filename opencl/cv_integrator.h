#ifndef __cv_integrator_h
#define __cv_integrator_h

#include <stdio.h>
#include <stdlib.h>

#include <cmath>
#include <string>

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

#ifdef USE_SUNDIALS

   #include <cvodes/cvodes.h>             /* prototypes for CVODE fcts., consts. */
   #include <nvector/nvector_serial.h>  /* serial N_Vector types, fcts., macros */
   #include <cvodes/cvodes_dense.h>       /* prototype for CVDense */
   #include <cvodes/cvodes_diag.h>       /* prototype for CVDiag */
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

namespace CV
{

template <class Func>
static int CV_UserDefinedRHS (realtype t, N_Vector y, N_Vector ydot, void *user_data)
{
   int neq = NV_LENGTH_S(y);
   double *y_data = NV_DATA_S(y);
   double *ydot_data = NV_DATA_S(ydot);

   Func &func = *((Func *) user_data);

   func (neq, t, y_data, ydot_data);

   return (0);
}

template <class Func>
static int CV_UserDefinedJac (long int N, realtype t, N_Vector y, N_Vector fy, DlsMat Jac, void *user_data,
                       N_Vector tmp1, N_Vector tmp2, N_Vector tmp3)
{
   int neq = NV_LENGTH_S(y);
   double *y_data = NV_DATA_S(y);
   double *fy_data = NV_DATA_S(fy);
   double *Jac_data = Jac->data;

   Func &func = *((Func *) user_data);

   func.jac (neq, t, y_data, Jac_data);

   return (0);
}

template <class Func>
struct Integrator
{
   typedef double value_type;

   int itol;
   double rtol, atol;

   int min_iters, max_iters;

   void *cv_mem;

   const int neq;

   int nst, nfe, nje, nni, nlu;

   Integrator(const int _neq, bool use_analytical_jacobian = false) :
      neq(_neq),
      itol(1), atol(1.e-9), rtol(1.e-11),
      min_iters(0), max_iters(2000),
      cv_mem(NULL),
      nst(0), nfe(0), nje(0), nni(0), nlu(0)
   {
      /* Create CVODE: allocate internal solver memory
         and specify the BDF and Newton options */
      this->cv_mem = CVodeCreate (CV_BDF, CV_NEWTON);
      assert(this->cv_mem);

      /* Create a vector of length NEQ for IC */
      //N_Vector y = N_VMake_Serial(neq,NULL);
      N_Vector y = N_VNew_Serial(neq);

      int ierr;

      /* Initialize CVODE: initialize internal memory, specify the RHS function and IC's */
      ierr = CVodeInit (cv_mem, CV_UserDefinedRHS<Func>, 0.0, y);
      assert(ierr == CV_SUCCESS);

      /* Specify integration tolerances: useing scalar+scalar */
      ierr = CVodeSStolerances (cv_mem, rtol, atol);
      assert(ierr == CV_SUCCESS);

      /* Set the max # of internal steps for CV_NORMAL */
      ierr = CVodeSetMaxNumSteps (cv_mem, max_iters);
      assert(ierr == CV_SUCCESS);

      /* Set the error file to stdout */
      ierr = CVodeSetErrFile (cv_mem, stderr);
      assert(ierr == CV_SUCCESS);

      /* Initialize the SUNDIALS dense matrix object */
      //if (use_cvdiag)
      //{
      //   ierr = CVDiag(this->cv_mem);
      //   assert(ierr == CV_SUCCESS);
      //   printf("initialized CVDiag\n");
      //}
      {
         ierr = CVDense(this->cv_mem, this->neq);
         assert(ierr == CV_SUCCESS);

         if (use_analytical_jacobian)
         {
            /* Set the Jacobian function, if available. */
            ierr = CVDlsSetDenseJacFn (cv_mem, CV_UserDefinedJac<Func>);
            assert(ierr == CV_SUCCESS);
         }
      }

      N_VDestroy_Serial(y);
   }

   ~Integrator()
   {
      if (cv_mem);
         CVodeFree(&cv_mem);
   }

   int init (double &t, const double &t_stop, double y_data[], Func& func)
   {
      /* Create a vector of length NEQ for IC */
      N_Vector y = N_VMake_Serial(neq,y_data);

      /* Set the user-defined-data pointer */
      int ierr = CVodeSetUserData (cv_mem, (void*)&func);
      assert(ierr == CV_SUCCESS);

      this->nst = 0;
      this->nfe = 0;
      this->nje = 0;
      this->nni = 0;
      this->nlu = 0;

      /* Set the user-defined-data pointer */
      ierr = CVodeSetUserData (cv_mem, (void*)&func);
      assert(ierr == CV_SUCCESS);

      ierr = CVodeReInit (cv_mem, t, y);
      assert(ierr == CV_SUCCESS);

      N_VDestroy_Serial(y);

      return 0;
   }

   int solve (double &t, const double &t_stop, double y_data[], Func &func, const int itask = CV_NORMAL)
   {
      N_Vector y = N_VMake_Serial(neq,y_data);

      int ierr = CVodeReInit (cv_mem, t, y);
      assert(ierr == CV_SUCCESS);

      while ((t - t_stop) < 0.0)
      {
         int ierr = CVode (cv_mem, t_stop, y, &t, itask);
         //int ierr = CVode (cv_mem, t_stop, y, &t, CV_ONE_STEP);
         //printf("%e %e\n", t, y_data[neq-1]);

         if (ierr != CV_SUCCESS)
         {
            if (ierr == CV_MEM_NULL || ierr == CV_NO_MALLOC) {
               fprintf(stderr,"CVODE ERROR: ierr == CV_MEM_NULL || ierr == CV_NO_MALLOC\n"); exit(1);
            }
            else if (ierr == CV_TOO_MUCH_WORK) {
               fprintf(stderr,"CVODE WARNING: ierr == CV_TOO_MUCH_WORK: t=%e, t_stop=%e\n", t, t_stop);
            }
            else if (ierr == CV_TOO_MUCH_ACC) {
               fprintf(stderr,"CVODE ERROR: ierr == CV_TOO_MUCH_ACC\n"); exit(1);
            }
            else if (ierr == CV_ERR_FAILURE || ierr == CV_CONV_FAILURE) {
               fprintf(stderr,"CVODE ERROR: ierr == CV_ERR_FAILURE || CV_CONV_FAILURE\n"); exit(1);
            }
            else if (ierr == CV_LINIT_FAIL || ierr == CV_LSETUP_FAIL || ierr == CV_LSOLVE_FAIL) {
               fprintf(stderr,"CVODE ERROR: ierr == CV_LINIT_FAIL || ierr == CV_LSETUP_FAIL || ierr == CV_LSOLVE_FAIL\n"); exit(1);
            }
            /*else if (ierr == CV_SUCCESS) {
               if (t >= t_stop)
                  break; // exit the while loop
               else {
                  //printf("success but haven't reached t_stop yet: t=%e, t_stop=%e\n", t, t_stop);
                  continue; // jump to the start of the loop .. skip the next stuff
               }
            }*/
            else {
               fprintf(stderr,"CVODE ERROR: Unknown error flag = %d\n", ierr); exit(-1);
            }
         }

         long int _nst, _nfe, _nsetups, _nje, _nni, _nfeDQ;
         ierr = CVodeGetNumSteps (cv_mem, &_nst);
         ierr = CVodeGetNumRhsEvals (cv_mem, &_nfe);
         ierr = CVodeGetNumLinSolvSetups (cv_mem, &_nsetups);
         ierr = CVodeGetNumNonlinSolvIters (cv_mem, &_nni);
         ierr = CVDlsGetNumJacEvals(cv_mem, &_nje);
         ierr = CVDlsGetNumRhsEvals(cv_mem, &_nfeDQ);

         this->nfe += _nfe + _nfeDQ;
         this->nst += _nst;
         this->nni += _nni;
         this->nje += _nje;
         this->nlu += _nsetups;
      }

      N_VDestroy_Serial(y);

      return 0;
   }
};

} // end namespace CV

#endif
