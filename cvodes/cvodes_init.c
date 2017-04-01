/*! \file cvodes_init.c
 *  \brief Implementation of the necessary initialization for the CVODE interface
 *
 * \author Nicholas Curtis
 * \date 03/09/2015
 *
 */

#include "header.h"
#include "solver_options.h"
#include "cvodes_dydt.h"

#ifndef FINITE_DIFFERENCE
 	#include "cvodes_jac.h"
#endif

/* CVODES INCLUDES */
#include "sundials/sundials_types.h"
#include "sundials/sundials_math.h"
#include "sundials/sundials_nvector.h"
#include "nvector/nvector_serial.h"
#include "cvodes/cvodes.h"
#include "cvodes/cvodes_lapack.h"

#ifdef GENERATE_DOCS
namespace cvode {
#endif

/** The state vectors used in CVODE operation */
N_Vector *y_locals;
/** The base state vectors used in N_Vector creation */
double* y_local_vectors;
/** The stored CVODE integrator objects */
void** integrators;

/*! \fn void initialize_solver(int num_threads)
   \brief Initializes the solver
   \param num_threads The number of OpenMP threads to use

   Various definitions in the generated options, e.g. #FINITE_DIFFERENCE, #CV_MAX_ERRTEST_FAILS,
   etc. affect the options set for the CVODEs solver. @see solver_options.h

   The RHS function dydt_cvodes() will be used in the solver.

   If #FINITE_DIFFERENCE is not defined, the jacobian function in eval_jacob_cvodes
   will be used in the CVODE solver.  Else, the CVODE finite difference based on the
   RHS function will be used.
*/
 void initialize_solver(int num_threads) {
 	y_locals = (N_Vector*)malloc(num_threads * sizeof(N_Vector));
 	y_local_vectors = (double*)calloc(num_threads * NSP, sizeof(double));
 	integrators = (void**)malloc(num_threads * sizeof(void*));

 	for (int i = 0; i < num_threads; i++)
	{
		integrators[i] = CVodeCreate(CV_BDF, CV_NEWTON);
		y_locals[i] = N_VMake_Serial(NSP, &y_local_vectors[i * NSP]);
		if (integrators[i] == NULL)
		{
			printf("Error creating CVodes Integrator\n");
			exit(-1);
		}

		//initialize
		int flag = CVodeInit(integrators[i], dydt_cvodes, 0, y_locals[i]);
		if (flag != CV_SUCCESS) {
		    if (flag == CV_MEM_FAIL) {
		        printf("Memory allocation failed.\n");
		    } else if (flag == CV_ILL_INPUT) {
		        printf("Illegal value for CVodeInit input argument.\n");
	        } else if (flag == CV_MEM_NULL) {
	        	printf("CVODEs Memory was not initialized with CVodeInit!\n");
		    }
		    exit(flag);
		}

		//set tolerances
		flag = CVodeSStolerances(integrators[i], RTOL, ATOL);
		if (flag != CV_SUCCESS) {
		    if (flag == CV_NO_MALLOC) {
		        printf("CVODE memory block not initialized by CVodeCreate.\n");
		    } else if (flag == CV_ILL_INPUT) {
		        printf("Illegal value for CVodeInit input argument.\n");
	        } else if (flag == CV_MEM_NULL) {
	        	printf("CVODEs Memory was not initialized with CVodeInit!\n");
		    }
		    exit(flag);
		}

    	//setup the solver
	    flag = CVLapackDense(integrators[i], NSP);
		if (flag != CVDLS_SUCCESS) {
		    if (flag == CVDLS_MEM_FAIL) {
		        printf("CVODE memory block not initialized by CVodeCreate.\n");
		    } else if (flag == CVDLS_ILL_INPUT) {
		        printf("Illegal value for CVodeInit input argument.\n");
	        } else if (flag == CVDLS_MEM_NULL) {
	        	printf("CVODEs Memory was not initialized with CVodeInit!\n");
		    }
		    exit(flag);
		}

	    #ifndef FINITE_DIFFERENCE
	    	flag = CVDlsSetDenseJacFn(integrators[i], eval_jacob_cvodes);
	    	if (flag != CV_SUCCESS) {
		    	printf("Error setting analytic jacobian\n");
		    	exit(flag);
	    	}
	    #endif

	    #ifdef CV_MAX_ORD
	        CVodeSetMaxOrd(integrators[i], CV_MAX_ORD);
	        if (flag != CV_SUCCESS) {
		    	printf("Error setting max order\n");
		    	exit(flag);
	    	}
	    #endif

	    #ifdef CV_MAX_STEPS
	        CVodeSetMaxNumSteps(integrators[i], CV_MAX_STEPS);
	        if (flag != CV_SUCCESS) {
		    	printf("Error setting max steps\n");
		    	exit(flag);
	    	}
	    #endif

	    #ifdef CV_HMAX
	        CVodeSetMaxStep(integrators[i], CV_HMAX);
	        if (flag != CV_SUCCESS) {
		    	printf("Error setting max timestep\n");
		    	exit(flag);
	    	}
	    #endif
	    #ifdef CV_HMIN
	        CVodeSetMinStep(integrators[i], CV_HMIN);
	        if (flag != CV_SUCCESS) {
		    	printf("Error setting min timestep\n");
		    	exit(flag);
	    	}
	    #endif
	    #ifdef CV_MAX_ERRTEST_FAILS
	        CVodeSetMaxErrTestFails(integrators[i], CV_MAX_ERRTEST_FAILS);
	        if (flag != CV_SUCCESS) {
		    	printf("Error setting max error test fails\n");
		    	exit(flag);
	    	}
	    #endif
	    #ifdef CV_MAX_HNIL
	        CVodeSetMaxHnilWarns(integrators[i], CV_MAX_HNIL);
	        if (flag != CV_SUCCESS) {
		    	printf("Error setting max hnil warnings\n");
		    	exit(flag);
	    	}
	    #endif
	}
 }

/*!
   \fn void cleanup_solver(int num_threads)
   \brief Cleans up the created solvers
   \param num_threads The number of OpenMP threads used

   Frees and cleans up allocated CVODE memory.
*/
 void cleanup_solver(int num_threads) {
 	//free the integrators and nvectors
	for (int i = 0; i < num_threads; i++)
	{
		CVodeFree(&integrators[i]);
		N_VDestroy(y_locals[i]);
	}
	free(y_locals);
	free(y_local_vectors);
	free(integrators);
 }

/*!
   \fn char* solver_name()
   \brief Returns a descriptive solver name
*/
 const char* solver_name() {
#ifdef SUNDIALS_ANALYTIC_JACOBIAN
 	const char* name = "cvodes-analytic-int";
#else
 	const char* name = "cvodes-int";
#endif
 	return name;
 }

 void init_solver_log() {
 }
 void solver_log() {
 }

#ifdef GENERATE_DOCS
}
#endif
