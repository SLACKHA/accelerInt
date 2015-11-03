/* rb43_init.cu
*  Implementation of the necessary initialization for the 4th order (3rd order embedded) Rosenbrock Solver
 * \file rb43_init.cu
 *
 * \author Nicholas Curtis
 * \date 03/09/2015
 *
 */

#include "header.h"
#include "solver_options.h"
#include "cvodes_dydt.h"

#ifdef SUNDIALS_ANALYTIC_JACOBIAN
 	#include "cvodes_jac.h"
#endif

/* CVODES INCLUDES */
#include "sundials/sundials_types.h"
#include "sundials/sundials_math.h"
#include "sundials/sundials_nvector.h"
#include "nvector/nvector_serial.h"
#include "cvodes/cvodes.h"
#include "cvodes/cvodes_lapack.h"

N_Vector *y_locals;
double* y_local_vectors;
void** integrators;

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
			printf("Error creating CVodes Integrator");
			exit(1);
		}

		//initialize
		int flag = CVodeInit(integrators[i], dydt_cvodes, 0, y_locals[i]);
		if (flag != CV_SUCCESS) {
		    if (flag == CV_MEM_FAIL) {
		        printf("Memory allocation failed.");
	            exit(-1);
		    } else if (flag == CV_ILL_INPUT) {
		        printf("Illegal value for CVodeInit input argument.");
	            exit(-1);
		    } else {
		        printf("CVodeInit failed.");
	            exit(-1);
		    }
		}

		//set tolerances
		flag = CVodeSStolerances(integrators[i], RTOL, ATOL);
		if (flag != CV_SUCCESS) {
	        if (flag == CV_MEM_FAIL) {
	            printf("Memory allocation failed.");
	            exit(-1);
	        } else if (flag == CV_ILL_INPUT) {
	        	printf("Illegal value for CVodeInit input argument.");
	            exit(-1);
	        } else {
	        	printf("CVodeInit failed.");
	            exit(-1);
	        }
    	}

    	//setup the solver
	    flag = CVLapackDense(integrators[i], NSP);

	    if (flag != CV_SUCCESS) {
	    	printf("Error setting up CVODES solver");
	    	exit(-1);
	    }

	    #ifdef SUNDIALS_ANALYTIC_JACOBIAN
	    	flag = CVDlsSetDenseJacFn(integrators[i], eval_jacob_cvodes);
	    #endif

	    if (flag != CV_SUCCESS) {
	    	printf("Error setting analytic jacobian");
	    	exit(-1);
	    }

	    #ifdef CV_MAX_ORD
	        CVodeSetMaxOrd(integrators[i], CV_MAX_ORD);
	    #endif

	    #ifdef CV_MAX_STEPS
	        CVodeSetMaxNumSteps(integrators[i], CV_MAX_STEPS);
	    #endif

	    #ifdef CV_HMAX
	        CVodeSetMaxStep(integrators[i], CV_HMAX);
	    #endif
	    #ifdef CV_HMIN
	        CVodeSetMinStep(integrators[i], CV_HMIN);
	    #endif
	    #ifdef CV_MAX_ERRTEST_FAILS
	        CVodeSetMaxErrTestFails(integrators[i], CV_MAX_ERRTEST_FAILS);
	    #endif
	    #ifdef CV_MAX_HNIL
	        CVodeSetMaxHnilWarns(integrators[i], CV_MAX_HNIL);
	    #endif
	}
 }

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

 const char* solver_name() {
#ifdef SUNDIALS_ANALYTIC_JACOBIAN
 	const char* name = "cvodes-analytical-int";
#else
 	const char* name = "cvodes-int";
#endif
 	return name;
 }

 void init_solver_log() {
 	
 }
 void solver_log() {
 	
 }