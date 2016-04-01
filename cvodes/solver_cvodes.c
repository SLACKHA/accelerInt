/*solver_generic.cu
 * the generic integration driver for all GPU solvers
 * \file solver_generic.cu
 *
 * \author Nicholas Curtis
 * \date 03/10/2015
 *
 */

#include "header.h"
#include "solver.h"

/* CVODES INCLUDES */
#include "sundials/sundials_types.h"
#include "sundials/sundials_math.h"
#include "sundials/sundials_nvector.h"
#include "nvector/nvector_serial.h"
#include "cvodes/cvodes.h"
#include "cvodes/cvodes_lapack.h"

extern N_Vector *y_locals;
extern double* y_local_vectors;
extern void** integrators;

void intDriver (const int NUM, const double t, const double t_end,
                const double *pr_global, double *y_global)
{
    int tid;
    double t_next;
    #pragma omp parallel for shared(y_global, pr_global, integrators, y_locals) private(tid, t_next)
    for (tid = 0; tid < NUM; ++tid) {
        int index = omp_get_thread_num();

        // local array with initial values
        N_Vector fill = y_locals[index];
        double pr_local = pr_global[tid];

        // load local array with initial values from global array
        double* y_local = NV_DATA_S(fill);
        
        for (int i = 0; i < NSP; i++)
        {
            y_local[i] = y_global[tid + i * NUM];
        }

        //reinit this integrator for time t, w/ updated state
        int flag = CVodeReInit(integrators[index], t, fill);
        if (flag != CV_SUCCESS)
        {
            printf("Error reinitializing integrator for thread %d, code: %d\n", tid, flag);
            exit(flag);
        }

        //set user data to Pr
        flag = CVodeSetUserData(integrators[index], &pr_local);
        if (flag != CV_SUCCESS)
        {
            printf("Error setting user data for thread %d, code: %d\n", tid, flag);
            exit(flag);
        }

        // call integrator for one time step
        flag = CVode(integrators[index], t_end, fill, &t_next, CV_NORMAL);
        if (flag != CV_SUCCESS || t_next != t_end)
        {
            printf("Error on integration step for thread %d, code %d", tid, flag);
            exit(flag);
        }

        // update global array with integrated values
        
        for (int i = 0; i < NSP; i++)
        {
            y_global[tid + i * NUM] = y_local[i];
        }

    } // end tid loop

} // end intDriver