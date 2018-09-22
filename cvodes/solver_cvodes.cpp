/**
 * \file
 * \brief The integration driver for the CVODE solver
 *
 * \author Nicholas Curtis
 * \date 03/10/2015
 *
 */

#include "solver_cvodes.hpp"
#include <cstdlib>
#include <cstring>

/* CVODES INCLUDES */
extern "C"
{
    #include "sundials/sundials_types.h"
    #include "sundials/sundials_math.h"
    #include "sundials/sundials_nvector.h"
    #include "nvector/nvector_serial.h"
    #include "cvodes/cvodes.h"
    #include "cvodes/cvodes_lapack.h"
}

namespace c_solvers {

    /**
     * \brief Driver function for CVODE integrator.
     *
     * \param[in,out] t     The time (starting, then ending).
     * \param[in] tEnd      The desired end time.
     * \param[in] pr        A parameter used for pressure or density to pass to the derivative function.
     * \param[in,out] y     Dependent variable array, integrated values replace initial conditions.
     */
    ErrorCode CVODEIntegrator::integrate (
        const double t_start, const double t_end, const double pr, double* __restrict__ y)
    {
        int tid = omp_get_thread_num();

        // local array with initial values
        N_Vector fill = y_locals[tid].vector;

        // copy initial values into NVector array
        double* __restrict__ y_local = NV_DATA_S(fill);
        std::memcpy(y_local, y, _neq * sizeof(double));

        //reinit this integrator for time t, w/ updated state
        CVODEErrorCheck(CVodeReInit(integrators[tid].get(), t_start, fill));

        //set user data to Pr
        double pr_local = pr;
        CVODEErrorCheck(CVodeSetUserData(integrators[tid].get(), &pr_local));

        //set end time
        CVODEErrorCheck(CVodeSetStopTime(integrators[tid].get(), t_end));

        // call integrator for one time step
        double t_next;
        CVODEErrorCheck(CVode(integrators[tid].get(), t_end, fill, &t_next, CV_NORMAL));
        if (t_next != t_end)
        {
            std::cerr << "Error on integration step for thread %d " << tid;
            exit(-1);
        }
        // copy back integrated values
        std::memcpy(y, y_local, _neq * sizeof(double));
    }

}
