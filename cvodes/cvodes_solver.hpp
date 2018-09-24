/**
 * \file
 * \brief Definition of the RKC CPU solver
 * \author Nicholas Curtis, Kyle Niemeyer
 * \date 09/19/2019
 */

#ifndef CVODES_SOLVER_HPP
#define CVODES_SOLVER_HPP

#include <vector>
#include "solver.hpp"
extern "C"
{
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
}


//! unique_ptr deleter support for CVODE integrator
void delete_integrator(void* integrator)
{
    CVodeFree(&integrator);
}

//! unique_ptr deleter support for N_Vector
void delete_nvector(void* n_vector)
{
    N_VDestroy((N_Vector)n_vector);
}

namespace c_solvers
{

    void CVODEErrorCheck(int code)
    {
        switch (code)
        {
            case CV_SUCCESS:
                return;
            case CV_TSTOP_RETURN:
                return;
            case CV_MEM_FAIL:
                std::cerr << "Memory allocation failed." << std::endl;
            case CV_ILL_INPUT:
                std::cerr << "Illegal value for CVodeInit input argument." << std::endl;
                exit(code);
            case CV_MEM_NULL:
                std::cerr << "CVODEs Memory was not initialized with CVodeInit!" << std::endl;
                exit(code);
            case CV_NO_MALLOC:
                std::cerr << "CVODE memory block not initialized by CVodeCreate." << std::endl;
                exit(code);
            default:
                std::cerr << "Unknown CVODEs error encountered: " << code << std::endl;
                exit(code);
        }
    }

    class CVODEIntegrator : public Integrator
    {
    private:
        void clean()
        {
            //free the integrators and nvectors
            y_locals.clear();
            integrators.clear();
        }

        //! The required memory size of this integrator in bytes.
        //! This is cummulative with any base classes
        std::size_t _ourMemSize;

    protected:
        std::vector<std::unique_ptr<void, void(*)(void *)>> y_locals;
        std::vector<std::unique_ptr<void, void(*)(void *)>> integrators;

        //! the offset to the CVODEs state vectors
        std::size_t _phi_cvodes;

        /*
         * \brief Determines offsets for memory access from #working_buffer in integrator
         */
        std::size_t setOffsets()
        {
            _phi_cvodes = Integrator::requiredSolverMemorySize();
            return _phi_cvodes + _neq * sizeof(double);
        }

        /*
         * \brief Return the required memory size (per-thread) in bytes
         */
        virtual std::size_t requiredSolverMemorySize()
        {
            return _ourMemSize;
        }


    public:

        CVODEIntegrator(int neq, int numThreads, const SolverOptions& options) :
            Integrator(neq, numThreads, options)
        {
            _ourMemSize = this->setOffsets();
            this->reinitialize(numThreads);
        }


        ~CVODEIntegrator()
        {
            this->clean();
        }

        /*!
           \fn char* solverName()
           \brief Returns a descriptive solver name
        */
        const char* solverName() const {
            #ifdef SUNDIALS_ANALYTIC_JACOBIAN
                const char* name = "cvodes-analytic-int";
            #else
                const char* name = "cvodes-int";
            #endif
            return name;
        }

        void reinitialize(int numThreads)
        {
            // clear our memory
            this->clean();
            // re-init base
            Integrator::reinitialize(numThreads);

            for (int i = 0; i < numThreads; i++)
            {
                double* __restrict__ phi = _unique<double>(i, _phi_cvodes);
                // create intgerator
                integrators.push_back(
                    std::unique_ptr<void, void(*)(void *)>(
                        CVodeCreate(CV_BDF, CV_NEWTON),
                        delete_integrator));
                // create N_Vector
                y_locals.push_back(
                    std::unique_ptr<void, void(*)(void*)>(
                        (void*)N_VMake_Serial(_neq, phi),
                        delete_nvector));

                // check
                if (integrators[i] == NULL)
                {
                    printf("Error creating CVodes Integrator\n");
                    exit(-1);
                }

                //initialize
                CVODEErrorCheck(CVodeInit(integrators[i].get(), dydt_cvodes, 0, (N_Vector)y_locals[i].get()));

                //set tolerances
                CVODEErrorCheck(CVodeSStolerances(integrators[i].get(), rtol(), atol()));

                //setup the solver
                CVODEErrorCheck(CVLapackDense(integrators[i].get(), _neq));

                #ifndef FINITE_DIFFERENCE
                    CVODEErrorCheck(CVDlsSetDenseJacFn(integrators[i].get(), eval_jacob_cvodes));
                #endif

                #ifdef CV_MAX_ORD
                    CVODEErrorCheck(CVodeSetMaxOrd(integrators[i].get(), CV_MAX_ORD));
                #endif

                #define CV_MAX_STEPS (100000)
                #ifdef CV_MAX_STEPS
                    CVODEErrorCheck(CVodeSetMaxNumSteps(integrators[i].get(), CV_MAX_STEPS));
                #endif

                #ifdef CV_HMAX
                    CVODEErrorCheck(CVodeSetMaxStep(integrators[i].get(), CV_HMAX));
                #endif
                #ifdef CV_HMIN
                    CVODEErrorCheck(CVodeSetMinStep(integrators[i].get(), CV_HMIN));
                #endif
                #ifdef CV_MAX_ERRTEST_FAILS
                    CVODEErrorCheck(CVodeSetMaxErrTestFails(integrators[i].get(), CV_MAX_ERRTEST_FAILS));
                #endif
                #ifdef CV_MAX_HNIL
                    CVODEErrorCheck(CVodeSetMaxHnilWarns(integrators[i].get(), CV_MAX_HNIL));
                #endif
            }
        }

        /**
         * \brief Driver function for CVODE integrator.
         *
         * \param[in,out] t     The time (starting, then ending).
         * \param[in] tEnd      The desired end time.
         * \param[in] pr        A parameter used for pressure or density to pass to the derivative function.
         * \param[in,out] y     Dependent variable array, integrated values replace initial conditions.
         */
        ErrorCode integrate (
            const double t_start, const double t_end, const double pr, double* __restrict__ y);

    };
}

#endif
