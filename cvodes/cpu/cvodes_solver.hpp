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
    #include "cvodes_includes.h"
    #include "cvodes_dydt.h"
    #ifndef FINITE_DIFFERENCE
        #include "cvodes_jac.h"
    #endif
}


//! unique_ptr deleter support for CVODE integrator
#ifdef NEW_SUNDIALS
void delete_sunlinsol(void* linsol)
{
    SUNLinSolFree((SUNLinearSolver)linsol);
}
void delete_sunmat(void* mat)
{
    SUNMatDestroy((SUNMatrix) mat);
}
#endif
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
        #ifdef NEW_SUNDIALS
        std::vector<std::unique_ptr<void, void(*)(void *)>> linsols;
        std::vector<std::unique_ptr<void, void(*)(void *)>> mats;
        #endif
        std::vector<CVUserData> user_data;

        //! the offset to the CVODEs state vectors
        std::size_t _phi_cvodes;

        /**
         * \brief Determines offsets for memory access from #working_buffer in integrator
         */
        std::size_t setOffsets()
        {
            _phi_cvodes = Integrator::requiredSolverMemorySize();
            return _phi_cvodes + _neq * sizeof(double);
        }

        /**
         * \brief Return the required memory size (per-thread) in bytes
         */
        virtual std::size_t requiredSolverMemorySize()
        {
            return _ourMemSize;
        }


    public:

        CVODEIntegrator(int neq, int numThreads, const IVP& ivp, const SolverOptions& options) :
            Integrator(neq, numThreads, ivp, options),
            user_data(numThreads)
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
            const char* name = "cvodes-int";
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
                #ifdef NEW_SUNDIALS
                // create linear solver and matrix
                mats.push_back(
                     std::unique_ptr<void, void(*)(void*)>(
                        (void*)SUNDenseMatrix(_neq, _neq),
                        delete_sunmat));
                linsols.push_back(
                     std::unique_ptr<void, void(*)(void*)>(
                        (void*)SUNLapackDense((N_Vector)y_locals[i].get(),
                                              (SUNMatrix)mats[i].get()),
                        delete_sunmat));
                #endif

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
                #ifdef NEW_SUNDIALS
                CVDlsSetLinearSolver(integrators[i].get(), (SUNLinearSolver)linsols[i].get(),
                                     (SUNMatrix)mats[i].get());
                #else
                CVODEErrorCheck(CVLapackDense(integrators[i].get(), _neq));
                #endif

                #ifndef FINITE_DIFFERENCE
                    #ifdef NEW_SUNDIALS
                    CVODEErrorCheck(CVDlsSetJacFn(integrators[i].get(), eval_jacob_cvodes));
                    #else
                    CVODEErrorCheck(CVDlsSetDenseJacFn(integrators[i].get(), eval_jacob_cvodes));
                    #endif
                #endif

                #ifdef CV_MAX_ORD
                    CVODEErrorCheck(CVodeSetMaxOrd(integrators[i].get(), CV_MAX_ORD));
                #endif

                #ifdef CV_MAX_STEPS
                    CVODEErrorCheck(CVodeSetMaxNumSteps(integrators[i].get(), options.maxIters()));
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
