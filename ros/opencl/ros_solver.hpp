#ifndef ROS_HPP
#define ROS_HPP

#include "solver.hpp"
#include "ros_types.h"


namespace opencl_solvers
{

    class ROSIntegrator : public Integrator<ros_t, ros_counters_t>
    {
    private:
        std::vector<std::string> _files;
        std::vector<std::string> _includes;
        std::vector<std::string> _paths;

    protected:
        ros_t ros_vals;
        virtual void init(ros_t* ros) = 0;
        int ros_create (ros_t *ros, const SolverOptions& options)
        {

            ros->max_iters = options.maxIters();
            ros->min_iters = options.minIters();

            ros->adaption_limit = 5;

            ros->s_rtol = options.rtol();
            ros->s_atol = options.atol();

            init(ros);
            return SUCCESS;
        }

        const ros_t& getSolverStruct() const
        {
            return ros_vals;
        }

        //! \brief The requird size, in bytes of the RKF45 solver (per-IVP)
        std::size_t requiredSolverMemorySize() const
        {
            // 2 working buffers for Rosenbrock solvers + Jacobian + ktmp (nstate, neq)
            std::size_t mem = IntegratorBase::requiredSolverMemorySize();
            long our_mem = (2 * _neq + _neq*_neq + _neq * ros_vals.numStages) * sizeof(double);
            our_mem -= reusableSolverMemorySize();
            our_mem = std::max(our_mem , 0L);
            return mem + our_mem;
        }

        /**
         * \brief Return the size of integer working memory in bytes (per-IVP)
         */
        std::size_t requiredSolverIntegerMemorySize() const
        {
            return IntegratorBase::requiredSolverIntegerMemorySize() + _neq * sizeof(int);
        }

        //! \brief return the list of files for this solver
        const std::vector<std::string>& solverFiles() const
        {
            return _files;
        }

        const std::vector<std::string>& solverIncludes() const
        {
            return _includes;
        }

        //! \brief return the list of include paths for this solver
        const std::vector<std::string>& solverIncludePaths() const
        {
            return _paths;
        }

    public:
        ROSIntegrator(int neq, std::size_t numWorkGroups, const IVP& ivp, const SolverOptions& options) :
            Integrator(neq, numWorkGroups, ivp, options),
            _files({file_relative_to_me(__FILE__, "ros.cl")}),
            _includes({"ros_types.h"}),
            _paths({path_of(__FILE__)}),
            ros_vals()
        {

        }

    };

    class ROS3Integrator : public ROSIntegrator
    {
    protected:
        //! \brief initializer for 3rd-order L-stable Rosenbrock method with 4 stages.
        // -- E. Hairer and G. Wanner, "Solving ordinary differential equations II:
        //    stiff and differential-algebraic problems," Springer series in
        //    computational mathematics, Springer-Verlag (1990).
        void init(ros_t *ros)
        {
            ros->numStages = 3;
            ros->ELO = 3;

            ros->A[0] = 1.0;
            ros->A[1] = 1.0;
            ros->A[2] = 0.0;

            ros->C[0] =-1.0156171083877702091975600115545;
            ros->C[1] = 4.0759956452537699824805835358067;
            ros->C[2] = 9.2076794298330791242156818474003;

            ros->newFunc[0] = 1;
            ros->newFunc[1] = 1;
            ros->newFunc[2] = 0;

            ros->M[0] = 1.0;
            ros->M[1] = 6.169794704382824559255361568973;
            ros->M[2] =-0.42772256543218573326238373806514;

            ros->E[0] = 0.5;
            ros->E[1] =-2.9079558716805469821718236208017;
            ros->E[2] = 0.22354069897811569627360909276199;

            ros->alpha[0] = 0.0;
            ros->alpha[1] = 0.43586652150845899941601945119356;
            ros->alpha[2] = 0.43586652150845899941601945119356;

            ros->gamma[0] = 0.43586652150845899941601945119356;
            ros->gamma[1] = 0.24291996454816804366592249683314;
            ros->gamma[2] = 2.1851380027664058511513169485832;
        }

    public:

        ROS3Integrator(int neq, std::size_t numWorkGroups, const IVP& ivp, const SolverOptions& options):
            ROSIntegrator(neq, numWorkGroups, ivp, options)
        {
            // init the rk struct
            ros_create(&ros_vals, options);
            // and initialize the kernel
            this->initialize_kernel();
        }
    };

    class ROS4Integrator : public ROSIntegrator
    {
    protected:
        //! \brief initializer for 4th-order L-stable Rosenbrock method with 4 stages.
        // -- E. Hairer and G. Wanner, "Solving ordinary differential equations II:
        //    stiff and differential-algebraic problems," Springer series in
        //    computational mathematics, Springer-Verlag (1990).
        void init(ros_t *ros)
        {
            ros->numStages = 4;
            ros->ELO = 4;

            // A and C are strictly lower-triangular matrices in row-major order!!!!
            // -- A(i,j) = [(i)*(i-1)/2 + j] ... A(1,0) = A[0], A(2,0) = A[1]
            ros->A[0] = 2.0;
            ros->A[1] = 1.867943637803922;
            ros->A[2] = 0.2344449711399156;
            ros->A[3] = ros->A[1];
            ros->A[4] = ros->A[2];
            ros->A[5] = 0.0;

            ros->C[0] =-7.137615036412310;
            ros->C[1] = 2.580708087951457;
            ros->C[2] = 0.6515950076447975;
            ros->C[3] =-2.137148994382534;
            ros->C[4] =-0.3214669691237626;
            ros->C[5] =-0.6949742501781779;

            // Does the stage[i] need a new function eval or can it reuse the
            // prior one from stage[i-1]?
            ros->newFunc[0] = 1;
            ros->newFunc[1] = 1;
            ros->newFunc[2] = 1;
            ros->newFunc[3] = 0;

            // M_i = Coefficients for new step solution
            ros->M[0] = 2.255570073418735;
            ros->M[1] = 0.2870493262186792;
            ros->M[2] = 0.4353179431840180;
            ros->M[3] = 1.093502252409163;

            // E_i = Coefficients for error estimator
            ros->E[0] =-0.2815431932141155;
            ros->E[1] =-0.07276199124938920;
            ros->E[2] =-0.1082196201495311;
            ros->E[3] =-1.093502252409163;

            // Y( T + h*alpha_i )
            ros->alpha[0] = 0.0;
            ros->alpha[1] = 1.14564;
            ros->alpha[2] = 0.65521686381559;
            ros->alpha[3] = ros->alpha[2];

            // gamma_i = \Sum_j  gamma_{i,j}
            ros->gamma[0] = 0.57282;
            ros->gamma[1] =-1.769193891319233;
            ros->gamma[2] = 0.7592633437920482;
            ros->gamma[3] =-0.104902108710045;
        }

    public:

        ROS4Integrator(int neq, std::size_t numWorkGroups, const IVP& ivp, const SolverOptions& options):
            ROSIntegrator(neq, numWorkGroups, ivp, options)
        {
            // init the rk struct
            ros_create(&ros_vals, options);
            // and initialize the kernel
            this->initialize_kernel();
        }
    };

    class RODAS3Integrator : public ROSIntegrator
    {
    protected:
        //! \brief initializer for 3rd-order RODAS linearly-implicit solver
        void init(ros_t *ros)
        {
            ros->numStages = 4;
            ros->ELO = 3;

            ros->A[0] = 0.0;
            ros->A[1] = 2.0;
            ros->A[2] = 0.0;
            ros->A[3] = 2.0;
            ros->A[4] = 0.0;
            ros->A[5] = 1.0;

            ros->C[0] = 4.0;
            ros->C[1] = 1.0;
            ros->C[2] =-1.0;
            ros->C[3] = 1.0;
            ros->C[4] =-1.0;
            ros->C[5] =-(8.0/3.0);

            ros->newFunc[0] = 1;
            ros->newFunc[1] = 0;
            ros->newFunc[2] = 1;
            ros->newFunc[3] = 1;

            // M_i = Coefficients for new step solution
            ros->M[0] = 2.0;
            ros->M[1] = 0.0;
            ros->M[2] = 1.0;
            ros->M[3] = 1.0;

            ros->E[0] = 0.0;
            ros->E[1] = 0.0;
            ros->E[2] = 0.0;
            ros->E[3] = 1.0;

            ros->alpha[0] = 0.0;
            ros->alpha[1] = 0.0;
            ros->alpha[2] = 1.0;
            ros->alpha[3] = 1.0;

            ros->gamma[0] = 0.5;
            ros->gamma[1] = 1.5;
            ros->gamma[2] = 0.;
            ros->gamma[3] = 0.;
        }

    public:

        RODAS3Integrator(int neq, std::size_t numWorkGroups, const IVP& ivp, const SolverOptions& options):
            ROSIntegrator(neq, numWorkGroups, ivp, options)
        {
            // init the rk struct
            ros_create(&ros_vals, options);
            // and initialize the kernel
            this->initialize_kernel();
        }
    };

    class RODAS4Integrator : public ROSIntegrator
    {
    protected:
        //! \brief initializer for 4th-order RODAS linearly-implicit solver
        void init(ros_t *ros)
        {
            ros->numStages = 6;
            ros->ELO = 4;

            ros->A[ 0] = 1.544;
            ros->A[ 1] = 0.9466785280815826;
            ros->A[ 2] = 0.2557011698983284;
            ros->A[ 3] = 3.314825187068521;
            ros->A[ 4] = 2.896124015972201;
            ros->A[ 5] = 0.9986419139977817;
            ros->A[ 6] = 1.221224509226641;
            ros->A[ 7] = 6.019134481288629;
            ros->A[ 8] = 12.53708332932087;
            ros->A[ 9] =-0.687886036105895;
            ros->A[10] = ros->A[6];
            ros->A[11] = ros->A[7];
            ros->A[12] = ros->A[8];
            ros->A[13] = ros->A[9];
            ros->A[14] = 1.0;

            ros->C[ 0] =-5.6688;
            ros->C[ 1] =-2.430093356833875;
            ros->C[ 2] =-0.2063599157091915;
            ros->C[ 3] =-0.1073529058151375;
            ros->C[ 4] =-0.9594562251023355e+01;
            ros->C[ 5] =-0.2047028614809616e+02;
            ros->C[ 6] = 0.7496443313967647e+01;
            ros->C[ 7] =-0.1024680431464352e+02;
            ros->C[ 8] =-0.3399990352819905e+02;
            ros->C[ 9] = 0.1170890893206160e+02;
            ros->C[10] = 0.8083246795921522e+01;
            ros->C[11] =-0.7981132988064893e+01;
            ros->C[12] =-0.3152159432874371e+02;
            ros->C[13] = 0.1631930543123136e+02;
            ros->C[14] =-0.6058818238834054e+01;

            ros->newFunc[0] = 1;
            ros->newFunc[1] = 1;
            ros->newFunc[2] = 1;
            ros->newFunc[3] = 1;
            ros->newFunc[4] = 1;
            ros->newFunc[5] = 1;

            // M_i = Coefficients for new step solution
            ros->M[0] = ros->A[6];
            ros->M[1] = ros->A[7];
            ros->M[2] = ros->A[8];
            ros->M[3] = ros->A[9];
            ros->M[4] = 1.0;
            ros->M[5] = 1.0;

            ros->E[0] = 0.0;
            ros->E[1] = 0.0;
            ros->E[2] = 0.0;
            ros->E[3] = 0.0;
            ros->E[4] = 0.0;
            ros->E[5] = 1.0;

            ros->alpha[0] = 0.0;
            ros->alpha[1] = 0.386;
            ros->alpha[2] = 0.210;
            ros->alpha[3] = 0.630;
            ros->alpha[4] = 1.0;
            ros->alpha[5] = 1.0;

            ros->gamma[0] = 0.25;
            ros->gamma[1] =-0.1043;
            ros->gamma[2] = 0.1035;
            ros->gamma[3] =-0.3620000000000023E-01;
            ros->gamma[4] = 0.0;
            ros->gamma[5] = 0.0;
        }

    public:

        RODAS4Integrator(int neq, std::size_t numWorkGroups, const IVP& ivp, const SolverOptions& options):
            ROSIntegrator(neq, numWorkGroups, ivp, options)
        {
            // init the rk struct
            ros_create(&ros_vals, options);
            // and initialize the kernel
            this->initialize_kernel();
        }
    };
}

#endif
