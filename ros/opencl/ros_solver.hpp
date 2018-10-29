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
        void init(ros_t *ros);

    public:

        ROS3Integrator(int neq, std::size_t numWorkGroups, const IVP& ivp, const SolverOptions& options):
            ROSIntegrator(neq, numWorkGroups, ivp, options)
        {
            // init the rk struct
            ros_create(&ros_vals, options);
            // and initialize the kernel
            this->initialize_kernel();
        }

        //! \brief Return the numerical order of the solver
        virtual size_t solverOrder() const
        {
            return 3;
        }
    };

    class ROS4Integrator : public ROSIntegrator
    {
    protected:
        //! \brief initializer for 4th-order L-stable Rosenbrock method with 4 stages.
        // -- E. Hairer and G. Wanner, "Solving ordinary differential equations II:
        //    stiff and differential-algebraic problems," Springer series in
        //    computational mathematics, Springer-Verlag (1990).
        void init(ros_t *ros);

    public:

        ROS4Integrator(int neq, std::size_t numWorkGroups, const IVP& ivp, const SolverOptions& options):
            ROSIntegrator(neq, numWorkGroups, ivp, options)
        {
            // init the rk struct
            ros_create(&ros_vals, options);
            // and initialize the kernel
            this->initialize_kernel();
        }

        //! \brief Return the numerical order of the solver
        virtual size_t solverOrder() const
        {
            return 4;
        }
    };

    class RODAS3Integrator : public ROSIntegrator
    {
    protected:
        //! \brief initializer for 3rd-order RODAS linearly-implicit solver
        void init(ros_t *ros);

    public:

        RODAS3Integrator(int neq, std::size_t numWorkGroups, const IVP& ivp, const SolverOptions& options):
            ROSIntegrator(neq, numWorkGroups, ivp, options)
        {
            // init the rk struct
            ros_create(&ros_vals, options);
            // and initialize the kernel
            this->initialize_kernel();
        }

        //! \brief Return the numerical order of the solver
        virtual size_t solverOrder() const
        {
            return 3;
        }
    };

    class RODAS4Integrator : public ROSIntegrator
    {
    protected:
        //! \brief initializer for 4th-order RODAS linearly-implicit solver
        void init(ros_t *ros);

    public:

        RODAS4Integrator(int neq, std::size_t numWorkGroups, const IVP& ivp, const SolverOptions& options):
            ROSIntegrator(neq, numWorkGroups, ivp, options)
        {
            // init the rk struct
            ros_create(&ros_vals, options);
            // and initialize the kernel
            this->initialize_kernel();
        }

        //! \brief Return the numerical order of the solver
        virtual size_t solverOrder() const
        {
            return 4;
        }
    };
}

#endif
