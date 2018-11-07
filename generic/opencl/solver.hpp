/**
 * \file
 * \brief Contains skeleton of all methods that need to be defined on a per solver basis.
 *
 * \author Nicholas Curtis
 * \date 03/10/2015
 *
 *
 */

#ifndef SOLVER_H
#define SOLVER_H

#include <iostream>
#include <fstream>
#include <cstdio>
#include <cassert>
#include <cstring>
#include <sstream>
#include <chrono>
#include <vector>
#include <memory>
#include <algorithm>
#include <stdexcept>
#include <cmath>
#include "error_codes.hpp"
#include "stepper_types.hpp"
#include "../paths/path.h"
#include "../paths/resolver.h"
extern "C" {
#include "CL/cl.h"
}

#define xstringify(s) (stringify(s))
#define stringify(s) (#s)

// #include <ros.h>

const std::string getErrorString(cl_int error)
{
    switch(error){
        // run-time and JIT compiler errors
        case 0: return "CL_SUCCESS";
        case -1: return "CL_DEVICE_NOT_FOUND";
        case -2: return "CL_DEVICE_NOT_AVAILABLE";
        case -3: return "CL_COMPILER_NOT_AVAILABLE";
        case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case -5: return "CL_OUT_OF_RESOURCES";
        case -6: return "CL_OUT_OF_HOST_MEMORY";
        case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
        case -8: return "CL_MEM_COPY_OVERLAP";
        case -9: return "CL_IMAGE_FORMAT_MISMATCH";
        case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        case -11: return "CL_BUILD_PROGRAM_FAILURE";
        case -12: return "CL_MAP_FAILURE";
        case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
        case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
        case -15: return "CL_COMPILE_PROGRAM_FAILURE";
        case -16: return "CL_LINKER_NOT_AVAILABLE";
        case -17: return "CL_LINK_PROGRAM_FAILURE";
        case -18: return "CL_DEVICE_PARTITION_FAILED";
        case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

        // compile-time errors
        case -30: return "CL_INVALID_VALUE";
        case -31: return "CL_INVALID_DEVICE_TYPE";
        case -32: return "CL_INVALID_PLATFORM";
        case -33: return "CL_INVALID_DEVICE";
        case -34: return "CL_INVALID_CONTEXT";
        case -35: return "CL_INVALID_QUEUE_PROPERTIES";
        case -36: return "CL_INVALID_COMMAND_QUEUE";
        case -37: return "CL_INVALID_HOST_PTR";
        case -38: return "CL_INVALID_MEM_OBJECT";
        case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
        case -40: return "CL_INVALID_IMAGE_SIZE";
        case -41: return "CL_INVALID_SAMPLER";
        case -42: return "CL_INVALID_BINARY";
        case -43: return "CL_INVALID_BUILD_OPTIONS";
        case -44: return "CL_INVALID_PROGRAM";
        case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
        case -46: return "CL_INVALID_KERNEL_NAME";
        case -47: return "CL_INVALID_KERNEL_DEFINITION";
        case -48: return "CL_INVALID_KERNEL";
        case -49: return "CL_INVALID_ARG_INDEX";
        case -50: return "CL_INVALID_ARG_VALUE";
        case -51: return "CL_INVALID_ARG_SIZE";
        case -52: return "CL_INVALID_KERNEL_ARGS";
        case -53: return "CL_INVALID_WORK_DIMENSION";
        case -54: return "CL_INVALID_WORK_GROUP_SIZE";
        case -55: return "CL_INVALID_WORK_ITEM_SIZE";
        case -56: return "CL_INVALID_GLOBAL_OFFSET";
        case -57: return "CL_INVALID_EVENT_WAIT_LIST";
        case -58: return "CL_INVALID_EVENT";
        case -59: return "CL_INVALID_OPERATION";
        case -60: return "CL_INVALID_GL_OBJECT";
        case -61: return "CL_INVALID_BUFFER_SIZE";
        case -62: return "CL_INVALID_MIP_LEVEL";
        case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
        case -64: return "CL_INVALID_PROPERTY";
        case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
        case -66: return "CL_INVALID_COMPILER_OPTIONS";
        case -67: return "CL_INVALID_LINKER_OPTIONS";
        case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

        // extension errors
        case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
        case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
        case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
        case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
        case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
        case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
        default: return "Unknown OpenCL error";
    }
}

#define __clerror(__errcode, CMD) \
{ \
    std::cerr << "Error executing CL cmd = " << CMD << std::endl; \
    std::cerr << "\terrcode = " << (__errcode); \
    std::cerr << ", " << getErrorString(__errcode) << std::endl; \
    throw std::runtime_error(getErrorString(__errcode)); \
}

#define CL_EXEC(__cmd__) {\
    cl_uint _ret = (__cmd__); \
    if (_ret != CL_SUCCESS) \
    { \
        __clerror(_ret, __cmd__); \
    } \
}

namespace opencl_solvers {
    class OpenCLException : public std::runtime_error {
    public:
        explicit OpenCLException(std::string message) : std::runtime_error(message)
        {

        }
    };

    //! \brief A wrapper that contains information for the OpenCL kernel
    typedef struct
    {
        char function_name[1024];
        char attributes[1024];
        cl_uint num_args;
        cl_uint reference_count;
        size_t compile_work_group_size[3];
        size_t work_group_size;
        size_t preferred_work_group_size_multiple;
        cl_ulong local_mem_size;
        size_t global_work_size[3];
        cl_ulong private_mem_size;
    } kernelInfo_t;

    //! \brief the maximum number of platforms to check
    static constexpr std::size_t cl_max_platforms = 16;

    //! \brief A wrapper that contains information for the OpenCL platform
    typedef struct
    {
        cl_uint num_platforms;
        cl_platform_id platform_ids[cl_max_platforms]; // All the platforms ...
        cl_platform_id platform_id; // The one we'll use ...

        char name[1024];
        char version[1024];
        char vendor[1024];
        char extensions[2048];

        int is_nvidia;
    } platformInfo_t;

    //! \brief the maximum number of devices to query
    static const std::size_t cl_max_devices = 16;
    //! \brief A wrapper the contains information about the OpenCL device
    typedef struct
    {
        cl_device_id device_ids[cl_max_devices];
        cl_device_id device_id;
        cl_uint num_devices;

        cl_device_type type;

        char name[1024];
        char profile[1024];
        char version[1024];
        char vendor[1024];
        char driver_version[1024];
        char opencl_c_version[1024];
        char extensions[1024];

        cl_uint native_vector_width_char;
        cl_uint native_vector_width_short;
        cl_uint native_vector_width_int;
        cl_uint native_vector_width_long;
        cl_uint native_vector_width_float;
        cl_uint native_vector_width_double;
        cl_uint native_vector_width_half;

        cl_uint preferred_vector_width_char;
        cl_uint preferred_vector_width_short;
        cl_uint preferred_vector_width_int;
        cl_uint preferred_vector_width_long;
        cl_uint preferred_vector_width_float;
        cl_uint preferred_vector_width_double;
        cl_uint preferred_vector_width_half;

        cl_uint max_compute_units;
        cl_uint max_clock_frequency;

        cl_ulong max_constant_buffer_size;
        cl_uint  max_constant_args;

        cl_ulong max_work_group_size;

        cl_ulong max_mem_alloc_size;
        cl_ulong global_mem_size;
        cl_uint  global_mem_cacheline_size;
        cl_ulong global_mem_cache_size;

        cl_device_mem_cache_type global_mem_cache_type;

        cl_ulong local_mem_size;
        cl_device_local_mem_type local_mem_type;

        cl_device_fp_config fp_config;
    } deviceInfo_t;


    //! Wrapper for CL_DEVICE_TYPE
    enum DeviceType : cl_uint
    {
        //! CPU device
        CPU = CL_DEVICE_TYPE_CPU,
        //! GPU device
        GPU = CL_DEVICE_TYPE_GPU,
        //! Accelerator
        ACCELERATOR = CL_DEVICE_TYPE_ACCELERATOR,
        //! Default
        DEFAULT = CL_DEVICE_TYPE_DEFAULT
    };

    //! Wrapper for OpenCL kernel creation
    typedef struct
    {
        platformInfo_t platform_info;
        deviceInfo_t device_info;

        cl_context context;
        cl_command_queue command_queue;
        cl_program program;

        size_t blockSize;
        size_t numBlocks;
        size_t vectorSize;

        int use_queue;
    } cl_data_t;

    //! \brief Implementation of a initial value problem
    class IVP
    {
    public:
        IVP(const std::vector<std::string>& kernelSource,
            std::size_t requiredMemorySize,
            std::size_t requiredIntegerMemorySize=0,
            const std::vector<std::string>& includePaths=std::vector<std::string>()):
                _kernelSource(kernelSource),
                _requiredMemorySize(requiredMemorySize),
                _requiredIntegerMemorySize(requiredIntegerMemorySize),
                _includePaths(includePaths)
        {

        }

        //! \brief Return the list of filenames of OpenCL kernels that implement
        //!        the source term and Jacobian
        const std::vector<std::string>& kernelSource() const
        {
            return _kernelSource;
        }

        //! \brief Return the required amount of (unvectorized) double-precision memory
        //         required by the IVP kernels [in bytes]
        std::size_t requiredMemorySize() const
        {
            return _requiredMemorySize;
        }

        //! \brief Return the required amount of (unvectorized) integer memory
        //         required by the IVP kernels [in bytes]
        std::size_t requiredIntegerMemorySize() const
        {
            return _requiredIntegerMemorySize;
        }

        const std::vector<std::string>& includePaths() const
        {
            return _includePaths;
        }

    protected:
        std::vector<std::string> _kernelSource;
        std::size_t _requiredMemorySize;
        std::size_t _requiredIntegerMemorySize;
        std::vector<std::string> _includePaths;

    };

    class SolverOptions
    {
    public:
        SolverOptions(std::size_t vectorSize=1, std::size_t blockSize=1,
                      double atol=1e-10, double rtol=1e-6,
                      bool logging=false, bool use_queue=true, std::string order="C",
                      std::string platform = "", DeviceType deviceType=DeviceType::DEFAULT,
                      size_t minIters = 1, size_t maxIters = 1000,
                      StepperType stepperType = StepperType::ADAPTIVE,
                      double h_const=std::numeric_limits<double>::quiet_NaN(),
                      bool estimate_chemistry_time=false,
                      bool verbose=true):
            _vectorSize(vectorSize),
            _blockSize(blockSize),
            _atol(atol),
            _rtol(rtol),
            _logging(logging),
            _order(order),
            _use_queue(use_queue),
            _platform(platform),
            _deviceType(deviceType),
            _minIters(minIters),
            _maxIters(maxIters),
            _stepperType(stepperType),
            _h_const(h_const),
            _estimate_chemistry_time(estimate_chemistry_time),
            _verbose(verbose)
        {
            if (order.compare("C") && order.compare("F"))
            {
                std::ostringstream err;
                err << "Order " << order << " not recognized";
                throw OpenCLException(err.str());
            }

            // ensure our internal error code match the enum-types
            #include "error_codes.h"
            static_assert(ErrorCode::SUCCESS == OCL_SUCCESS, "Enum mismatch");
            static_assert(ErrorCode::TOO_MUCH_WORK == OCL_TOO_MUCH_WORK, "Enum mismatch");
            static_assert(ErrorCode::TDIST_TOO_SMALL == OCL_TDIST_TOO_SMALL, "Enum mismatch");
            //static_assert(ErrorCode::MAX_STEPS_EXCEEDED == RK_HIN_MAX_ITERS, "Enum mismatch");
        }

        inline double atol() const
        {
            return _atol;
        }

        inline double rtol() const
        {
            return _rtol;
        }

        inline bool logging() const
        {
            return _logging;
        }

        inline std::size_t vectorSize() const
        {
            return _vectorSize;
        }

        inline std::size_t blockSize() const
        {
            return _blockSize;
        }

        inline const std::string& order() const
        {
            return _order;
        }

        inline bool useQueue() const
        {
            return _use_queue;
        }

        inline const std::string& platform() const
        {
            return _platform;
        }

        inline const DeviceType& deviceType() const
        {
            return _deviceType;
        }

        inline std::size_t minIters() const
        {
            return _minIters;
        }

        inline std::size_t maxIters() const
        {
            return _maxIters;
        }

        inline StepperType stepperType() const
        {
            return _stepperType;
        }

        inline double constantTimestep() const
        {
            return _h_const;
        }

        inline bool estimateChemistryTime() const
        {
            return _estimate_chemistry_time;
        }

        inline bool verbose() const
        {
            return _verbose;
        }

    protected:
        //! vector size
        std::size_t _vectorSize;
        //! block size
        std::size_t _blockSize;
        //! the absolute tolerance for this integrator
        const double _atol;
        //! the relative tolerance for this integrator
        const double _rtol;
        //! whether logging is enabled or not
        bool _logging;
        //! The data-ordering
        const std::string _order;
        //! Use queue driver?
        const bool _use_queue;
        //! OpenCL platform to use
        const std::string _platform;
        //! The OpenCL device type to use
        const DeviceType _deviceType;
        //! The minimum number of iterations allowed
        const std::size_t _minIters;
        //! The maximum number of iterations allowed
        const std::size_t _maxIters;
        //! The type of time-stepping to utilize constant, or adaptive
        StepperType _stepperType;
        //! The constant integration step to take (if #_stepperType == StepperTypes::CONSTANT)
        double _h_const;
        //! \brief If true, the calling code wants the last time-step taken by the integratior
        //!        as an estimation of the chemistry time-scale
        bool _estimate_chemistry_time;
        //! \brief If true, the solver will output information on the decvice used, kernel, etc.
        bool _verbose;
    };


    class IntegratorBase
    {
    public:
        IntegratorBase(int neq, std::size_t numWorkGroups,
                       const IVP& ivp,
                       const SolverOptions& options):
            _numWorkGroups(numWorkGroups),
            _neq(neq),
            _log(),
            _ivp(ivp),
            _options(options),
            _data(),
            _clmem(),
            _kernel_info(),
            _kernel(),
            _storedNumProblems(0),
            _start_time_index(-1),
            _end_time_index(-1),
            _param_index(-1),
            _phi_index(-1),
            _solver_index(-1),
            _counter_index(-1),
            _queue_index(-1),
            _chem_time_index(-1),
            _initialized(false)
        {

        }

        virtual ~IntegratorBase()
        {
            this->clean();
        }

        void log(const int NUM, const double t, double const * __restrict__ phi)
        {
            // allocate new memory
            _log.emplace_back(std::move(std::unique_ptr<double[]>(new double[1 + NUM * _neq])));
            double* __restrict__ set = _log.back().get();
            // and set
            set[0] = t;
            std::memcpy(&set[1], phi, NUM * _neq * sizeof(double));
        }

        /*! \brief Copy the current log of data to the given array */
        void getLog(const int NUM, double* __restrict__ times, double* __restrict__ phi) const
        {
            for (std::size_t index = 0; index < _log.size(); ++index)
            {
                const double* __restrict__ log_entry = _log[index].get();
                times[index] = log_entry[0];
                std::memcpy(&phi[index * NUM * _neq], &log_entry[1], NUM * _neq * sizeof(double));
            }
        }

        /*! \brief Return the number of integration steps completed by this Integrator */
        std::size_t numSteps() const
        {
            return _log.size();
        }

        /*! \brief Resize the Integrator to utilize the specified number of work groups and solve #numProblems */
        virtual void reinitialize(std::size_t numWorkGroups, std::size_t numProblems) = 0;

        /*! checkError
                \brief Checks the return code of the given thread (IVP) for an error, and exits if found
                \param tid The thread (IVP) index
                \param code The return code of the thread
                @see ErrorCodes
        */
        void checkError(int tid, ErrorCode code) const
        {
            switch(code)
            {
                case ErrorCode::TOO_MUCH_WORK:
                    std::cerr << "During integration of ODE#" << tid <<
                        ", the maximum number of allowed iterations was exceeded..."
                        << std::endl;
                    throw std::runtime_error(code);
                case ErrorCode::TDIST_TOO_SMALL:
                    std::cerr << "During integration of ODE#" << tid <<
                        ", the requested integration duration was smaller than allowed "
                        "by machine precision, exiting..." << std::endl;
                    throw std::runtime_error(code);
                default:
                    break;
            }
        }

        //! return the absolute tolerance
        inline double atol() const
        {
            return _options.atol();
        }

        //! return the relative tolerance
        inline double rtol() const
        {
            return _options.rtol();
        }

        inline bool logging() const
        {
            return _options.logging();
        }

        inline const std::string& order() const
        {
            return _options.order();
        }

        inline bool verbose() const
        {
            return _options.verbose();
        }

        //! \brief return the number of equations to solve
        inline int neq() const
        {
            return _neq;
        }

        //! \brief Return the number of OpenCL work-groups to launch
        //         On the CPU / Accelerators this corresponds to the # of threads
        //         On the GPU, this corresponds to the # of "thread-blocks", in CUDA terminology
        inline int numWorkGroups() const
        {
            return _numWorkGroups;
        }

        //! \brief Return the numerical order of the solver
        virtual size_t solverOrder() const = 0;

        /**
         * \brief Integration driver for the CPU integrators
         * \param[in]       NUM             The (non-padded) number of IVPs to integrate
         * \param[in]       t               The current system time
         * \param[in]       t_end           The IVP integration end time
         * \param[in]       param           The system constant variable (pressures / densities)
         * \param[in,out]   phi             The system state vectors at time t.
         * \param[out]      last_stepsize   If supplied, store last step-size taken by the integrator for each IVP. Useful for OpenFOAM / chemistry timescale integration
         * \returns system state vectors at time t_end
         *
         */
        void intDriver (const int NUM, const double t,
                        const double t_end, const double* __restrict__ param,
                        double* __restrict__ phi,
                        double* __restrict__ last_stepsize=NULL)
        {
            std::vector<double> t_end_vec (t_end, NUM);
            this->intDriver(NUM, t, &t_end_vec[0], param, phi, last_stepsize);
        }

        /**
         * \brief Integration driver for the CPU integrators
         * \param[in]       NUM             The (non-padded) number of IVPs to integrate
         * \param[in]       t               The (array) of current system times
         * \param[in]       t_end           The (array) of IVP integration end times
         * \param[in]       param           The system constant variable (pressures / densities)
         * \param[in,out]   phi             The system state vectors at time t.
         * \param[out]      last_stepsize   If supplied, store last step-size taken by the integrator for each IVP. Useful for OpenFOAM / chemistry timescale integration
         * \returns system state vectors at time t_end
         *
         */
        virtual void intDriver (const int NUM, const double t,
                                const double* __restrict__ t_end,
                                const double* __restrict__ param, double* __restrict__ phi,
                                double* __restrict__ last_stepsize=NULL) = 0;



    protected:
        //! the number of OpenCL work-groups to launch
        int _numWorkGroups;
        //! the number of equations to solver per-IVP
        const int _neq;
        //! working memory for this integrator
        std::unique_ptr<char[]> working_buffer;
        //! log of state vectors / times
        std::vector<std::unique_ptr<double[]>> _log;
        //! \brief IVP information
        const IVP& _ivp;
        //! \brief Solver options for OpenCL execution
        const SolverOptions& _options;
        //! \brief struct holding opencl context, program, etc.
        cl_data_t _data;
        //! \brief CL memory
        std::vector<cl_mem> _clmem;
        //! \brief CL Kernel holder
        kernelInfo_t _kernel_info;
        //! \brief CL Kernel
        cl_kernel _kernel;
        //! \brief # of problems previous solved, used to detect if we must resize the buffers
        std::size_t _storedNumProblems;
        //! \brief The kernel argument index of the start time
        int _start_time_index;
        //! \brief The index of the end time buffer in #_clmem
        int _end_time_index;
        //! \brief The index of the parameter buffer in #_clmem
        int _param_index;
        //! \brief The index of the state vector buffer in #_clmem
        int _phi_index;
        //! \brief The index of the solution struct buffer in #_clmem
        int _solver_index;
        //! \brief The index of the counter struct buffer in #_clmem
        int _counter_index;
        //! \brief The index of the queue buffer in #_clmem
        int _queue_index;
        //! \brief The chemistry time index
        int _chem_time_index;
        //! \brief simple flag to mark whether the device / context / kernel have been created
        bool _initialized;


        //! \brief an initialization function for the kernel, to be called from
        //!        the final derived integrator class's constructor
        void initialize_kernel();

        const std::string path_of(const std::string& owner) const;

        const std::string file_relative_to_me(const std::string& owner, const std::string& filename) const
        {
            filesystem::path source_dir(path_of(owner));
            // get parent and return
            return (source_dir/filesystem::path(filename)).str();
        }


        void load_source_from_file (const std::string& flname, std::ostringstream& source_str)
        {
            std::ifstream ifile(flname);
            std::string content ((std::istreambuf_iterator<char>(ifile)),
                                 (std::istreambuf_iterator<char>()));
            if (!content.length())
            {
                std::ostringstream err;
                err << "Kernel file " << flname << " not found!" << std::endl;
                throw OpenCLException(err.str());
            };
            source_str << content;
        }

        bool isPower2 (int x)
        {
            return ( (x > 0) && ((x & (x-1)) == 0) );
        }

        int imax(int a, int b) { return (a > b) ? a : b; }
        int imin(int a, int b) { return (a < b) ? a : b; }


        #define __Alignment (128)

        cl_mem CreateBuffer (cl_context *context, cl_mem_flags flags, std::size_t size, void *host_ptr);

        cl_device_type getDeviceType (cl_device_id device_id)
        {
            cl_device_type val;
            CL_EXEC( clGetDeviceInfo (device_id, CL_DEVICE_TYPE, sizeof(cl_device_type), &val, NULL) );

            return val;
        }

        void printKernelInfo (const kernelInfo_t* info)
        {
            if (verbose())
            {
                std::cout << "Kernel Info:" << std::endl;
                std::cout << "\t" << "function_name = " << info->function_name << std::endl;
                #if (__OPENCL_VERSION__ >= 120)
                std::cout << "\t" << "attributes = " << info->attributes << std::endl;
                #endif
                std::cout << "\t" << "num_args = " << info->num_args << std::endl;
                std::cout << "\t" << "reference_count = " << info->reference_count << std::endl;
                std::cout << "\t" << "compile_work_group_size = (" << info->compile_work_group_size[0] <<
                          "," << info->compile_work_group_size[1] << "," << info->compile_work_group_size[2] <<
                          ")" << std::endl;
                std::cout << "\t" << "work_group_size = " << info->work_group_size << std::endl;
                std::cout << "\n" << "preferred_work_group_size_multiple = " <<
                                        info->preferred_work_group_size_multiple << std::endl;
                std::cout << "local_mem_size =" << info->local_mem_size << std::endl;
                std::cout << "private_mem_size = " << info->private_mem_size << std::endl;
            }
        }

        void getKernelInfo (kernelInfo_t *info, cl_kernel kernel, cl_device_id device_id)
        {
             CL_EXEC( clGetKernelInfo (kernel, CL_KERNEL_FUNCTION_NAME, sizeof(info->function_name), info->function_name, NULL) );
             #if (__OPENCL_VERSION__ >= 120)
             CL_EXEC( clGetKernelInfo (kernel, CL_KERNEL_ATTRIBUTES, sizeof(info->attributes), info->attributes, NULL) );
             #endif
             CL_EXEC( clGetKernelInfo (kernel, CL_KERNEL_NUM_ARGS, sizeof(info->num_args), &info->num_args, NULL) );
             CL_EXEC( clGetKernelInfo (kernel, CL_KERNEL_REFERENCE_COUNT, sizeof(info->reference_count), &info->reference_count, NULL) );
             CL_EXEC( clGetKernelWorkGroupInfo (kernel, device_id, CL_KERNEL_COMPILE_WORK_GROUP_SIZE, sizeof(info->compile_work_group_size), info->compile_work_group_size, NULL) );
             CL_EXEC( clGetKernelWorkGroupInfo (kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(info->work_group_size), &info->work_group_size, NULL) );
             CL_EXEC( clGetKernelWorkGroupInfo (kernel, device_id, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(info->preferred_work_group_size_multiple), &info->preferred_work_group_size_multiple, NULL) );
             //CL_EXEC( clGetKernelWorkGroupInfo (kernel, device_id, CL_KERNEL_GLOBAL_WORK_SIZE, sizeof(info->global_work_size), info->global_work_size, NULL) );
             CL_EXEC( clGetKernelWorkGroupInfo (kernel, device_id, CL_KERNEL_LOCAL_MEM_SIZE, sizeof(info->local_mem_size), &info->local_mem_size, NULL) );
             CL_EXEC( clGetKernelWorkGroupInfo (kernel, device_id, CL_KERNEL_PRIVATE_MEM_SIZE, sizeof(info->private_mem_size), &info->private_mem_size, NULL) );
             printKernelInfo (info);
        }

        void getPlatformInfo (platformInfo_t *info);

        void getDeviceInfo (const cl_uint& device_type,
                            deviceInfo_t *device_info,
                            const platformInfo_t *platform_info);

        int cl_init (cl_data_t* data);

        //! \brief Return the total number of OpenCL "threads" to launch, i.e., the global size
        std::size_t numThreads() const
        {
            return _data.blockSize * _data.numBlocks;
        }

        //! \brief Set the starting time on our kernel
        void set_start_time(const double t_start)
        {
            if (_start_time_index < 0)
            {
                throw std::runtime_error("Not implemented!");
            }
            CL_EXEC( clSetKernelArg(_kernel, _start_time_index, sizeof(double), &t_start) );
        }

        //! \brief free the OpenCL memory
        void release_memory()
        {
            for (cl_mem buffer : _clmem)
            {
                clReleaseMemObject (buffer);
            }
        }

        //! \brief return the list of files for this solver
        virtual const std::vector<std::string>& solverFiles() const = 0;

         //! \brief return the list of to include for this solver
        virtual const std::vector<std::string>& solverIncludes() const = 0;

        //! \brief return the list of include paths for this solver
        virtual const std::vector<std::string>& solverIncludePaths() const = 0;

        /**
         * \brief Return the size of double precision working memory in bytes (per-IVP)
         */
        virtual std::size_t requiredSolverMemorySize() const
        {
            // 1 for the parameter, 1 for the step-size, and four working vectors
            //      1. local copy of state vector
            //      2. ydot  for get_hin
            //      3. y1    for get_hin
            //      4. ydot1 for get_hin
            return (2 + 4 * _neq) * sizeof(double);
        }

        /**
         * \brief Return the amount of the requiredSolverMemory that may be safely
         *        reused by the solver
         */
        std::size_t reusableSolverMemorySize() const
        {
            // the three working vectors for get_hin
            return (3 * _neq) * sizeof(double);
        }

        /**
         * \brief Return the size of integer working memory in bytes (per-IVP)
         */
        virtual std::size_t requiredSolverIntegerMemorySize() const
        {
            return 0;
        }



    private:
        void clean()
        {
            _log.clear();
        }

    };


    // skeleton of the opencl-solver
    template <typename solver_struct, typename counter_struct>
    class Integrator : public IntegratorBase
    {
    public:
        Integrator(int neq, int numThreads,
                   const IVP& ivp,
                   const SolverOptions& options) :
            IntegratorBase(neq, numThreads, ivp, options)
        {

        }

        ~Integrator()
        {
            // free memory
            this->release_memory();
            try
            {
                // release kernel
                CL_EXEC(clReleaseKernel(_kernel));
            }
            catch (std::runtime_error)
            { }
            try
            {
                // release program
                CL_EXEC(clReleaseProgram(_data.program));
            }
            catch (std::runtime_error)
            { }
            try
            {
                // release queue
                CL_EXEC(clReleaseCommandQueue(_data.command_queue));
            }
            catch (std::runtime_error)
            { }
            try
            {
                // and context
                CL_EXEC(clReleaseContext(_data.context));
            }
            catch (std::runtime_error)
            { }
        }

    protected:

        std::size_t indexof(const std::vector<cl_mem>::const_iterator begin, const std::vector<cl_mem>::const_iterator end,
                            const cl_mem& buffer)
        {
            auto it = std::find(begin, end, buffer);
            if (it == end)
            {
                throw std::runtime_error("buffer not found!");
            }
            return std::distance(begin, it);
        }

        void resize(const int NUM)
        {
            // create memory
            std::size_t lenrwk = (_ivp.requiredMemorySize() + requiredSolverMemorySize())*_data.vectorSize;
            if (verbose())
                std::cout << "lenrwk = "<< lenrwk << std::endl;

            std::size_t leniwk = (_ivp.requiredIntegerMemorySize() + requiredSolverIntegerMemorySize()) * _data.vectorSize;
            if (verbose())
                std::cout << "leniwk = "<< leniwk << std::endl;

            if (verbose())
                std::cout << "NP = " << NUM << ", blockSize = " << _data.blockSize <<
                             ", vectorSize = " << _data.vectorSize <<
                             ", numBlocks = " << _data.numBlocks <<
                             ", numThreads = " << numThreads() << std::endl;


            cl_mem buffer_param = CreateBuffer (&_data.context, CL_MEM_READ_ONLY, sizeof(double)*NUM, NULL);
            cl_mem tf = CreateBuffer (&_data.context, CL_MEM_READ_ONLY, sizeof(double)*NUM, NULL);
            cl_mem buffer_phi = CreateBuffer (&_data.context, CL_MEM_READ_WRITE, sizeof(double)*_neq*NUM, NULL);
            cl_mem buffer_solver = CreateBuffer (&_data.context, CL_MEM_READ_ONLY, sizeof(solver_struct), NULL);
            cl_mem buffer_rwk = CreateBuffer (&_data.context, CL_MEM_READ_WRITE, lenrwk*numThreads(), NULL);
            cl_mem buffer_iwk = CreateBuffer (&_data.context, CL_MEM_READ_WRITE, leniwk*numThreads(), NULL);
            cl_mem buffer_counters = CreateBuffer (&_data.context, CL_MEM_READ_WRITE, sizeof(counter_struct)*NUM, NULL);
            _clmem.assign({buffer_param, tf, buffer_phi, buffer_solver, buffer_rwk, buffer_iwk, buffer_counters});

            _param_index = indexof(_clmem.begin(), _clmem.end(), buffer_param);
            _end_time_index = indexof(_clmem.begin(), _clmem.end(), tf);
            _phi_index = indexof(_clmem.begin(), _clmem.end(), buffer_phi);
            _solver_index = indexof(_clmem.begin(), _clmem.end(), buffer_solver);
            _counter_index = indexof(_clmem.begin(), _clmem.end(), buffer_counters);

            cl_mem buffer_queue;
            if (_options.useQueue())
            {
                buffer_queue = CreateBuffer (&_data.context, CL_MEM_READ_WRITE, sizeof(int), NULL);
                _clmem.push_back(buffer_queue);
                _queue_index = indexof(_clmem.begin(), _clmem.end(), buffer_queue);
            }

            cl_mem chem_time;
            if (_options.estimateChemistryTime())
            {
                chem_time = CreateBuffer (&_data.context, CL_MEM_READ_WRITE, sizeof(double) * NUM, NULL);
                _clmem.push_back(chem_time);
                _chem_time_index = indexof(_clmem.begin(), _clmem.end(), chem_time);
            }

            /* Set kernel arguments */
            int argc = 0;
            // dummy time
            double t = 0;
            CL_EXEC( clSetKernelArg(_kernel, argc++, sizeof(cl_mem), &buffer_param) );
            _start_time_index = argc;
            CL_EXEC( clSetKernelArg(_kernel, argc++, sizeof(double), &t) );
            CL_EXEC( clSetKernelArg(_kernel, argc++, sizeof(cl_mem), &tf) );
            CL_EXEC( clSetKernelArg(_kernel, argc++, sizeof(cl_mem), &buffer_phi) );
            CL_EXEC( clSetKernelArg(_kernel, argc++, sizeof(cl_mem), &buffer_solver) );
            CL_EXEC( clSetKernelArg(_kernel, argc++, sizeof(cl_mem), &buffer_rwk) );
            CL_EXEC( clSetKernelArg(_kernel, argc++, sizeof(cl_mem), &buffer_iwk) );
            CL_EXEC( clSetKernelArg(_kernel, argc++, sizeof(cl_mem), &buffer_counters) );
            CL_EXEC( clSetKernelArg(_kernel, argc++, sizeof(int), &NUM) );
            if (_options.useQueue())
            {
                CL_EXEC( clSetKernelArg(_kernel, argc++, sizeof(cl_mem), &buffer_queue) );
            }
            if (_options.estimateChemistryTime())
            {
                CL_EXEC( clSetKernelArg(_kernel, argc++, sizeof(cl_mem), &chem_time) );
            }

            _storedNumProblems = NUM;

        }

    public:

        /*! \brief Resize the Integrator to utilize the specified number of work groups and solve #numProblems */
        void reinitialize(std::size_t numWorkGroups, std::size_t numProblems)
        {
            if (numProblems < 1)
            {
                throw std::runtime_error("Number of problems to solve must be at least one!");
            }
            if (numWorkGroups != _numWorkGroups)
            {
                // set internal
                _numWorkGroups = numWorkGroups;
                // set data
                _data.numBlocks = numWorkGroups;
            }
            if (numProblems != _storedNumProblems)
            {
                // have to reset the memory arrays
                this->release_memory();
                this->resize(numProblems);
            }
        }

        /**
         * \brief Integration driver for the CPU integrators
         * \param[in]       NUM             The (non-padded) number of IVPs to integrate
         * \param[in]       t               The (array) of current system times
         * \param[in]       t_end           The (array) of IVP integration end times
         * \param[in]       param           The system constant variable (pressures / densities)
         * \param[in,out]   phi             The system state vectors at time t.
         * \param[out]      last_stepsize   If supplied, store last step-size taken by the integrator for each IVP. Useful for OpenFOAM / chemistry timescale integration
         * \returns system state vectors at time t_end
         *
         */
        void intDriver (const int NUM, const double t,
                        const double* __restrict__ t_end,
                        const double* __restrict__ param, double* __restrict__ phi,
                        double* __restrict__ last_stepsize=NULL)
        {
            if (NUM < 1)
            {
                throw std::runtime_error("Number of problems to solve must be at least one!");
            }

            if (NUM != _storedNumProblems)
            {
                // resize data if needed
                this->reinitialize(_numWorkGroups, NUM);
            }

            // set start
            set_start_time(t);

            auto t_data = std::chrono::high_resolution_clock::now();

            if (_end_time_index < 0 || _phi_index < 0 || _param_index < 0 || _solver_index < 0 || !_initialized)
            {
                throw std::runtime_error("Implementation error!");
            }

            /* transfer data to device */
            CL_EXEC( clEnqueueWriteBuffer(_data.command_queue, _clmem[_param_index], CL_TRUE, 0, sizeof(double)*NUM, param, 0, NULL, NULL) );
            CL_EXEC( clEnqueueWriteBuffer(_data.command_queue, _clmem[_end_time_index], CL_TRUE, 0, sizeof(double)*NUM, t_end, 0, NULL, NULL) );
            CL_EXEC( clEnqueueWriteBuffer(_data.command_queue, _clmem[_phi_index], CL_TRUE, 0, sizeof(double)*NUM*_neq, phi, 0, NULL, NULL) );
            CL_EXEC( clEnqueueWriteBuffer(_data.command_queue, _clmem[_solver_index], CL_TRUE, 0, sizeof(solver_struct),
                                          &this->getSolverStruct(), 0, NULL, NULL) );
            if (_options.useQueue())
            {
                // initialize queue with: global work-size * the vectorSize, such that each work-group has at least
                // one IVP to solve (and more importantly, we can avoid synchronization between work-groups)
                int queue_val = numThreads() * _data.vectorSize;
                CL_EXEC( clEnqueueWriteBuffer(_data.command_queue, _clmem[_queue_index], CL_TRUE, 0, sizeof(int),
                                              &queue_val, 0, NULL, NULL) )
                if (verbose())
                    std::cout << "Queue enabled" << std::endl;
            }

            if (_options.estimateChemistryTime())
            {
                if (last_stepsize == NULL)
                {
                    throw std::runtime_error("Must supply a valid `last_stepsize` buffer when estimating chemistry time!");
                }
                CL_EXEC( clEnqueueWriteBuffer(_data.command_queue, _clmem[_chem_time_index], CL_TRUE, 0, sizeof(double)*NUM,
                                              last_stepsize, 0, NULL, NULL) );

            }

            if (verbose())
                std::cout << "Host->Dev = " <<
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::high_resolution_clock::now() - t_data).count() <<
                    " (ms)" << std::endl;


            /* Execute kernel */
            auto tk = std::chrono::high_resolution_clock::now();
            cl_event ev;
            std::size_t nt = numThreads();
            CL_EXEC( clEnqueueNDRangeKernel (_data.command_queue, _kernel,
                                             1 /* work-group dims */,
                                             NULL /* offset */,
                                             &nt /* global work size */,
                                             &_data.blockSize /* local work-group size */,
                                             0, NULL, /* wait list */
                                             &ev /* this kernel's event */) );
            /* Wait for the kernel to finish */
            clWaitForEvents(1, &ev);
            auto tk_end = std::chrono::high_resolution_clock::now();
            if (verbose())
                std::cout << "Kernel execution = " <<
                    std::chrono::duration_cast<std::chrono::milliseconds>(tk_end - tk).count() <<
                    " (ms)" << std::endl;


            t_data = std::chrono::high_resolution_clock::now();
            /* copy out */
            CL_EXEC( clEnqueueReadBuffer(_data.command_queue, _clmem[_phi_index], CL_TRUE, 0, sizeof(double)*_neq*NUM, phi, 0, NULL, NULL) );
            if (_options.estimateChemistryTime())
            {
                if (last_stepsize == NULL)
                {
                    throw std::runtime_error("Must supply a valid `last_stepsize` buffer when estimating chemistry time!");
                }
                CL_EXEC( clEnqueueReadBuffer(_data.command_queue, _clmem[_chem_time_index], CL_TRUE, 0, sizeof(double)*NUM,
                                             last_stepsize, 0, NULL, NULL) );

            }

            counter_struct* counters = (counter_struct*) malloc(sizeof(counter_struct)*NUM);
            if (counters == NULL)
            {
                std::ostringstream err;
                err << "Allocation error " << __FILE__ << " " << __LINE__ << nl
                throw std::runtime_error(err.str());
            }
            CL_EXEC( clEnqueueReadBuffer(_data.command_queue, _clmem[_counter_index], CL_TRUE, 0, sizeof(counter_struct)*NUM, counters, 0, NULL, NULL) );

            int nst_ = 0, nit_ = 0;
            for (int i = 0; i < NUM; ++i)
            {
                checkError(i, (ErrorCode)counters[i].niters);
                nst_ += counters[i].nsteps;
                nit_ += counters[i].niters;
            }
            if (verbose())
                std::cout << "nst = " << nst_ << ", nit = " << nit_ << std::endl;

            free(counters);
        }

    protected:
        //! \brief return a reference to the initialized solver_struct
        virtual const solver_struct& getSolverStruct() const = 0;

    };

}

#endif
