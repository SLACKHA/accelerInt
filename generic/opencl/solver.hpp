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
#include <algorithm>
#include <stdexcept>
#include "error_codes.hpp"
#include "../paths/path.h"
#include "../paths/resolver.h"
extern "C" {
#include "CL/cl.h"
}

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
}

#define CL_EXEC(__cmd__) {\
    cl_uint _ret = (__cmd__); \
    if (_ret != CL_SUCCESS) \
    { \
        __clerror(_ret, __cmd__); \
        exit(-1); \
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
            std::size_t requiredMemorySize):
                _kernelSource(kernelSource),
                _requiredMemorySize(requiredMemorySize)
        {

        }

        //! \brief Return the list of filenames of OpenCL kernels that implement
        //!        the source term and Jacobian
        const std::vector<std::string>& kernelSource() const
        {
            return _kernelSource;
        }

        //! \brief Return the (unvectorized) memory-size required by the IVP kernels
        std::size_t requiredMemorySize() const
        {
            return _requiredMemorySize;
        }

    protected:
        std::vector<std::string> _kernelSource;
        std::size_t _requiredMemorySize;

    };

    class SolverOptions
    {
    public:
        SolverOptions(std::size_t vectorSize=1, std::size_t blockSize=1,
                      double atol=1e-10, double rtol=1e-6,
                      bool logging=false, double h_init=1e-6,
                      bool use_queue=true, std::string order="C",
                      std::string platform = "", DeviceType deviceType=DeviceType::DEFAULT):
            _vectorSize(vectorSize),
            _blockSize(blockSize),
            _atol(atol),
            _rtol(rtol),
            _logging(logging),
            _h_init(h_init),
            _order(order),
            _use_queue(use_queue),
            _platform(platform),
            _deviceType(deviceType)
        {
            if (order.compare("C") && order.compare("F"))
            {
                std::ostringstream err;
                err << "Order " << order << " not recognized";
                throw OpenCLException(err.str());
            }
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

        inline double h_init() const
        {
            return _h_init;
        }

        inline double vectorSize() const
        {
            return _vectorSize;
        }

        inline double blockSize() const
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
        //! The initial step-size
        const double _h_init;
        //! The data-ordering
        const std::string _order;
        //! Use queue driver?
        const bool _use_queue;
        //! OpenCL platform to use
        const std::string _platform;
        //! The OpenCL device type to use
        const DeviceType _deviceType;
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
            _verbose(true),
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
            _initialized(false)
        {

        }

        ~IntegratorBase()
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
            throw std::runtime_error("not implemented");
        }

        //! return the absolute tolerance
        inline const double atol() const
        {
            return _options.atol();
        }

        //! return the relative tolerance
        inline const double rtol() const
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

        //! \brief return the number of equations to solve
        inline const int neq() const
        {
            return _neq;
        }

        //! \brief Return the number of OpenCL work-groups to launch
        //         On the CPU / Accelerators this corresponds to the # of threads
        //         On the GPU, this corresponds to the # of "thread-blocks", in CUDA terminology
        inline const int numWorkGroups() const
        {
            return _numWorkGroups;
        }

        //! return the initial step-size
        inline const double h_init() const
        {
            return _options.h_init();
        }


        /**
         * \brief Integration driver for the CPU integrators
         * \param[in]       NUM             The (non-padded) number of IVPs to integrate
         * \param[in]       t               The current system time
         * \param[in]       t_end           The IVP integration end time
         * \param[in]       param           The system constant variable (pressures / densities)
         * \param[in,out]   phi             The system state vectors at time t.
         * \returns system state vectors at time t_end
         *
         */
        void intDriver (const int NUM, const double t,
                        const double t_end, const double* __restrict__ param,
                        double* __restrict__ phi)
        {
            std::vector<double> t_end_vec (t_end, NUM);
            this->intDriver(NUM, t, &t_end_vec[0], param, phi);
        }

        /**
         * \brief Integration driver for the CPU integrators
         * \param[in]       NUM             The (non-padded) number of IVPs to integrate
         * \param[in]       t               The (array) of current system times
         * \param[in]       t_end           The (array) of IVP integration end times
         * \param[in]       param           The system constant variable (pressures / densities)
         * \param[in,out]   phi             The system state vectors at time t.
         * \returns system state vectors at time t_end
         *
         */
        virtual void intDriver (const int NUM, const double t,
                                const double* __restrict__ t_end,
                                const double* __restrict__ param, double* __restrict__ phi) = 0;



    protected:
        //! return reference to the beginning of the working memory
        //! for this thread `tid`
        double* phi(int tid);

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
        //! \brief Verbosity of kernel compilation
        bool _verbose;
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
        //! \brief simple flag to mark whether the device / context / kernel have been created
        bool _initialized;


        /*
         * \brief Return the required memory size in bytes (per-IVP)
         */
        virtual std::size_t requiredSolverMemorySize() = 0;

        //! \brief return the base kernel name
        virtual const char* solverName() const = 0;

        //! \brief an initialization function for the kernel, to be called from
        //!        the final derived integrator class's constructor
        void initialize_kernel()
        {
            cl_int ret;

            /* get device, context, platform, etc. */
            cl_init(&_data);

            /* get the kernel name */
            std::ostringstream kernel_name;
            kernel_name << this->solverName();
            // all named driver
            kernel_name << "_driver";
            // and queue
            if (_options.useQueue())
            {
                kernel_name << "_queue";
            }
            std::string k_name = kernel_name.str();

            /* Extract the kernel from the program */
            _kernel = clCreateKernel(_data.program, k_name.c_str(), &ret);
            if (ret != CL_SUCCESS)
            {

                __clerror(ret, "clCreateKernel");
                exit(-1);
            }

            // Query the kernel's info
            getKernelInfo (&_kernel_info, _kernel, _data.device_info.device_id);
            // mark initialized
            _initialized = true;
        }

        //! \brief return the list of files for this solver
        virtual const std::vector<std::string>& solverFiles() const = 0;

        //! \brief return the list of include paths for this solver
        virtual const std::vector<std::string>& solverIncludePaths() const = 0;


        const std::string path_of(const std::string& owner) const
        {
            // base path
            filesystem::path thisfile = filesystem::path(owner);
            thisfile.make_absolute();
            return thisfile.parent_path().str();
        }


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

        cl_mem CreateBuffer (cl_context *context, cl_mem_flags flags, std::size_t size, void *host_ptr)
        {
            cl_int ret;
            cl_mem buf = clCreateBuffer (*context, flags, size + __Alignment, host_ptr, &ret);
            //assert( ret == CL_SUCCESS );
            if (ret != CL_SUCCESS)
            {
                __clerror(ret, "CreateBuffer");
                exit(-1);
            }

            // and zero memory
            char zero = 0;
            CL_EXEC(clEnqueueFillBuffer(_data.command_queue, buf, &zero, sizeof(char), 0,
                                        size, 0, NULL, NULL));

            return buf;
        }

        cl_device_type getDeviceType (cl_device_id device_id)
        {
            cl_device_type val;
            CL_EXEC( clGetDeviceInfo (device_id, CL_DEVICE_TYPE, sizeof(cl_device_type), &val, NULL) );

            return val;
        }

        void printKernelInfo (const kernelInfo_t* info)
        {
            if (_verbose)
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

        void getPlatformInfo (platformInfo_t *info)
        {
            CL_EXEC ( clGetPlatformIDs ((cl_uint)cl_max_platforms, info->platform_ids, &info->num_platforms) );
            if (info->num_platforms == 0)
            {
                std::cerr << "clError: num_platforms = 0" << std::endl;
                exit(-1);
            }

            bool found = false;
            if (!_options.platform().empty())
            {
                std::string test(_options.platform());
                std::transform(test.begin(), test.end(), test.begin(), ::tolower);
                // find user specified platform
                for (std::size_t i = 0; i < info->num_platforms; ++i)
                {
                    char name[1024] = {0};
                    CL_EXEC( clGetPlatformInfo (info->platform_ids[i], CL_PLATFORM_NAME, sizeof(name), name, NULL) );
                    std::string cname(name);
                    std::transform(cname.begin(), cname.end(), cname.begin(), ::tolower);
                    if (cname.find(test) != std::string::npos)
                    {
                        if (_verbose)
                            std::cout << "Found user specified platform " << _options.platform() << std::endl;
                        info->platform_id = info->platform_ids[i];
                        found = true;
                        break;
                    }
                }
            }

            if (!found)
            {
                char name[1024] = {0};
                CL_EXEC( clGetPlatformInfo (info->platform_ids[0], CL_PLATFORM_NAME, sizeof(name), name, NULL) );
                if (_verbose)
                    std::cout << "User specified platform either not specified or not found, defaulting to platform: (" <<
                        name << ")" << std::endl;
                info->platform_id = info->platform_ids[0];
            }

            #define __getInfo( __STR, __VAR) { \
                CL_EXEC( clGetPlatformInfo (info->platform_id, (__STR), sizeof(__VAR), __VAR, NULL) ); \
                if (_verbose)\
                    std::cout << "\t" << #__STR << " = " << __VAR << std::endl; }

            if (_verbose)
                std::cout << "Platform Info:" << std::endl;
            __getInfo( CL_PLATFORM_NAME,       info->name);
            __getInfo( CL_PLATFORM_VERSION,    info->version);
            __getInfo( CL_PLATFORM_VENDOR,     info->vendor);
            __getInfo( CL_PLATFORM_EXTENSIONS, info->extensions);

            #undef __getInfo

            info->is_nvidia = 0; // Is an NVIDIA device?
            if (strstr(info->vendor, "NVIDIA") != NULL)
                info->is_nvidia = 1;
            if (_verbose)
                std::cout << "\tIs-NVIDIA = " << info->is_nvidia << std::endl;
        }

        void getDeviceInfo (const cl_uint& device_type,
                            deviceInfo_t *device_info,
                            const platformInfo_t *platform_info)
        {

            CL_EXEC( clGetDeviceIDs (platform_info->platform_id, CL_DEVICE_TYPE_ALL, (cl_uint)cl_max_devices, device_info->device_ids, &device_info->num_devices) );
            if (device_info->num_devices == 0)
            {
                std::cerr << "clError: num_devices = 0" << std::endl;
                exit(-1);
            }

            device_info->device_id = device_info->device_ids[0];

            #define get_char_info(__str__, __val__, __verbose__) { \
                CL_EXEC( clGetDeviceInfo (device_info->device_id, (__str__), sizeof(__val__), __val__, NULL) ); \
                if (__verbose__) std::cout << "\t" << #__str__ << " = " << __val__ << std::endl; \
            }
            #define get_info(__str__, __val__, __verbose__) { \
                CL_EXEC( clGetDeviceInfo (device_info->device_id, __str__, sizeof(__val__), &__val__, NULL) ); \
                if (__verbose__) std::cout << "\t" << #__str__ << " = " << __val__ << std::endl; \
            }

            if (_verbose)
                std::cout << "Device Info:" << std::endl;

            get_info( CL_DEVICE_TYPE, device_info->type, _verbose);

            for (std::size_t i = 0; i < device_info->num_devices; ++i)
            {
                cl_device_type val;
                //get_info( CL_DEVICE_TYPE, val, 1);
                CL_EXEC( clGetDeviceInfo (device_info->device_ids[i], CL_DEVICE_TYPE, sizeof(cl_device_type), &val, NULL) );
                if (_verbose)
                {
                    std::cout << "\t" << "CL_DEVICE_TYPE" << " = " << val << std::endl;
                }

                if (val == device_type)
                {
                    device_info->device_id = device_info->device_ids[i];
                    device_info->type = val;
                    break;
                }
            }

            std::string device_type_name;
            {
                if (device_info->type == CL_DEVICE_TYPE_GPU)
                     device_type_name = "GPU";
                else if (device_info->type == CL_DEVICE_TYPE_CPU)
                     device_type_name = "CPU";
                else if (device_info->type == CL_DEVICE_TYPE_ACCELERATOR)
                     device_type_name = "ACCELERATOR";
                else if (device_info->type == CL_DEVICE_TYPE_DEFAULT)
                     device_type_name = "DEFAULT";
            }
            if (_verbose) std::cout<< "\t" << "Type Name = " << device_type_name << std::endl;

            get_char_info( CL_DEVICE_NAME, device_info->name, _verbose );
            get_char_info( CL_DEVICE_PROFILE, device_info->profile, _verbose );
            get_char_info( CL_DEVICE_VERSION, device_info->version, _verbose );
            get_char_info( CL_DEVICE_VENDOR, device_info->vendor, _verbose );
            get_char_info( CL_DRIVER_VERSION, device_info->driver_version, _verbose );
            get_char_info( CL_DEVICE_OPENCL_C_VERSION, device_info->opencl_c_version, _verbose );
            //get_char_info( CL_DEVICE_BUILT_IN_KERNELS );
            get_char_info( CL_DEVICE_EXTENSIONS, device_info->extensions, _verbose );

            get_info( CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR, device_info->native_vector_width_char, _verbose );
            get_info( CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT, device_info->native_vector_width_short, _verbose );
            get_info( CL_DEVICE_NATIVE_VECTOR_WIDTH_INT, device_info->native_vector_width_int, _verbose );
            get_info( CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG, device_info->native_vector_width_long, _verbose );
            get_info( CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT, device_info->native_vector_width_float, _verbose );
            get_info( CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE, device_info->native_vector_width_double, _verbose );
            get_info( CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF, device_info->native_vector_width_half, _verbose );

            get_info( CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR, device_info->preferred_vector_width_char, _verbose );
            get_info( CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT, device_info->preferred_vector_width_short, _verbose );
            get_info( CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT, device_info->preferred_vector_width_int, _verbose );
            get_info( CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG, device_info->preferred_vector_width_long, _verbose );
            get_info( CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, device_info->preferred_vector_width_float, _verbose );
            get_info( CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE, device_info->preferred_vector_width_double, _verbose );
            get_info( CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF, device_info->preferred_vector_width_half, _verbose );

            get_info( CL_DEVICE_MAX_COMPUTE_UNITS, device_info->max_compute_units, _verbose );
            get_info( CL_DEVICE_MAX_CLOCK_FREQUENCY, device_info->max_clock_frequency, _verbose );

            get_info( CL_DEVICE_MAX_WORK_GROUP_SIZE, device_info->max_work_group_size, _verbose );

            get_info( CL_DEVICE_GLOBAL_MEM_SIZE, device_info->global_mem_size, _verbose );
            get_info( CL_DEVICE_MAX_MEM_ALLOC_SIZE, device_info->max_mem_alloc_size, _verbose );
            get_info( CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, device_info->global_mem_cacheline_size, _verbose );
            get_info( CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, device_info->global_mem_cache_size, _verbose );

            get_info( CL_DEVICE_GLOBAL_MEM_CACHE_TYPE, device_info->global_mem_cache_type, _verbose);
            if (_verbose)
            {
                std::cout << "\t" << "CL_DEVICE_GLOBAL_MEM_CACHE_TYPE = ";
                if (device_info->global_mem_cache_type == CL_NONE)
                     std::cout << "CL_NONE" << std::endl;
                else if (device_info->global_mem_cache_type == CL_READ_ONLY_CACHE)
                     std::cout << "CL_READ_ONLY_CACHE" << std::endl;
                else if (device_info->global_mem_cache_type == CL_READ_WRITE_CACHE)
                     std::cout << "CL_READ_WRITE_CACHE" << std::endl;
            }

            get_info( CL_DEVICE_LOCAL_MEM_SIZE, device_info->local_mem_size, _verbose );
            get_info( CL_DEVICE_LOCAL_MEM_TYPE, device_info->local_mem_type, _verbose );
            if (_verbose)
            {
                std::cout << "\t" << "CL_DEVICE_LOCAL_MEM_TYPE = " <<
                          ((device_info->local_mem_type == CL_LOCAL) ? "LOCAL" : "GLOBAL") <<
                          std::endl;
            }

            get_info( CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, device_info->max_constant_buffer_size, _verbose );
            get_info( CL_DEVICE_MAX_CONSTANT_ARGS, device_info->max_constant_args, _verbose );

            get_info( CL_DEVICE_DOUBLE_FP_CONFIG, device_info->fp_config, _verbose );

            #undef get_char_info
            #undef get_info
        }

        int cl_init (cl_data_t* data)
        {
            cl_int ret;

            getPlatformInfo(&data->platform_info);

            getDeviceInfo(_options.deviceType(), &data->device_info, &data->platform_info);

            data->context = clCreateContext(NULL, 1, &data->device_info.device_id, NULL, NULL, &ret);
            if (ret != CL_SUCCESS )
            {
                __clerror(ret, "clCreateContext");
                exit(-1);
            }

            data->command_queue = clCreateCommandQueue(data->context, data->device_info.device_id, 0, &ret);
            if (ret != CL_SUCCESS )
            {
                __clerror(ret, "clCreateCommandQueue");
                exit(-1);
            }

            data->use_queue = _options.useQueue();
            data->blockSize = _options.blockSize();
            data->numBlocks = data->device_info.max_compute_units;
            if (numWorkGroups() > 0)
            {
                // if user has specified the number of work groups to use
                data->numBlocks = numWorkGroups();
            }
            data->vectorSize = _options.vectorSize();
            if (data->vectorSize > 0 && !isPower2(data->vectorSize))
            {
                std::ostringstream err;
                err << "Vector size: " << data->vectorSize << " is not a power of 2!";
                throw OpenCLException(err.str());
            }
            if (data->blockSize > 0 && !isPower2(data->blockSize))
            {
                std::ostringstream err;
                err << "Block size: " << data->blockSize << " is not a power of 2!";
                throw OpenCLException(err.str());
            }

            if (data->blockSize < data->vectorSize)
                data->blockSize = data->vectorSize;

            data->blockSize /= data->vectorSize;

            std::ostringstream program_source_str;
            std::string dp_header = "#if defined(cl_khr_fp64) \n"
                                    "#pragma OPENCL EXTENSION cl_khr_fp64 : enable  \n"
                                    "#elif defined(cl_amd_fp64) \n"
                                    "#pragma OPENCL EXTENSION cl_amd_fp64 : enable  \n"
                                    "#endif \n";
            program_source_str << dp_header;

            if (__Alignment)
            {
                std::ostringstream align;
                align << "#define __Alignment (" << __Alignment << ") "<< std::endl;
                program_source_str << align.str();
            }

            std::ostringstream vsize;
            vsize << "#define __ValueSize " << data->vectorSize << std::endl;
            program_source_str << vsize.str();

            std::ostringstream bsize;
            bsize << "#define __blockSize (" << data->blockSize << ")" << std::endl;
            program_source_str << bsize.str();

            // neq
            std::ostringstream sneq;
            sneq << "#define neq (" << _neq << ")" << std::endl;
            program_source_str << sneq.str();

            // source rate evaluation work-size
            std::ostringstream rwk;
            rwk << "#define rk_lensrc (" << _ivp.requiredMemorySize() << ")" << std::endl;
            program_source_str << rwk.str();

            // order
            std::ostringstream sord;
            sord << "#define __order '" << _options.order() << "'" << std::endl;
            program_source_str << sord.str();

            if (data->use_queue)
            {
                std::ostringstream oqueue;
                oqueue << "#define __EnableQueue" << std::endl;
                program_source_str << oqueue.str();
            }

            // Load the common macros ...
            load_source_from_file (file_relative_to_me(__FILE__, "solver.h"), program_source_str);

            // Load the header and source files text ...
            for (const std::string& file : solverFiles())
            {
                load_source_from_file (file, program_source_str);
            }

            // load the user specified kernels last such that any macros we use there are already
            // defined
            for (const std::string& kernel : _ivp.kernelSource())
            {
                load_source_from_file(kernel, program_source_str);
            }

            /* Build Program */
            std::string psource_str = program_source_str.str();
            const char* psource = psource_str.c_str();
            std::size_t psource_len = psource_str.length();
            data->program = clCreateProgramWithSource(data->context, 1,
                                                      (const char **)&psource,
                                                      (const size_t *)&psource_len, &ret);
            if (ret != CL_SUCCESS)
            {
                __clerror(ret, "clCreateProgramWithSource");
            }

            #define DEBUG_CL_COMPILATION
            #ifdef DEBUG_CL_COMPILATION
            // write to file
            std::ofstream temp("temp.cl");
            temp << psource_str;
            temp.close();
            #endif

            std::ostringstream build_options;
            build_options << "-I" << path_of(__FILE__);
            for (const std::string& ipath : solverIncludePaths())
            {
                build_options << " -I" << ipath;
            }

            if (data->platform_info.is_nvidia)
                 build_options << " -cl-nv-verbose";

            if (_verbose)
                std::cout << "build_options = " << build_options.str();

            std::string cbuild = build_options.str();
            ret = clBuildProgram(data->program, 1, &data->device_info.device_id, cbuild.c_str(), NULL, NULL);
            if (_verbose)
                std::cout << "clBuildProgram = " << ret << std::endl;


            cl_build_status build_status;
            CL_EXEC( clGetProgramBuildInfo (data->program, data->device_info.device_id, CL_PROGRAM_BUILD_STATUS,
                                            sizeof(cl_build_status), &build_status, NULL) );
            if (_verbose || ret != CL_SUCCESS)
                std::cout << "CL_PROGRAM_BUILD_STATUS = " << build_status << std::endl;;

            // get the program build log size
            size_t build_log_size;
            CL_EXEC( clGetProgramBuildInfo (data->program, data->device_info.device_id, CL_PROGRAM_BUILD_LOG,
                                            0, NULL, &build_log_size) );

            // and alloc the build log
            std::vector<char> build_log(build_log_size + 1);
            CL_EXEC( clGetProgramBuildInfo (data->program, data->device_info.device_id, CL_PROGRAM_BUILD_LOG,
                                            build_log_size, &build_log[0], &build_log_size) );
            if (build_log_size > 0 && (_verbose || ret != CL_SUCCESS))
            {
                std::string blog(build_log.begin(), build_log.end());
                std::cout << "CL_PROGRAM_BUILD_LOG = " << blog << std::endl;
            }

            if (ret != CL_SUCCESS)
            {
                __clerror(ret, "clBuildProgram");
            }

            if (data->platform_info.is_nvidia && 0)
            {
                /* Query binary (PTX file) size */
                size_t binary_size;
                CL_EXEC( clGetProgramInfo (data->program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &binary_size, NULL) );

                /* Read binary (PTX file) to memory buffer */
                std::vector<char> ptx_binary(binary_size + 1);
                CL_EXEC( clGetProgramInfo (data->program, CL_PROGRAM_BINARIES, sizeof(unsigned char *), &ptx_binary[0], NULL) );
                /* Save PTX to add_vectors_ocl.ptx */
                FILE *ptx_binary_file = std::fopen("ptx_binary_ocl.ptx", "wb");
                std::fwrite(&ptx_binary[0], sizeof(char), binary_size, ptx_binary_file);
                std::fclose(ptx_binary_file);
            }

            return CL_SUCCESS;
        }

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
            // release kernel
            CL_EXEC(clReleaseKernel(_kernel));
            // release program
            CL_EXEC(clReleaseProgram(_data.program));
            // release queue
            CL_EXEC(clReleaseCommandQueue(_data.command_queue));
            // and context
            CL_EXEC(clReleaseContext(_data.context));
        }

    protected:

        void resize(const int NUM)
        {
            // create memory
            std::size_t lenrwk = (_ivp.requiredMemorySize() + requiredSolverMemorySize())*_data.vectorSize;
            if (_verbose)
                std::cout << "lenrwk = "<< lenrwk << std::endl;

            if (_verbose)
                std::cout << "NP = " << NUM << ", blockSize = " << _data.blockSize <<
                             ", vectorSize = " << _data.vectorSize <<
                             ", numBlocks = " << _data.numBlocks <<
                             ", numThreads = " << numThreads() << std::endl;

            cl_mem buffer_param = CreateBuffer (&_data.context, CL_MEM_READ_ONLY, sizeof(double)*NUM, NULL);
            cl_mem tf = CreateBuffer (&_data.context, CL_MEM_READ_ONLY, sizeof(double)*NUM, NULL);
            cl_mem buffer_phi = CreateBuffer (&_data.context, CL_MEM_READ_WRITE, sizeof(double)*_neq*NUM, NULL);
            cl_mem buffer_solver = CreateBuffer (&_data.context, CL_MEM_READ_ONLY, sizeof(solver_struct), NULL);
            cl_mem buffer_rwk = CreateBuffer (&_data.context, CL_MEM_READ_WRITE, lenrwk*numThreads(), NULL);
            cl_mem buffer_counters = CreateBuffer (&_data.context, CL_MEM_READ_WRITE, sizeof(counter_struct)*NUM, NULL);
            _clmem.assign({buffer_param, tf, buffer_phi, buffer_solver, buffer_rwk, buffer_counters});

            _param_index = 0;
            _end_time_index = 1;
            _phi_index = 2;
            _solver_index = 3;
            _counter_index = 5;

            cl_mem buffer_queue;
            if (_options.useQueue())
            {
                buffer_queue = CreateBuffer (&_data.context, CL_MEM_READ_WRITE, sizeof(int), NULL);
                _clmem.push_back(buffer_queue);
                _queue_index = 6;
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
            CL_EXEC( clSetKernelArg(_kernel, argc++, sizeof(cl_mem), &buffer_counters) );
            CL_EXEC( clSetKernelArg(_kernel, argc++, sizeof(int), &NUM) );
            if (_options.useQueue())
            {
                CL_EXEC( clSetKernelArg(_kernel, argc++, sizeof(cl_mem), &buffer_queue) );
            }

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
         * \returns system state vectors at time t_end
         *
         */
        void intDriver (const int NUM, const double t,
                        const double* __restrict__ t_end,
                        const double* __restrict__ param, double* __restrict__ phi)
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
                if (_verbose)
                    std::cout << "Queue enabled" << std::endl;
            }

            if (_verbose)
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
            if (_verbose)
                std::cout << "Kernel execution = " <<
                    std::chrono::duration_cast<std::chrono::milliseconds>(tk_end - tk).count() <<
                    " (ms)" << std::endl;


            t_data = std::chrono::high_resolution_clock::now();
            /* copy out */
            CL_EXEC( clEnqueueReadBuffer(_data.command_queue, _clmem[_phi_index], CL_TRUE, 0, sizeof(double)*_neq*NUM, phi, 0, NULL, NULL) );

            counter_struct* counters = (counter_struct*) malloc(sizeof(counter_struct)*NUM);
            if (counters == NULL)
            {
                fprintf(stderr,"Allocation error %s %d\n", __FILE__, __LINE__);
                exit(-1);
            }
            CL_EXEC( clEnqueueReadBuffer(_data.command_queue, _clmem[_counter_index], CL_TRUE, 0, sizeof(counter_struct)*NUM, counters, 0, NULL, NULL) );

            int nst_ = 0, nit_ = 0;
            for (int i = 0; i < NUM; ++i)
            {
                nst_ += counters[i].nsteps;
                nit_ += counters[i].niters;
            }
            if (_verbose)
                std::cout << "nst = " << nst_ << ", nit = " << nit_ << std::endl;

            free(counters);
            // turn off messaging after first compilation
            _verbose = false;
        }

    protected:
        //! \brief return a reference to the initialized solver_struct
        virtual const solver_struct& getSolverStruct() const = 0;

    };

}

#endif
