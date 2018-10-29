#include "solver.hpp"

namespace opencl_solvers
{

    void IntegratorBase::void initialize_kernel()
    {
        cl_int ret;

        /* get device, context, platform, etc. */
        cl_init(&_data);

        /* get the kernel name */
        std::ostringstream kernel_name;
        // all named driver
        kernel_name << "driver";
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

    cl_mem IntegratorBase::CreateBuffer (cl_context *context, cl_mem_flags flags, std::size_t size, void *host_ptr)
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


    void IntegratorBase::getPlatformInfo (platformInfo_t *info)
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

    void IntegratorBase::void getDeviceInfo (
        const cl_uint& device_type,
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


    int IntegratorBase::cl_init (cl_data_t* data)
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
        data->blockSize = std::max(_options.blockSize(), (size_t)1);
        data->numBlocks = data->device_info.max_compute_units;
        if (numWorkGroups() > 0)
        {
            // if user has specified the number of work groups to use
            data->numBlocks = numWorkGroups();
        }
        data->vectorSize = std::max(_options.vectorSize(), (size_t)1);
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
        rwk << "#define rwk_lensrc (" << _ivp.requiredMemorySize() << ")" << std::endl;
        program_source_str << rwk.str();

        std::ostringstream rwks;
        rwks << "#define rwk_lensol (" << requiredSolverMemorySize() / sizeof(double) << ")" << std::endl;
        program_source_str << rwks.str();

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

        if (_options.stepperType() == StepperType::CONSTANT)
        {
            std::ostringstream stype;
            if (std::isnan(_options.constantTimestep()))
            {
                throw std::runtime_error("Constant time-step not specified!");
            }
            stype << "#define CONSTANT_TIMESTEP (" << _options.constantTimestep() << ")" << std::endl;
            program_source_str << stype.str();
        }

        // Load the common macros ...
        load_source_from_file (file_relative_to_me(__FILE__, "solver.h"), program_source_str);
        // load error codes
        load_source_from_file (file_relative_to_me(__FILE__, "error_codes.h"), program_source_str);

        // Load any solver types for the driver
        for (const std::string& file : solverIncludes())
        {
            program_source_str << "#include \"" << file << "\"" << std::endl;
        }
        // load the drivers
        load_source_from_file (file_relative_to_me(__FILE__, "drivers.cl"), program_source_str);

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

        for (const std::string& ipath : _ivp.includePaths())
        {
            build_options << " -I" << ipath;
        }

        if (data->platform_info.is_nvidia)
             build_options << " -cl-nv-verbose";

        if (_verbose)
            std::cout << "build_options = " << build_options.str() << std::endl;

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


    template <typename solver_struct, typename counter_struct>
    void Integrator<solver_struct, counter_struct>::resize(const int NUM)
    {
        // create memory
        std::size_t lenrwk = (_ivp.requiredMemorySize() + requiredSolverMemorySize())*_data.vectorSize;
        if (_verbose)
            std::cout << "lenrwk = "<< lenrwk << std::endl;

        std::size_t leniwk = (_ivp.requiredIntegerMemorySize() + requiredSolverIntegerMemorySize()) * _data.vectorSize;
        if (_verbose)
            std::cout << "leniwk = "<< leniwk << std::endl;

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
            _queue_index = indexof(_clmem.begin(), _clmem.end(), buffer_queue);;
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

    }


    template <typename solver_struct, typename counter_struct>
    void Integrator<solver_struct, counter_struct>::intDriver (
        const int NUM, const double t,
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
            checkError(i, (ErrorCode)counters[i].niters);
            nst_ += counters[i].nsteps;
            nit_ += counters[i].niters;
        }
        if (_verbose)
            std::cout << "nst = " << nst_ << ", nit = " << nit_ << std::endl;

        free(counters);
        // turn off messaging after first compilation
        _verbose = false;
    }

}
