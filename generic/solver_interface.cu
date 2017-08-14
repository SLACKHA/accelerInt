/**
 * \file
 * \brief Interface implementation for GPU solvers to be called as a library
 *
 * \author Nicholas Curtis
 * \date 03/09/2015
 *
 * Contains initialization, integration and cleanup functions
 */

#include "solver_interface.cuh"

#ifdef GENERATE_DOCS
namespace genericcu {
#endif

//! Padded # of ODEs to solve
int padded;
//! The solver memory structs
solver_memory* host_solver, *device_solver;
//! The mechanism memory structs
mechanism_memory* host_mech, *device_mech;
//! block and grid sizes
dim3 dimBlock, dimGrid;
//! result flag
int* result_flag;
//! temorary storage
double* y_temp;

/**
 * \brief A convienience method to copy memory between host pointers of different pitches, widths and heights.
 *        Enables easier use of CUDA's cudaMemcpy2D functions.
 *
 * \param[out]              dst             The destination array
 * \param[in]               pitch_dst       The width (in bytes) of the destination array.
                                            This corresponds to the padded number of IVPs to be solved.
 * \param[in]               src             The source pointer
 * \param[in]               pitch_src       The width (in bytes)  of the source array.
                                            This corresponds to the (non-padded) number of IVPs read by read_initial_conditions
 * \param[in]               offset          The offset within the source array (IVP index) to copy from.
                                            This is useful in the case (for large models) where the solver and state vector memory will not fit in device memory
                                            and the integration must be split into multiple kernel calls.
 * \param[in]               width           The size (in bytes) of memory to copy for each entry in the state vector
 * \param[in]               height          The number of entries in the state vector
 */
inline void memcpy2D_in(double* dst, const int pitch_dst, double const * src, const int pitch_src,
                                     const int offset, const size_t width, const int height) {
    for (int i = 0; i < height; ++i)
    {
        memcpy(dst, &src[offset], width);
        dst += pitch_dst;
        src += pitch_src;
    }
}

/**
 * \brief A convienience method to copy memory between host pointers of different pitches, widths and heights.
 *        Enables easier use of CUDA's cudaMemcpy2D functions.
 *
 * \param[out]              dst             The destination array
 * \param[in]               pitch_dst       The width (in bytes)  of the source array.
                                            This corresponds to the (non-padded) number of IVPs read by read_initial_conditions
 * \param[in]               src             The source pointer
 * \param[in]               pitch_src       The width (in bytes) of the destination array.
                                            This corresponds to the padded number of IVPs to be solved.
 * \param[in]               offset          The offset within the destination array (IVP index) to copy to.
                                            This is useful in the case (for large models) where the solver and state vector memory will not fit in device memory
                                            and the integration must be split into multiple kernel calls.
 * \param[in]               width           The size (in bytes) of memory to copy for each entry in the state vector
 * \param[in]               height          The number of entries in the state vector
 */
inline void memcpy2D_out(double* dst, const int pitch_dst, double const * src, const int pitch_src,
                                      const int offset, const size_t width, const int height) {
    for (int i = 0; i < height; ++i)
    {
        memcpy(&dst[offset], src, width);
        dst += pitch_dst;
        src += pitch_src;
    }
}


/**
 * \brief Initializes the solver
 * \param[in]       NUM         The number of ODEs to integrate
 * \param[in]       device      The CUDA device number, if < 0 set to the first available GPU
 */
void accelerInt_initialize(int NUM, int device) {
    device = device < 0 ? 0 : device;

    // set & initialize device using command line argument (if any)
    cudaDeviceProp devProp;
    // get number of devices
    int num_devices;
    cudaGetDeviceCount(&num_devices);

    if ((device >= 0) && (device < num_devices))
    {
        cudaErrorCheck( cudaSetDevice (device) );
    }
    else
    {
        // not in range, error
        printf("Error: GPU device number not in correct range\n");
        printf("Provide number between 0 and %i\n", num_devices - 1);
        exit(1);
    }
    cudaErrorCheck (cudaGetDeviceProperties(&devProp, device));

    // reset device
    cudaErrorCheck( cudaDeviceReset() );
    cudaErrorCheck( cudaPeekAtLastError() );
    cudaErrorCheck( cudaDeviceSynchronize() );

    //bump up shared mem bank size
    cudaErrorCheck(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));
    //and L1 size
    cudaErrorCheck(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));

    //get the memory sizes
    size_t size_per_thread = required_mechanism_size() + required_solver_size();
    size_t free_mem = 0;
    size_t total_mem = 0;
    cudaErrorCheck( cudaMemGetInfo (&free_mem, &total_mem) );

    //conservatively estimate the maximum allowable threads
    int max_threads = int(floor(0.8 * ((double)free_mem) / ((double)size_per_thread)));
    int padded = min(NUM, max_threads);
    //padded is next factor of block size up
    padded = int(ceil(padded / float(TARGET_BLOCK_SIZE)) * TARGET_BLOCK_SIZE);
    if (padded == 0)
    {
        printf("Mechanism is too large to fit into global CUDA memory... exiting.");
        exit(-1);
    }

    //initalize memory
    initialize_gpu_memory(padded, &host_mech, &device_mech);
    initialize_solver(padded, &host_solver, &device_solver);

    //grid sizes
    dimBlock = dim3(TARGET_BLOCK_SIZE, 1);
    dimGrid = dim3(padded / TARGET_BLOCK_SIZE, 1 );
    //local storage
    result_flag = (int*)malloc(padded * sizeof(int));
    y_temp = (double*)malloc(padded * NSP * sizeof(double));
}


/**
 * \brief integrate NUM odes from time `t_start` to time `t_end`, using stepsizes of `stepsize`
 *
 * \param[in]           NUM             The number of ODEs to integrate.  This should be the size of the leading dimension of `y_host` and `var_host`.  @see accelerint_indx
 * \param[in]           t_start         The starting time
 * \param[in]           t_end           The end time
 * \param[in]           stepsize        The integration step size.  If `stepsize` < 0, the step size will be set to `t_end - t`
 * \param[in,out]       y_host          The state vectors to integrate.
 * \param[in]           var_host        The parameters to use in dydt() and eval_jacob()
 *
 */
void accelerInt_integrate(const int NUM, const double t_start, const double t_end, const double stepsize,
                          double * __restrict__ y_host, const double * __restrict__ var_host)
{
    double step = stepsize < 0 ? t_end - t_start : stepsize;
    double t = t_start;
    double t_next = fmin(end_time, t + step);
    int numSteps = 0;

    // time integration loop
    while (t + EPS < t_end)
    {
        numSteps++;
        int num_solved = 0;
        while (num_solved < NUM)
        {
            int num_cond = min(NUM - num_solved, padded);

            cudaErrorCheck( cudaMemcpy (host_mech->var, &var_host[num_solved],
                                        num_cond * sizeof(double), cudaMemcpyHostToDevice));

             //copy our memory into y_temp
            memcpy2D_in(y_temp, padded, y_host, NUM,
                            num_solved, num_cond * sizeof(double), NSP);
            // transfer memory to GPU
            cudaErrorCheck( cudaMemcpy2D (host_mech->y, padded * sizeof(double),
                                            y_temp, padded * sizeof(double),
                                            num_cond * sizeof(double), NSP,
                                            cudaMemcpyHostToDevice) );
            intDriver <<< dimGrid, dimBlock, SHARED_SIZE >>> (num_cond, t, t_next, host_mech->var, host_mech->y, device_mech, device_solver);
    #ifdef DEBUG
            cudaErrorCheck( cudaPeekAtLastError() );
            cudaErrorCheck( cudaDeviceSynchronize() );
    #endif
            // copy the result flag back
            cudaErrorCheck( cudaMemcpy(result_flag, host_solver->result, num_cond * sizeof(int), cudaMemcpyDeviceToHost) );
            check_error(num_cond, result_flag);
            // transfer memory back to CPU
            cudaErrorCheck( cudaMemcpy2D (y_temp, padded * sizeof(double),
                                            host_mech->y, padded * sizeof(double),
                                            num_cond * sizeof(double), NSP,
                                            cudaMemcpyDeviceToHost) );
            memcpy2D_out(y_host, NUM, y_temp, padded,
                            num_solved, num_cond * sizeof(double), NSP);

            num_solved += num_cond;

        }
        t = t_next;
        t_next = fmin(t_end, (numSteps + 1) * step);
    }
}


/**
 * \brief Cleans up the solver
 */
void accelerInt_cleanup() {
    free_gpu_memory(&host_mech, &device_mech);
    cleanup_solver(&host_solver, &device_solver);
    free(y_temp);
    free(host_mech);
    free(host_solver);
    free(result_flag);
    cudaErrorCheck( cudaDeviceReset() );
}




#ifdef GENERATE_DOCS
}
#endif