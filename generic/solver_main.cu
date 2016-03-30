/* solver_main.cu
 * the generic main file for all exponential solvers
 * \file solver_main.cu
 *
 * \author Nicholas Curtis
 * \date 03/09/2015
 *
 * Contains main and integration driver functions.
 */


/** Include common code. */
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdbool.h>
#include <complex.h>

/** Include CUDA libraries. */
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <cuComplex.h>

//our code
#include "header.cuh"
#include "timer.h"
//get our solver stuff
#include "solver.cuh"
#include "gpu_memory.cuh"
#include "read_initial_conditions.cuh"
#include "launch_bounds.cuh"

#ifdef DIVERGENCE_TEST
    #include <assert.h>
    __device__ int integrator_steps[DIVERGENCE_TEST] = {0};
#endif

void write_log(int NUM, double t, const double* y_host, FILE* pFile)
{
    fwrite(&t, sizeof(double), 1, pFile);
    double buffer[NN];
    for (int j = 0; j < NUM; j++)
    {
        double Y_N = 1.0;
        buffer[0] = y_host[j];
        for (int i = 1; i < NSP; ++i)
        {
            buffer[i] = y_host[NUM * i + j];
            Y_N -= buffer[i];
        }
        buffer[NSP] = Y_N;
        apply_reverse_mask(&buffer[1]);
        fwrite(buffer, sizeof(double), NN, pFile);
    }
}

inline void memcpy2D_in(double* dst, const int pitch_dst, double const * src, const int pitch_src,
                                     const int offset, const size_t width, const int height) {
    for (int i = 0; i < height; ++i)
    {
        memcpy(dst, &src[offset], width);
        dst += pitch_dst;
        src += pitch_src;
    }
}

inline void memcpy2D_out(double* dst, const int pitch_dst, double const * src, const int pitch_src,
                                      const int offset, const size_t width, const int height) {
    for (int i = 0; i < height; ++i)
    {
        memcpy(&dst[offset], src, width);
        dst += pitch_dst;
        src += pitch_src;
    }
}

//////////////////////////////////////////////////////////////////////////////

/** Main function
 *
 *
 *
 * \param[in]       argc    command line argument count
 * \param[in]       argv    command line argument vector
 */
int main (int argc, char *argv[])
{

//enable signaling NAN and other bad numerics tracking for easier debugging 
#ifdef DEBUG
    feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);
#endif
    /** Number of independent systems */
    int NUM = 1;

    // check for problem size given as command line option
    if (argc > 1)
    {
        int problemsize = NUM;
        if (sscanf(argv[1], "%i", &problemsize) != 1 || (problemsize <= 0))
        {
            printf("Error: Problem size not in correct range\n");
            printf("Provide number greater than 0\n");
            exit(1);
        }
        NUM = problemsize;
    }

    // set & initialize device using command line argument (if any)
    cudaDeviceProp devProp;
    if (argc <= 2)
    {
        // default device id is 0
        cudaErrorCheck (cudaSetDevice (0) );
        cudaErrorCheck (cudaGetDeviceProperties(&devProp, 0));
    }
    else
    {
        // use second argument for number

        // get number of devices
        int num_devices;
        cudaGetDeviceCount(&num_devices);

        int id = 0;
        if (sscanf(argv[2], "%i", &id) == 1 && (id >= 0) && (id < num_devices))
        {
            cudaErrorCheck( cudaSetDevice (id) );
        }
        else
        {
            // not in range, error
            printf("Error: GPU device number not in correct range\n");
            printf("Provide number between 0 and %i\n", num_devices - 1);
            exit(1);
        }
        cudaErrorCheck (cudaGetDeviceProperties(&devProp, id));
    }
    cudaErrorCheck( cudaDeviceReset() );
    cudaErrorCheck( cudaPeekAtLastError() );
    cudaErrorCheck( cudaDeviceSynchronize() );
    #ifdef DIVERGENCE_TEST
        NUM = DIVERGENCE_TEST;
        assert(NUM % 32 == 0);
    #endif
    //bump up shared mem bank size
    cudaErrorCheck(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));
    //and L1 size
    cudaErrorCheck(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));

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

    solver_memory* host_solver, *device_solver;
    mechanism_memory* host_mech, *device_mech;
    host_solver = (solver_memory*)malloc(sizeof(solver_memory));
    host_mech = (mechanism_memory*)malloc(sizeof(mechanism_memory));

    initialize_gpu_memory(padded, &host_mech, &device_mech);
    initialize_solver(padded, &host_solver, &device_solver);

    // print number of threads and block size
    printf ("# threads: %d \t block size: %d\n", NUM, TARGET_BLOCK_SIZE);

#ifdef SHUFFLE 
    const char* filename = "shuffled_data.bin";
#elif !defined(SAME_IC)
    const char* filename = "ign_data.bin";
#endif

    double* y_host;
    double* var_host;
#ifdef SAME_IC
    set_same_initial_conditions(NUM, &y_host, &var_host);
#else
    read_initial_conditions(filename, NUM, &y_host, &var_host);
#endif

    dim3 dimGrid (padded / TARGET_BLOCK_SIZE, 1 );
    dim3 dimBlock(TARGET_BLOCK_SIZE, 1);
    int* result_flag = (int*)malloc(padded * sizeof(int));

// flag for ignition
#ifdef IGN
    bool ign_flag = false;
    // ignition delay time, units [s]
    double t_ign = 0.0;
    double T0 = y_host[0];
#endif

#ifdef LOG_OUTPUT
    // file for data
    FILE *pFile;
    const char* f_name = solver_name();
    int len = strlen(f_name);
    char out_name[len + 13];
    sprintf(out_name, "log/%s-log.bin", f_name);
    pFile = fopen(out_name, "wb");

    write_log(NUM, 0, y_host, pFile);

    //initialize integrator specific log
    init_solver_log();
#endif

    double* y_temp = 0;
    y_temp = (double*)malloc(padded * NSP * sizeof(double));

    //////////////////////////////
    // start timer
    StartTimer();
    //////////////////////////////

    // set initial time
    double t = 0;
    double t_next = fmin(end_time, t_step);
    int numSteps = 0;

    // time integration loop
    while (t + EPS < end_time)
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
            intDriver <<< dimGrid, dimBlock, SHARED_SIZE >>> (NUM, t, t_next, host_mech->var, host_mech->y, device_mech, device_solver);
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
        t_next = fmin(end_time, (numSteps + 1) * t_step);

    #if defined(PRINT)
            printf("%.15le\t%.15le\n", t, y_host[0]);
    #endif
    #ifdef DEBUG
            // check if within bounds
            if ((y_host[0] < 0.0) || (y_host[0] > 10000.0))
            {
                printf("Error, out of bounds.\n");
                printf("Time: %e, ind %d val %e\n", t, 0, y_host[0]);
                return 1;
            }
    #endif
    #ifdef LOG_OUTPUT
            #if !defined(LOG_END_ONLY)
                write_log(NUM, t, y_host, pFile);
                solver_log();
            #endif
    #endif
    #ifdef IGN
            // determine if ignition has occurred
            if ((y_host[0] >= (T0 + 400.0)) && !(ign_flag)) {
                ign_flag = true;
                t_ign = t;
            }
    #endif
    }

#ifdef LOG_END_ONLY
    write_log(NUM, t, y_host, pFile);
    solver_log();
#endif

    /////////////////////////////////
    // end timer
    double runtime = GetTimer();
    /////////////////////////////////

    #ifdef DIVERGENCE_TEST
    int host_integrator_steps[DIVERGENCE_TEST];
    cudaErrorCheck( cudaMemcpyFromSymbol(host_integrator_steps, integrator_steps, DIVERGENCE_TEST * sizeof(int)) );
    int warps = NUM / 32;

    FILE *dFile;
    const char* f_name = solver_name();
    int len = strlen(f_name);
    char out_name[len + 13];
    sprintf(out_name, "log/%s-div.txt", f_name);
    dFile = fopen(out_name, "w");
    int index = 0;
    for (int i = 0; i < warps; ++i)
    {
        double d = 0;
        int max = 0;
        for (int j = 0; j < 32; ++j)
        {
            int steps = host_integrator_steps[index];
            d += steps;
            max = steps > max ? steps : max;
            index++;
        }
        d /= (32.0 * max);
        fprintf(dFile, "%.15e\n", d);
    }
    fclose(dFile);
    #endif

    runtime /= 1000.0;
    printf ("Time: %e sec\n", runtime);
    runtime = runtime / ((double)(numSteps));
    printf ("Time per step: %e (s)\t%e (s/thread)\n", runtime, runtime / NUM);
#ifdef IGN
    printf ("Ig. Delay (s): %e\n", t_ign);
#endif
    printf("TFinal: %e\n", y_host[0]);

#ifdef LOG_OUTPUT
    fclose (pFile);
#endif


    free_gpu_memory(&host_mech, &device_mech);
    cleanup_solver(&host_solver, &device_solver);
    free(y_temp);
    free(host_mech);
    free(host_solver);
    free(result_flag);
    cudaErrorCheck( cudaDeviceReset() );

    return 0;
}