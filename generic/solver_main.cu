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

void write_log(int padded, int NUM, double t, const double* y_host, FILE* pFile)
{
    fwrite(&t, sizeof(double), 1, pFile);
    double buffer[NN];
    for (int j = 0; j < NUM; j++)
    {
        double Y_N = 1.0;
        for (int i = 0; i < NSP; ++i)
        {
            buffer[i] = y_host[padded * i + j];
            Y_N -= buffer[i];
        }
        buffer[NSP] = Y_N;
        apply_reverse_mask(&buffer[1]);
        fwrite(buffer, sizeof(double), NN, pFile);
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
            checkCudaErrors (cudaSetDevice (id) );
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
    #ifdef DIVERGENCE_TEST
        NUM = DIVERGENCE_TEST;
        assert(NUM % 32 == 0);
    #endif
    //bump up shared mem bank size
    cudaErrorCheck(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));
    //and L1 size
    cudaErrorCheck(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));

    initialize_solver();

    int g_num = (int)ceil(((double)NUM) / ((double)TARGET_BLOCK_SIZE));
    if (g_num == 0)
        g_num = 1;

    // print number of threads and block size
    printf ("# threads: %d \t block size: %d\n", NUM, TARGET_BLOCK_SIZE);

#ifdef SHUFFLE 
    const char* filename = "shuffled_data.bin";
#elif !defined(SAME_IC)
    const char* filename = "ign_data.bin";
#endif

    double* y_device;
    double* y_host;
#ifdef CONP
    double* pres_device;
    double* pres_host;
#ifdef SAME_IC
    int padded = set_same_initial_conditions(NUM, &y_host, &y_device, &pres_host, &pres_device);
#else
    int padded = read_initial_conditions(filename, NUM, TARGET_BLOCK_SIZE, g_num, &y_host, &y_device, &pres_host, &pres_device);
#endif
#elif CONV
    double* rho_device;
    double* rho_host;
#ifdef SAME_IC
    int padded = set_same_initial_conditions(NUM, TARGET_BLOCK_SIZE, g_num, &y_host, &y_device, &rho_host, &rho_device);
#else
    int padded = read_initial_conditions(filename, NUM, TARGET_BLOCK_SIZE, g_num, &y_host, &y_device, &rho_host, &rho_device);
#endif
#endif

    dim3 dimGrid (g_num, 1 );
    dim3 dimBlock(TARGET_BLOCK_SIZE, 1);

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

    write_log(padded, NUM, 0, y_host, pFile);

    //initialize integrator specific log
    init_solver_log();
#endif

    //////////////////////////////
    // start timer
    StartTimer();
    //////////////////////////////

    //begin memory copy
#ifdef CONP
    cudaErrorCheck( cudaMemcpy (pres_device, pres_host, padded * sizeof(double), cudaMemcpyHostToDevice));
#elif CONV
    cudaErrorCheck( cudaMemcpy (rho_device, rho_host, padded * sizeof(double), cudaMemcpyHostToDevice));
#endif

    // set initial time
    double t = 0;
    double t_next = fmin(end_time, t_step);
    int numSteps = 0;

    // time integration loop
    while (t + EPS < end_time)
    {
        numSteps++;
        // transfer memory to GPU
        cudaErrorCheck( cudaMemcpy (y_device, y_host, padded * NSP * sizeof(double), cudaMemcpyHostToDevice) );

#if defined(CONP)
        // constant pressure case
        intDriver <<< dimGrid, dimBlock, SHARED_SIZE >>> (NUM, t, t_next, pres_device, y_device);
#elif defined(CONV)
        // constant volume case
        intDriver <<< dimGrid, dimBlock, SHARED_SIZE>>> (NUM, t, t_next, rho_device, y_device);
#endif
#ifdef DEBUG
        cudaErrorCheck( cudaPeekAtLastError() );
        cudaErrorCheck( cudaDeviceSynchronize() );
#endif
        // transfer memory back to CPU
        cudaErrorCheck( cudaMemcpy (y_host, y_device, padded * NSP * sizeof(double), cudaMemcpyDeviceToHost) );

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
            write_log(padded, NUM, t, y_host, pFile);
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

    cudaFreeHost (y_host);
#ifdef CONP
    cudaFreeHost (pres_host);
#elif CONV
    cudaFreeHost (rho_host);
#endif
    cleanup_solver();

#ifdef CONP
    free_gpu_memory(y_device, pres_device);
#elif CONV
    free_gpu_memory(y_device, rho_device);
#endif
    cudaErrorCheck( cudaDeviceReset() );

    return 0;
}