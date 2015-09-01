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

#ifdef DEBUG
//NAN check
#include <fenv.h>
#endif

//our code
#include "header.h"
#include "timer.h"
//get our solver stuff
#include "solver.cuh"
#include "gpu_memory.cuh"
#include "read_initial_conditions.cuh"
#include "launch_bounds.cuh"

#ifdef LOG_KRYLOV_AND_STEPSIZES
    //make logging array definitions
    __device__ double err_log[MAX_STEPS];
    __device__ int m_log[MAX_STEPS];
    __device__ int m1_log[MAX_STEPS];
    __device__ int m2_log[MAX_STEPS];
    __device__ double t_log[MAX_STEPS];
    __device__ double h_log[MAX_STEPS];
    __device__ bool reject_log[MAX_STEPS];
    __device__ int num_integrator_steps;
#endif

void write_log(int padded, int NUM, double t, const double* y_host, FILE* pFile)
{
    double buffer[NN + 2];
    for (int j = 0; j < NUM; j++)
    {
        buffer[0] = t;
        for (int i = 0; i < NN; ++i)
        {
            buffer[i + 1] = y_host[padded * i];
        }
        fwrite(buffer, sizeof(double), NN + 2, pFile);
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

    // time span
    double t_start = 0.0;
#ifdef SAME_IC
    double t_end = 1000 * t_step;
#else
    double t_end = 10 * t_step;
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
#endif


#ifdef LOG_KRYLOV_AND_STEPSIZES
    //create host logging arrays
    double* err_log_host = (double*)malloc(MAX_STEPS * sizeof(double));
    int* m_log_host = (int*)malloc(MAX_STEPS * sizeof(int));
    int* m1_log_host = (int*)malloc(MAX_STEPS * sizeof(int));
    int* m2_log_host = (int*)malloc(MAX_STEPS * sizeof(int));
    double* t_log_host = (double*)malloc(MAX_STEPS * sizeof(double));
    double* h_log_host = (double*)malloc(MAX_STEPS * sizeof(double));
    bool* reject_log_host = (bool*)malloc(MAX_STEPS * sizeof(bool));
    int num_integrator_steps_host = 0;
    //open files for krylov logging
    FILE *logFile;
    //open and clear
    const char* f_new_name = solver_name();
    int len_new = strlen(f_new_name);
    char out_name_new[len_new + 17];
    sprintf(out_name_new, "log/%s-kry-log.txt", f_new_name);
    logFile = fopen(out_name_new, "w");

    char out_reject_name[len_new + 23];
    sprintf(out_reject_name, "log/%s-kry-reject.txt", f_new_name);    
    //file for krylov logging
    FILE *rFile;
    //open and clear
    rFile = fopen(out_reject_name, "w");
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
    double t = t_start;
    double t_next = t + t_step;
    int numSteps = 0;

    // time integration loop
    while (t < t_end)
    {
        numSteps++;
        // transfer memory to GPU
        cudaErrorCheck( cudaMemcpy (y_device, y_host, padded * NN * sizeof(double), cudaMemcpyHostToDevice) );

#if defined(CONP)
        // constant pressure case
#ifdef SHARED_SIZE
        intDriver <<< dimGrid, dimBlock, SHARED_SIZE >>> (NUM, t, t_next, pres_device, y_device);
#else
        intDriver <<< dimGrid, dimBlock >>> (NUM, t, t_next, pres_device, y_device);
#endif
#elif defined(CONV)
        // constant volume case
#ifdef SHARED_SIZE
        intDriver <<< dimGrid, dimBlock, SHARED_SIZE>>> (NUM, t, t_next, rho_device, y_device);
#else
        intDriver <<< dimGrid, dimBlock >>> (NUM, t, t_next, rho_device, y_device);
#endif
#endif
#ifdef DEBUG
        cudaErrorCheck( cudaPeekAtLastError() );
        cudaErrorCheck( cudaDeviceSynchronize() );
#endif
        // transfer memory back to CPU
        cudaErrorCheck( cudaMemcpy (y_host, y_device, padded * NN * sizeof(double), cudaMemcpyDeviceToHost) );

        t = t_next;
        t_next += t_step;


#if defined(DEBUG) || defined(PRINT) 
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
        write_log(padded, NUM, t, y_host, pFile);
#endif
#ifdef IGN
        // determine if ignition has occurred
        if ((y_host[0] >= (T0 + 400.0)) && !(ign_flag)) {
            ign_flag = true;
            t_ign = t;
        }
#endif
#ifdef LOG_KRYLOV_AND_STEPSIZES
        //first copy back num steps to make sure we're inbounds
        cudaErrorCheck( cudaMemcpyFromSymbol(&num_integrator_steps_host, num_integrator_steps, sizeof(int)) );
        if (num_integrator_steps_host == -1)
            exit(-1);
        //otherwise copy back
        cudaErrorCheck( cudaMemcpyFromSymbol(err_log_host, err_log, num_integrator_steps_host * sizeof(double)) );
        cudaErrorCheck( cudaMemcpyFromSymbol(m_log_host, m_log, num_integrator_steps_host * sizeof(int)) );
        cudaErrorCheck( cudaMemcpyFromSymbol(m1_log_host, m1_log, num_integrator_steps_host * sizeof(int)) );
        cudaErrorCheck( cudaMemcpyFromSymbol(m2_log_host, m2_log, num_integrator_steps_host * sizeof(int)) );
        cudaErrorCheck( cudaMemcpyFromSymbol(t_log_host, t_log, num_integrator_steps_host * sizeof(double)) );
        cudaErrorCheck( cudaMemcpyFromSymbol(h_log_host, h_log, num_integrator_steps_host * sizeof(double)) );
        cudaErrorCheck( cudaMemcpyFromSymbol(reject_log_host, reject_log, num_integrator_steps_host * sizeof(bool)) );
        //and print
        for (int i = 0; i < num_integrator_steps_host; ++i)
        {
            if (reject_log_host[i])
            {
                fprintf(rFile, "%.15le\t%.15le\t%.15le\t%d\t%d\t%d\n", t_log_host[i], h_log_host[i], err_log_host[i], m_log_host[i], m1_log_host[i], m2_log_host[i]);
            }
            else
            {
                fprintf(logFile, "%.15le\t%.15le\t%.15le\t%d\t%d\t%d\n", t_log_host[i], h_log_host[i], err_log_host[i], m_log_host[i], m1_log_host[i], m2_log_host[i]);
            }
        }
#endif
    }

    /////////////////////////////////
    // end timer
    double runtime = GetTimer();
    /////////////////////////////////


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
#ifdef LOG_KRYLOV_AND_STEPSIZES
    free(err_log_host);
    free(m_log_host);
    free(m1_log_host);
    free(m2_log_host);
    free(t_log_host);
    free(h_log_host);
    free(reject_log_host);
    fclose(rFile);
    fclose(logFile);
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