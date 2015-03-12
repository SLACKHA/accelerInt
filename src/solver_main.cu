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
#include "solver_init.cuh"
#include "gpu_memory.cuh"
#include "read_initial_conditions.h"

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

    initialize_solver();

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

    /* Block size */
    int BLOCK_SIZE = 64;

    int g_num = (int)floor(((double)NUM) / ((double)BLOCK_SIZE) + 0.5);
    dim3 dimGrid (g_num, 1 );
    dim3 dimBlock(BLOCK_SIZE, 1);

    // print number of threads and block size
    printf ("# threads: %d \t block size: %d\n", NUM, BLOCK_SIZE);

    // time span
    double t_start = 0.0;
    double t_end = 1.0e-3;
    double h = 1.0e-6;

    double* y_device;
    double* y_host;
#ifdef CONP
    double* pres_device;
    double* pres_host;
#ifdef SAME_IC
    int padded = set_same_initial_conditions(NUM, BLOCK_SIZE, g_num, &y_host, &y_device, &pres_host, &pres_device);
#else
    int padded = read_initial_conditions(NUM, BLOCK_SIZE, g_num, &y_host, &y_device, &pres_host, &pres_device);
#endif
#elif CONV
    double* rho_device;
    double* rho_device;
#ifdef SAME_IC
    int padded = set_same_initial_conditions(NUM, BLOCK_SIZE, g_num, &y_host, &y_device, &rho_host, &rho_device);
#else
    int padded = read_initial_conditions(NUM, BLOCK_SIZE, g_num, &y_host, &y_device, &rho_host, &rho_device);
#endif
#endif

// flag for ignition
#ifdef IGN
    bool ign_flag = false;
    // ignition delay time, units [s]
    double t_ign = ZERO;
    double T0 = y_host[0];
#endif

#ifdef LOG_OUTPUT
    // file for data
    FILE *pFile;
    const char* f_name = solver_name();
    int len = strlen(f_name);
    char out_name[len + 9];
    sprintf(out_name, "%s-log.txt", f_name);
    pFile = fopen(out_name, "w");

    fprintf(pFile, "%e", t_start);
    for (int i = 0; i < NN; ++i)
    {
        fprintf(pFile, "\t%e", y_host[NUM * i]);
    }
    fprintf(pFile, "\n");
#endif


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
    Real t = t_start;
    Real t_next = t + h;
    int numSteps = 0;

    // time integration loop
    while (t < t_end)
    {
        numSteps++;
        // transfer memory to GPU
        cudaErrorCheck( cudaMemcpy (y_device, y_host, padded * NN * sizeof(double), cudaMemcpyHostToDevice) );

#if defined(CONP)
        // constant pressure case
        intDriver <<< dimGrid, dimBlock>>> (NUM, t, t_next, pres_device, y_device);
#elif defined(CONV)
        // constant volume case
        intDriver <<< dimGrid, dimBlock>>> (NUM, t, t_next, rho_device, y_device);
#endif

        t = t_next;
        t_next += h;

        // transfer memory back to CPU
        cudaErrorCheck( cudaMemcpy (y_host, y_device, padded * NN * sizeof(double), cudaMemcpyDeviceToHost) );

#if defined(DEBUG) || defined(PRINT) 
        printf("%.15le\t%.15le\n", t, y_host[0]);
#endif
#ifdef DEBUG
        // check if within bounds
        if ((y_host[0] < ZERO) || (y_host[0] > 10000.0))
        {
            printf("Error, out of bounds.\n");
            printf("Time: %e, ind %d val %e\n", t, 0, y_host[0]);
            return 1;
        }
#endif
#ifdef LOG_OUTPUT
        printf("%.15le", t);
        for (int i = 0; i < NN; i++) {
        	fprintf(pFile, ",");
        	fprintf(pFile, "%.15le", y_host[i * NUM]);
        }
        fprintf(pFile, "\n");
#endif
#ifdef IGN
        // determine if ignition has occurred
        if ((y_host[0] >= (T0 + 400.0)) && !(ign_flag)) {
            ign_flag = true;
            t_ign = t;
        }
#endif
    }

    /////////////////////////////////
    // end timer
    double runtime = GetTimer();
    /////////////////////////////////


    runtime /= 1000.0;
    printf ("Time: %e sec\n", runtime);
    runtime = runtime / ((Real)(numSteps));
    printf ("Time per step: %e (s)\t%e (s/thread)\n", runtime, runtime / NUM);

#ifdef LOG_OUTPUT
    fclose (pFile);
#endif

    free (y_host);
#ifdef CONP
    free (pres_host);
#elif CONV
    free (rho_host);
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