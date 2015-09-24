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

//our code
#include "header.h"
#include "solver.h"
#include "timer.h"
#include "read_initial_conditions.h"

void write_log(int NUM, double t, const double* y_host, FILE* pFile)
{
    fwrite(&t, sizeof(double), 1, pFile);
    double buffer[NN];
    for (int j = 0; j < NUM; j++)
    {
        for (int i = 0; i < NN; ++i)
        {
            buffer[i] = y_host[NUM * i + j];
        }
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

    int NUM = 1;
    int num_threads = 1;

    /////////////////////////////////////////////////////////////////////
    // OpenMP
    /////////////////////////////////////////////////////////////////////
    // set & initialize OpenMP threads via command line argument (if any)
    #ifdef _OPENMP
    int max_threads = omp_get_max_threads ();
    if (argc == 1) {
        // set to max threads (environment variable)
        omp_set_num_threads (max_threads);
    } else {
        if (argc > 1) {
            num_threads = max_threads;
            // first check if is number
              if (sscanf(argv[1], "%i", &num_threads) !=1 || (num_threads <= 0) || (num_threads > max_threads)) {
                printf("Error: Number of threads not in correct range\n");
                    printf("Provide number between 1 and %i\n", max_threads);
                    exit(1);
              }
            omp_set_num_threads (num_threads);
        }

        if (argc > 2) { //check for problem size
            int problemsize = NUM;
            if (sscanf(argv[2], "%i", &problemsize) !=1 || (problemsize <= 0))
            {
                printf("Error: Problem size not in correct range\n");
                    printf("Provide number greater than 0\n");
                    exit(1);
            }
            NUM = problemsize;
        }
    }

    //get max number of threads from OMP
    num_threads = 0;
    #pragma omp parallel reduction(+:num_threads)
    num_threads += 1;

    #endif

    // print number of independent ODEs
    printf ("# ODEs: %d\n", NUM);
    printf ("# threads: %d\n", num_threads);

    initialize_solver(num_threads);

    // time span
    double t_start = 0.0;
#ifdef SAME_IC
    double t_end = 1000 * t_step;
#else
    double t_end = 10 * t_step;
#endif

    /////////////////////////////////////////////////
    // arrays
    /////////////////////////////////////////////////

#ifdef SHUFFLE 
    const char* filename = "shuffled_data.bin";
#elif !defined(SAME_IC)
    const char* filename = "ign_data.bin";
#endif

    double* y_host;
#ifdef CONP
    double* pres_host;
#elif CONV
    double* rho_host;
#endif

#ifdef SAME_IC
#ifdef CONP
    set_same_initial_conditions(NUM, &y_host, &pres_host);
#else
    set_same_initial_conditions(NUM, &y_host, &rho_host);
#endif
#else
#ifdef CONP
    read_initial_conditions(filename, NUM, &y_host, &pres_host);
#elif CONV
    read_initial_conditions(filename, NUM, &y_host, &rho_host);
#endif
#endif

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
    init_solver_log();
#endif

    //////////////////////////////
    // start timer
    StartTimer();
    //////////////////////////////

    // set initial time
    double t = t_start;
    double t_next = t + t_step;
    int numSteps = 0;

    // time integration loop
    while (t < t_end)
    {
        numSteps++;

#if defined(CONP)
        // constant pressure case
        intDriver (NUM, t, t_next, pres_host, y_host);
#elif defined(CONV)
        // constant volume case
        intDriver (NUM, t, t_next, rho_host, y_host);
#endif

        t = t_next;
        t_next += t_step;

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
        write_log(NUM, t, y_host, pFile);
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
    runtime = runtime / ((double)(numSteps));
    printf ("Time per step: %e (s)\t%e (s/thread)\n", runtime, runtime / NUM);
#ifdef IGN
    printf ("Ig. Delay (s): %e\n", t_ign);
#endif
    printf("TFinal: %e\n", y_host[0]);

#ifdef LOG_OUTPUT
    fclose (pFile);
#endif

    free (y_host);
#ifdef CONP
    free (pres_host);
#elif CONV
    free (rho_host);
#endif
    cleanup_solver(num_threads);

    return 0;
}