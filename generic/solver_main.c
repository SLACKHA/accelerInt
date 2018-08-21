/**
 * \file
 * \brief the generic main file for all CPU solvers
 *
 * \author Nicholas Curtis
 * \date 03/09/2015
 *
 * Contains main function, setup, initialization, logging, timing and driver functions
 */


/** Include common code. */
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdbool.h>
#include <complex.h>
#include <sys/types.h>
#include <sys/stat.h>
#ifdef DEBUG
#include <fenv.h>
#endif

//our code
#include "header.h"
#include "solver.h"
#include "timer.h"
#include "read_initial_conditions.h"

#ifdef GENERATE_DOCS
namespace generic {
#endif

/**
 * \brief Writes state vectors to file
 * \param[in]           NUM                 the number of state vectors to write
 * \param[in]           t                   the current system time
 * \param[in]           y_host              the current state vectors
 * \param[in]           pFile               the opened binary file object
 *
 * The resulting file is updated as:
 * system time\n
 * temperature, mass fractions (State #1)\n
 * temperature, mass fractions (State #2)...
 */
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
        #if NN == NSP + 1 //pyjac
        buffer[NSP] = Y_N;
        #endif
        apply_reverse_mask(&buffer[1]);
        fwrite(buffer, sizeof(double), NN, pFile);
    }
}


//////////////////////////////////////////////////////////////////////////////

/** Main function
 *
 * \param[in]       argc    command line argument count
 * \param[in]       argv    command line argument vector
 *
 * This allows running the integrators from the command line.  The syntax is as follows:\n
 * `./solver-name [num_threads] [num_IVPs]`\n
 * *  num_threads  [Optional, Default:1]
 *      *  The number OpenMP threads to utilize
 *      *  The number of threads cannot be greater than recognized by OpenMP via `omp_get_max_threads()`
 * *  num_IVPs     [Optional, Default:1]
 *      *  The number of initial value problems to solve.
 *      *  This must be less than the number of conditions in the data file if #SAME_IC is not defined.
 *      *  If #SAME_IC is defined, then the initial conditions in the mechanism files will be used.
 *
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

    /////////////////////////////////////////////////
    // arrays
    /////////////////////////////////////////////////

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
    struct stat info;
    if (stat("./log/", &info) != 0)
    {
        printf("Expecting 'log' subdirectory in current working directory. Please run"
               " mkdir log (or the equivalent) and run again.\n");
        exit(-1);
    }
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
    double t = 0;
    double t_next = fmin(end_time, t_step);
    int numSteps = 0;

    // time integration loop
    while (t + EPS < end_time)
    {
        numSteps++;

        intDriver(NUM, t, t_next, var_host, y_host);
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


    runtime /= 1000.0;
    printf ("Time: %.15e sec\n", runtime);
    runtime = runtime / ((double)(numSteps));
    printf ("Time per step: %e (s)\t%.15e (s/thread)\n", runtime, runtime / NUM);
#ifdef IGN
    printf ("Ig. Delay (s): %e\n", t_ign);
#endif
    printf("TFinal: %e\n", y_host[0]);

#ifdef LOG_OUTPUT
    fclose (pFile);
#endif

    free (y_host);
    free(var_host);
    cleanup_solver(num_threads);

    return 0;
}

#ifdef GENERATE_DOCS
}
#endif
