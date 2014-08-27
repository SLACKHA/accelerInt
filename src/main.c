/** Main function file for exponential integration of H2 problem project.
 * \file main.c
 *
 * \author Kyle E. Niemeyer
 * \date 08/27/2013
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

#include "header.h"
#include "cf.h"
#include "exp4.h"
#include "mass_mole.h"
#include "timer.h"

#ifdef DEBUG
//NAN check
#include <fenv.h> 
#endif

Real complex poles[N_RA];
Real complex res[N_RA];

#define IGN

// load same initial conditions for all threads
#define SAME_IC

// shuffle initial conditions randomly
//#define SHUFFLE

/////////////////////////////////////////////////////////////////////////////

void intDriver (const int NUM, const Real t, const Real t_end, 
                const Real* pr_global, Real* y_global) {

	int tid;
	#pragma omp parallel for shared(y_global, pr_global) private(tid)
	for (tid = 0; tid < NUM; ++tid) {
		// local array with initial values
		Real y_local[NN];
		Real pr_local = pr_global[tid];

		// load local array with initial values from global array
		#pragma unroll
		for (int i = 0; i < NN; i++)
		{
			y_local[i] = y_global[tid + i * NUM];
		}

		// call integrator for one time step
		exp4_int (t, t_end, pr_local, y_local);

		// update global array with integrated values
		#pragma unroll
		for (int i = 0; i < NN; i++)
		{
			y_global[tid + i * NUM] = y_local[i];
		}

	} // end tid loop

} // end intDriver

/////////////////////////////////////////////////////////////////////////////

/** Main function
 * 
 * 
 * 
 * \param[in]		argc	command line argument count
 * \param[in]		argv	command line argument vector
 */
int main (int argc, char *argv[]) {
	
	#ifdef DEBUG
		feenableexcept(FE_DIVBYZERO|FE_INVALID|FE_OVERFLOW);
	#endif

	// get poles and residues for rational approximant to matrix exponential
	double *poles_r = (double*) calloc (N_RA, sizeof(double));
	double *poles_i = (double*) calloc (N_RA, sizeof(double));
	double *res_r = (double*) calloc (N_RA, sizeof(double));
	double *res_i = (double*) calloc (N_RA, sizeof(double));
	
	cf ( N_RA, poles_r, poles_i, res_r, res_i );
	
	for (int i = 0; i < N_RA; ++i) {
		poles[i] = poles_r[i] + poles_i[i] * _Complex_I;
		res[i] = res_r[i] + res_i[i] * _Complex_I;
	}
	
	// free memory
	free (poles_r);
	free (poles_i);
	free (res_r);
	free (res_i);
  
  /** Number of independent systems */
  int NUM = 1;
  
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
  		int num_threads = max_threads;
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
  
  // print number of independent ODEs
  printf ("# ODEs: %d\n", NUM);
  // print number of threads
  int num_threads = 0;
  
  #pragma omp parallel reduction(+:num_threads)
  num_threads += 1;
  printf ("# threads: %d\n", num_threads);
  
  #endif
  /////////////////////////////////////////////////////////////////////
	
	// time span
	double t_start = 0.0;
	double t_end = 1.0e-3;
  	double h = 1.0e-6;
	
	
	/////////////////////////////////////////////////
	// arrays

	// size of data array in bytes
	size_t size = NUM * sizeof(Real) * NN;

	Real* y_host;
	y_host = (Real *) malloc (size);

	// pressure/volume arrays
	Real* pres_host;
	pres_host = (Real *) malloc (NUM * sizeof(Real));

	Real* rho_host = NULL;
	#ifdef CONV
	rho_host = (Real *) malloc (NUM * sizeof(Real));
	#endif
	
	#ifdef SAME_IC
		set_same_initial_conditions(NUM, y_host, pres_host, rho_host);
	#else
		FILE* fp = fopen ("ign_data.txt", "r");
		int buff_size = 1024;
		int retries = 0;
		//all lines should be the same size, so make sure the buffer is large enough
		for (retries = 0; retries < 5; retries++)
		{
			char buffer [buff_size];
			if (fgets (buffer, buff_size, fp) != NULL) {
				break;
			}
			rewind (fp);
			buff_size *= 2;
		}
		if (retries == 5)
		{
			printf("Could not parse ign_data.txt line with maximum buffer size of %d", buff_size);
			exit(-1);
		}

		//rewind and read
		rewind (fp);

		char buffer [buff_size];
		char* ptr, *eptr;
		Real res[NN + 1];
		// load temperature and mass fractions for all threads (cells)
		for (int i = 0; i < NUM; ++i) {
			// read line from data file
			if (fgets (buffer, buff_size, fp) == NULL) {
				printf("Error reading ign_data.txt, exiting...");
				exit(-1);
			}
			//read doubles from buffer
			ptr = buffer;
			for (int j = 0 ; j <= NN; j++) {
				#ifdef DOUBLE
					res[j] = strtod(ptr, &eptr);
				#else
					res[j] = strtof(ptr, &eptr);
				#endif
				ptr = eptr;
			}
			//put into y_host
			y_host[i] = res[0];
			pres_host[i] = res[1];
			for (int j = 2; j <= NN; j++)
				y_host[i + (j - 1) * NUM] = res[j];

			// if constant volume, calculate density
			#ifdef CONV
		    Real Yi[NSP];
		    Real Xi[NSP];
		    
		    for (int j = 0; j < NSP; ++j) {
		      Yi[j] = y_host[i + j * NUM];
		    }
		    
		    mass2mole (Yi, Xi);
		    rho_host[i] = getDensity (y_host[i], pres, Xi);
			#endif
		}
		fclose (fp);

		#ifdef SHUFFLE
		// now need to shuffle order
		struct timeval tv;
		gettimeofday(&tv, NULL);
		int usec = tv.tv_usec;
		srand48(usec);

		for (size_t i = NUM - 1; i > 0; i--) {
			size_t j = (unsigned int) (drand48() * (i + 1));

			for (size_t ind = 0; ind < NN; ++ind) {
				Real t = y_host[j + NUM * ind];
				y_host[j + NUM * ind] = y_host[i + NUM * ind];
				y_host[i + NUM * ind] = t;

				#ifdef CONP
				t = pres_host[j];
				pres_host[j] = pres_host[i];
				pres_host[i] = t;
				#else
				t = rho_host[j];
				rho_host[j] = rho_host[i];
				rho_host[i] = t;
				#endif
			}
		}
		#endif
	#endif
  
  
  // flag for ignition
  #ifdef IGN
  bool ign_flag = false;
  // ignition delay time, units [s]
  Real t_ign = ZERO;
  Real T0 = y_host[0];
  #endif
  
  #ifdef PRINT
  // file for data
  FILE *pFile;
  pFile = fopen("cpu.txt", "w");

  fprintf(pFile, "%e", t_0);
  for (int i = 0; i < NN; ++i) {
  	fprintf(pFile, "\t%e", y_host[NUM * i]);
  }
  fprintf(pFile, "\n");
  #endif
  
  int numSteps = 0;
  
	//////////////////////////////
	// start timer
	StartTimer();
	//////////////////////////////
  
  
	// set initial time
	Real t = t_start;
	Real t_next = t + h;
	
	/*
	printf("%18.15e", t);
	for (int i = 0; i < NN; ++i) {
		printf(" %18.15e", y_host[i]);
	}
	printf("\n");
  */
  
	// time loop
	while (t < t_end) {
		numSteps++;

		#if defined(CONP)
			// constant pressure case
			intDriver (NUM, t, t_next, pres_host, y_host);
		#elif defined(CONV)
			// constant volume case
			intDriver (NUM, t, t_next, rho_host, y_host);
		#endif

		t = t_next;
		t_next += h;

		printf("%.15le\t%.15le\n", t, y_host[0]);

		
		// check if within bounds
		if ((y_host[0] < ZERO) || (y_host[0] > 10000.0)) {
			printf("Error, out of bounds.\n");
			printf("Time: %e, ind %d val %e\n", t, 0, y_host[0]);
			return 1;
		}
		//#pragma unroll NSP
    /*
		for (int i = 1; i < NN; ++i)
		{
			if ((y_host[NUM * i] < -SMALL) || (y_host[NUM * i] > ONE)) {
				printf("Error, out of bounds.\n");
				printf("Time: %e, ind %d val %e\n", t, i, y_host[NUM * i]);
				return 1;
			}
		}
    */

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
	printf("Time: %e sec\n", runtime);
	runtime = runtime / ((Real)(numSteps));
	printf("Time per step: %e (s)\t%e (s/thread)\n", runtime, runtime/NUM);

	#ifdef IGN
	// if calculating ignition delay, print ign delay; units [s]
	//fprintf (pFile, "Ignition delay: %le\n", t_ign);
	printf ("Ignition delay: %le\n", t_ign);
	#endif

	//#define DEBUG
	#ifdef DEBUG
	for (int i = 0; i < NUM; ++i) {
		printf("%e ", y_host[i]);
	}
	printf("\n");
	#endif
  
  #ifdef PRINT
  fclose(pFile);
  #endif

  free(y_host);
  free(pres_host);
  #ifdef CONV
  free(rho_host);
  #endif
	
	return 0;
}
