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

/** Include CUDA libraries. */
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <cuComplex.h>

#include "header.h"
extern "C" {
#include "cf.h"
}
#include "exp4.cuh"
#include "mass_mole.h"
#include "timer.h"
#include "mechanism.h"

#ifdef DEBUG
//NAN check
#include <fenv.h> 
#endif


#ifdef DOUBLE
__device__ __constant__ cuDoubleComplex poles[N_RA];
__device__ __constant__ cuDoubleComplex res[N_RA];
#else
__device__ __constant__ cuFloatComplex poles[N_RA];
__device__ __constant__ cuFloatComplex res[N_RA];
#endif

// load same initial conditions for all threads
#define SAME_IC

// shuffle initial conditions randomly
//#define SHUFFLE

/////////////////////////////////////////////////////////////////////////////

bool errorCheck (cudaError_t status) {
    if (status != cudaSuccess) {
        printf ("%s\n", cudaGetErrorString(status));
        return false;
    }
    return true;
}

/////////////////////////////////////////////////////////////////////////////

__global__
void intDriver (const int NUM, const Real t, const Real t_end, 
                const Real* pr_global, Real* y_global) {

	const int tid = threadIdx.x + (blockDim.x * blockIdx.x);
  
  if (tid < NUM) {

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

	}

} // end intDriver

//////////////////////////////////////////////////////////////////////////////

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
	
	cf (N_RA, poles_r, poles_i, res_r, res_i);
  
 	cuDoubleComplex polesHost[N_RA];
 	cuDoubleComplex resHost[N_RA];
	
	for (int i = 0; i < N_RA; ++i) {
		polesHost[i] = make_cuDoubleComplex(poles_r[i], poles_i[i]);
		resHost[i] = make_cuDoubleComplex(res_r[i], res_i[i]);
	}
	
	// free memory
	free (poles_r);
	free (poles_i);
	free (res_r);
	free (res_i);
  
  /** Number of independent systems */
  int NUM = 1;
  
  // check for problem size given as command line option
  if (argc > 1) {
    int problemsize = NUM;
    if (sscanf(argv[1], "%i", &problemsize) != 1 || (problemsize <= 0)) {
      printf("Error: Problem size not in correct range\n");
      printf("Provide number greater than 0\n");
      exit(1);
    }
    NUM = problemsize;
  }
  
  /* Block size */
  int BLOCK_SIZE = 64;
  if (NUM < 8388608) {
    BLOCK_SIZE = 128;
  } else if (NUM < 16777216) {
    BLOCK_SIZE = 256;
  } else {
    BLOCK_SIZE = 512;
  }
    
  // print number of threads and block size
  printf ("# threads: %d \t block size: %d\n", NUM, BLOCK_SIZE);
	
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
	//////////////////////////////////////////////////


	// species indices:
	// 0 H
	// 1 H2
	// 2 O
	// 3 OH
	// 4 H2O
	// 5 O2
	// 6 HO2
	// 7 H2O2
	// 8 N2
	// 9 AR
	// 10 HE
	// 11 CO
	// 12 CO2

/////////////////////////////////////////////////////////////////////////////
  
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
/////////////////////////////////////////////////////////////////////////////
  
  #ifdef PRINT
  // file for data
  FILE *pFile;
  pFile = fopen("gpu.txt", "w");

  fprintf(pFile, "%e", t_0);
  for (int i = 0; i < NN; ++i) {
  	fprintf(pFile, "\t%e", y_host[NUM * i]);
  }
  fprintf(pFile, "\n");
  #endif
  
  
	// set & initialize device using command line argument (if any)
	cudaDeviceProp devProp;
	if (argc <= 2) {
		// default device id is 0
		checkCudaErrors (cudaSetDevice (0) );
		checkCudaErrors (cudaGetDeviceProperties(&devProp, 0));
	} else {
		// use second argument for number

		// get number of devices
		int num_devices;
		cudaGetDeviceCount(&num_devices);

		int id = 0;
		if (sscanf(argv[2], "%i", &id) == 1 && (id >= 0) && (id < num_devices)) {
			checkCudaErrors (cudaSetDevice (id) );
		} else {
			// not in range, error
			printf("Error: GPU device number not in correct range\n");
			printf("Provide number between 0 and %i\n", num_devices - 1);
			exit(1);
		}
		checkCudaErrors (cudaGetDeviceProperties(&devProp, id));
	}
  
  
	//////////////////////////////
	// start timer
	StartTimer();
	//////////////////////////////
  
  // initialize GPU

	// block and grid dimensions
	dim3 dimBlock ( BLOCK_SIZE, 1 );
	#ifdef QUEUE
	dim3 dimGrid ( numSM, 1 );
	#else
	int g_num = NUM / BLOCK_SIZE;
	if (g_num == 0)
		g_num = 1;
	dim3 dimGrid ( g_num, 1 );
	#endif
  
  cudaError_t status = cudaSuccess;
  
	// Allocate device memory
	Real* y_device;
	cudaMalloc ((void**) &y_device, size);
  
	// device array for pressure or density
	Real* pr_device;
	cudaMalloc ((void**) &pr_device, NUM * sizeof(Real));
	#ifdef CONP
	status = cudaMemcpy (pr_device, pres_host, NUM * sizeof(Real), cudaMemcpyHostToDevice);
	#else
	status = cudaMemcpy (pr_device, rho_host, NUM * sizeof(Real), cudaMemcpyHostToDevice);
	#endif
  
  errorCheck (status);
  
  // copy poles and residuals
  status = cudaMemcpyToSymbol (poles, polesHost, N_RA * sizeof(cuDoubleComplex), 0, cudaMemcpyHostToDevice);
  errorCheck (status);
  
  status = cudaMemcpyToSymbol (res, resHost, N_RA * sizeof(cuDoubleComplex), 0, cudaMemcpyHostToDevice);
  errorCheck (status);
  
	// set initial time
	Real t = t_start;
	Real t_next = t + h;
  int numSteps = 0;
    
	// time integration loop
	while (t < t_end) {
    
		numSteps++;
    
		// transfer memory to GPU
		status = cudaMemcpy (y_device, y_host, size, cudaMemcpyHostToDevice);
    errorCheck (status);

		#if defined(CONP)
			// constant pressure case
			intDriver <<<dimGrid, dimBlock>>> (NUM, t, t_next, pr_device, y_device);
		#elif defined(CONV)
			// constant volume case
			intDriver <<<dimGrid, dimBlock>>> (NUM, t, t_next, pr_device, y_device);
		#endif

		t = t_next;
		t_next += h;
    
		// transfer memory back to CPU
		status = cudaMemcpy (y_host, y_device, size, cudaMemcpyDeviceToHost);
    errorCheck (status);

		printf("%le\t%le\n", t, y_host[0]);

		
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
    
	}

	/////////////////////////////////
	// end timer
	double runtime = GetTimer();
	/////////////////////////////////
	
  
	runtime /= 1000.0;
	printf ("Time: %e sec\n", runtime);
	runtime = runtime / ((Real)(numSteps));
	printf ("Time per step: %e (s)\t%e (s/thread)\n", runtime, runtime/NUM);

	//#define DEBUG
	#ifdef DEBUG
	for (int i = 0; i < NUM; ++i) {
		printf ("%e ", y_host[i]);
	}
	printf ("\n");
	#endif
  
  #ifdef PRINT
  fclose (pFile);
  #endif
  
  //free (polesHost);
  //free (resHost);
  free (y_host);
  free (pres_host);
  #ifdef CONV
  free (rho_host);
  #endif
  
  cudaFree (y_device);
  cudaFree (pr_device);
  
  status = cudaDeviceReset();
  errorCheck (status);
	
	return 0;
}
