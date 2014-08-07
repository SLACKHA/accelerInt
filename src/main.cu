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

#include "head.h"
extern "C" {
#include "cf.h"
}
#include "exp4.cuh"
#include "mass_mole.h"
#include "timer.h"

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
		y_local[0] = y_global[tid];
		y_local[1] = y_global[tid + NUM];
		y_local[2] = y_global[tid + NUM * 2];
		y_local[3] = y_global[tid + NUM * 3];
		y_local[4] = y_global[tid + NUM * 4];
		y_local[5] = y_global[tid + NUM * 5];
		y_local[6] = y_global[tid + NUM * 6];
		y_local[7] = y_global[tid + NUM * 7];
		y_local[8] = y_global[tid + NUM * 8];
		y_local[9] = y_global[tid + NUM * 9];
		y_local[10] = y_global[tid + NUM * 10];
		y_local[11] = y_global[tid + NUM * 11];
		y_local[12] = y_global[tid + NUM * 12];
		y_local[13] = y_global[tid + NUM * 13];

		// call integrator for one time step
		exp4_int (t, t_end, pr_local, y_local);

		// update global array with integrated values
		y_global[tid] = y_local[0];
		y_global[tid + NUM] = y_local[1];
		y_global[tid + NUM * 2] = y_local[2];
		y_global[tid + NUM * 3] = y_local[3];
		y_global[tid + NUM * 4] = y_local[4];
		y_global[tid + NUM * 5] = y_local[5];
		y_global[tid + NUM * 6] = y_local[6];
		y_global[tid + NUM * 7] = y_local[7];
		y_global[tid + NUM * 8] = y_local[8];
		y_global[tid + NUM * 9] = y_local[9];
		y_global[tid + NUM * 10] = y_local[10];
		y_global[tid + NUM * 11] = y_local[11];
		y_global[tid + NUM * 12] = y_local[12];
		y_global[tid + NUM * 13] = y_local[13];

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
	
	// get poles and residues for rational approximant to matrix exponential
	double *poles_r = (double*) calloc (N_RA, sizeof(double));
	double *poles_i = (double*) calloc (N_RA, sizeof(double));
	double *res_r = (double*) calloc (N_RA, sizeof(double));
	double *res_i = (double*) calloc (N_RA, sizeof(double));
	
	cf (N_RA, poles_r, poles_i, res_r, res_i);
  
  cuDoubleComplex *polesHost = (cuDoubleComplex*) calloc (N_RA, sizeof(cuDoubleComplex));
  cuDoubleComplex *resHost = (cuDoubleComplex*) calloc (N_RA, sizeof(cuDoubleComplex));
	
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

	#ifdef CONV
	Real* rho_host;
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
	// load same ICs for all threads

	// initial mole fractions
	Real Xi[NSP];
	for (int j = 0; j < NSP; ++ j) {
		Xi[j] = 0.0;
	}

	//
	// set initial mole fractions here
	//
	
	// h2
	Xi[1] = 2.0;
	// o2
	Xi[5] = 1.0;
	// n2
	Xi[8] = 3.76;
	
	// normalize mole fractions to sum to 1
	Real Xsum = 0.0;
	for (int j = 0; j < NSP; ++ j) {
		Xsum += Xi[j];
	}
	for (int j = 0; j < NSP; ++ j) {
		Xi[j] /= Xsum;
	}

	// initial mass fractions
	Real Yi[NSP];
	mole2mass ( Xi, Yi );

	// set initial pressure, units [dyn/cm^2]
	// 1 atm = 1.01325e6 dyn/cm^2
	Real pres = 1.01325e6;

	// set initial temperature, units [K]
	Real T0 = 1600.0;

	// load temperature and mass fractions for all threads (cells)
	for (int i = 0; i < NUM; ++i) {
		y_host[i] = T0;

		// loop through species
		for (int j = 1; j < NN; ++j) {
			y_host[i + NUM * j] = Yi[j - 1];
		}
	}

	#ifdef CONV
	// if constant volume, calculate density
	Real rho;
  rho = getDensity (T0, pres, Xi);
	#endif

	for (int i = 0; i < NUM; ++i) {
		#ifdef CONV
		// density
		rho_host[i] = rho;
		#else
		// pressure
		pres_host[i] = pres;
		#endif
	}
/////////////////////////////////////////////////////////
	#else
/////////////////////////////////////////////////////////
	// load different ICs for all threads

	// load ICs from file
	FILE* fp = fopen ("ign_data.txt", "r");
	char buffer[1024];

	// load temperature and mass fractions for all threads (cells)
	for (int i = 0; i < NUM; ++i) {
		// read line from data file
		if (fgets (buffer, 1024, fp) == NULL) {
			rewind (fp);
			fgets (buffer, 1024, fp);
		}
		sscanf (buffer, "%le %le %le %le %le %le %le %le %le %le %le %le %le %le", \
		&y_host[i], &pres_host[i], &y_host[i + NUM], &y_host[i + NUM*2], &y_host[i + NUM*3], \
		&y_host[i + NUM*4], &y_host[i + NUM*5], &y_host[i + NUM*6], &y_host[i + NUM*7], \
		&y_host[i + NUM*8], &y_host[i + NUM*9], &y_host[i + NUM*10], &y_host[i + NUM*12], \
		&y_host[i + NUM*13]);

		// if constant volume, calculate density
		#ifdef CONV
    Real Yi[NSP];
    Real Xi[NSP];
    
    for (int j = 0; j < NSP; ++j) {
      Yi[j] = y_host[i + NUM];
    }
    
    mass2mole (Yi, Xi);
    rho_host[i] = getDensity (T0, pres, Xi);
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
		if (sscanf(argv[1], "%i", &id) == 1 && (id >= 0) && (id < num_devices)) {
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
  status = cudaMemcpyToSymbol (poles, &polesHost, N_RA * sizeof(cuDoubleComplex), 0, cudaMemcpyHostToDevice);
  errorCheck (status);
  
  status = cudaMemcpyToSymbol (res, &resHost, N_RA * sizeof(cuDoubleComplex), 0, cudaMemcpyHostToDevice);
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
  
  free (polesHost);
  free (resHost);
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
