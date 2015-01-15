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
//#include "exprb43.cuh"
#include "mass_mole.h"
#include "timer.h"

#include "rates.cuh"

#include "cuda_profiler_api.h"
#include "cudaProfiler.h"

// load same initial conditions for all threads
#define SAME_IC
#define REPEATS 100
#define BLOCK_SIZE 64
#define LOW_T 1000
#define HI_T 2000
#define GLOBAL_MEM

// shuffle initial conditions randomly
//#define SHUFFLE


static inline double getSign()
{
	return ((double)rand()/(double)RAND_MAX) > 0.5 ? 1.0 : -1.0;
}

static inline double getRand()
{
	return ((double)rand()/(double)RAND_MAX);// * getSign();
}

/////////////////////////////////////////////////////////////////////////////

bool errorCheck (cudaError_t status) {
    if (status != cudaSuccess) {
        printf ("%s\n", cudaGetErrorString(status));
        exit(-1);
        return false;
    }
    return true;
}

  /* Block size */

void populate(int NUM, int padded_size, Real pres, Real* y, Real* conc_arrays)
{
	// mass-averaged density
	Real rho;
	rho = (y[1] / 2.01594) + (y[2] / 1.00797) + (y[3] / 15.9994) + (y[4] / 31.9988)
	  + (y[5] / 17.00737) + (y[6] / 18.01534) + (y[7] / 33.00677) + (y[8] / 34.01474)
	  + (y[9] / 12.01115) + (y[10] / 13.01912) + (y[11] / 14.02709) + (y[12] / 14.02709)
	  + (y[13] / 15.03506) + (y[14] / 16.04303) + (y[15] / 28.01055) + (y[16] / 44.00995)
	  + (y[17] / 29.01852) + (y[18] / 30.02649) + (y[19] / 31.03446) + (y[20] / 31.03446)
	  + (y[21] / 32.04243) + (y[22] / 25.03027) + (y[23] / 26.03824) + (y[24] / 27.04621)
	  + (y[25] / 28.05418) + (y[26] / 29.06215) + (y[27] / 30.07012) + (y[28] / 41.02967)
	  + (y[29] / 42.03764) + (y[30] / 42.03764) + (y[31] / 14.0067) + (y[32] / 15.01467)
	  + (y[33] / 16.02264) + (y[34] / 17.03061) + (y[35] / 29.02137) + (y[36] / 30.0061)
	  + (y[37] / 46.0055) + (y[38] / 44.0128) + (y[39] / 31.01407) + (y[40] / 26.01785)
	  + (y[41] / 27.02582) + (y[42] / 28.03379) + (y[43] / 41.03252) + (y[44] / 43.02522)
	  + (y[45] / 43.02522) + (y[46] / 43.02522) + (y[47] / 42.01725) + (y[48] / 28.0134)
	  + (y[49] / 39.948) + (y[50] / 43.08924) + (y[51] / 44.09721) + (y[52] / 43.04561)
	  + (y[53] / 44.05358);
	rho = pres / (8.31451000e+07 * LOW_T * rho);

	// species molar concentrations
	Real conc1[53];
	conc1[0] = rho * y[1] / 2.01594;
	conc1[1] = rho * y[2] / 1.00797;
	conc1[2] = rho * y[3] / 15.9994;
	conc1[3] = rho * y[4] / 31.9988;
	conc1[4] = rho * y[5] / 17.00737;
	conc1[5] = rho * y[6] / 18.01534;
	conc1[6] = rho * y[7] / 33.00677;
	conc1[7] = rho * y[8] / 34.01474;
	conc1[8] = rho * y[9] / 12.01115;
	conc1[9] = rho * y[10] / 13.01912;
	conc1[10] = rho * y[11] / 14.02709;
	conc1[11] = rho * y[12] / 14.02709;
	conc1[12] = rho * y[13] / 15.03506;
	conc1[13] = rho * y[14] / 16.04303;
	conc1[14] = rho * y[15] / 28.01055;
	conc1[15] = rho * y[16] / 44.00995;
	conc1[16] = rho * y[17] / 29.01852;
	conc1[17] = rho * y[18] / 30.02649;
	conc1[18] = rho * y[19] / 31.03446;
	conc1[19] = rho * y[20] / 31.03446;
	conc1[20] = rho * y[21] / 32.04243;
	conc1[21] = rho * y[22] / 25.03027;
	conc1[22] = rho * y[23] / 26.03824;
	conc1[23] = rho * y[24] / 27.04621;
	conc1[24] = rho * y[25] / 28.05418;
	conc1[25] = rho * y[26] / 29.06215;
	conc1[26] = rho * y[27] / 30.07012;
	conc1[27] = rho * y[28] / 41.02967;
	conc1[28] = rho * y[29] / 42.03764;
	conc1[29] = rho * y[30] / 42.03764;
	conc1[30] = rho * y[31] / 14.0067;
	conc1[31] = rho * y[32] / 15.01467;
	conc1[32] = rho * y[33] / 16.02264;
	conc1[33] = rho * y[34] / 17.03061;
	conc1[34] = rho * y[35] / 29.02137;
	conc1[35] = rho * y[36] / 30.0061;
	conc1[36] = rho * y[37] / 46.0055;
	conc1[37] = rho * y[38] / 44.0128;
	conc1[38] = rho * y[39] / 31.01407;
	conc1[39] = rho * y[40] / 26.01785;
	conc1[40] = rho * y[41] / 27.02582;
	conc1[41] = rho * y[42] / 28.03379;
	conc1[42] = rho * y[43] / 41.03252;
	conc1[43] = rho * y[44] / 43.02522;
	conc1[44] = rho * y[45] / 43.02522;
	conc1[45] = rho * y[46] / 43.02522;
	conc1[46] = rho * y[47] / 42.01725;
	conc1[47] = rho * y[48] / 28.0134;
	conc1[48] = rho * y[49] / 39.948;
	conc1[49] = rho * y[50] / 43.08924;
	conc1[50] = rho * y[51] / 44.09721;
	conc1[51] = rho * y[52] / 43.04561;
	conc1[52] = rho * y[53] / 44.05358;

	rho = pres / (8.31451000e+07 * HI_T * rho);

	Real conc2[53];
	conc2[0] = rho * y[1] / 2.01594;
	conc2[1] = rho * y[2] / 1.00797;
	conc2[2] = rho * y[3] / 15.9994;
	conc2[3] = rho * y[4] / 31.9988;
	conc2[4] = rho * y[5] / 17.00737;
	conc2[5] = rho * y[6] / 18.01534;
	conc2[6] = rho * y[7] / 33.00677;
	conc2[7] = rho * y[8] / 34.01474;
	conc2[8] = rho * y[9] / 12.01115;
	conc2[9] = rho * y[10] / 13.01912;
	conc2[10] = rho * y[11] / 14.02709;
	conc2[11] = rho * y[12] / 14.02709;
	conc2[12] = rho * y[13] / 15.03506;
	conc2[13] = rho * y[14] / 16.04303;
	conc2[14] = rho * y[15] / 28.01055;
	conc2[15] = rho * y[16] / 44.00995;
	conc2[16] = rho * y[17] / 29.01852;
	conc2[17] = rho * y[18] / 30.02649;
	conc2[18] = rho * y[19] / 31.03446;
	conc2[19] = rho * y[20] / 31.03446;
	conc2[20] = rho * y[21] / 32.04243;
	conc2[21] = rho * y[22] / 25.03027;
	conc2[22] = rho * y[23] / 26.03824;
	conc2[23] = rho * y[24] / 27.04621;
	conc2[24] = rho * y[25] / 28.05418;
	conc2[25] = rho * y[26] / 29.06215;
	conc2[26] = rho * y[27] / 30.07012;
	conc2[27] = rho * y[28] / 41.02967;
	conc2[28] = rho * y[29] / 42.03764;
	conc2[29] = rho * y[30] / 42.03764;
	conc2[30] = rho * y[31] / 14.0067;
	conc2[31] = rho * y[32] / 15.01467;
	conc2[32] = rho * y[33] / 16.02264;
	conc2[33] = rho * y[34] / 17.03061;
	conc2[34] = rho * y[35] / 29.02137;
	conc2[35] = rho * y[36] / 30.0061;
	conc2[36] = rho * y[37] / 46.0055;
	conc2[37] = rho * y[38] / 44.0128;
	conc2[38] = rho * y[39] / 31.01407;
	conc2[39] = rho * y[40] / 26.01785;
	conc2[40] = rho * y[41] / 27.02582;
	conc2[41] = rho * y[42] / 28.03379;
	conc2[42] = rho * y[43] / 41.03252;
	conc2[43] = rho * y[44] / 43.02522;
	conc2[44] = rho * y[45] / 43.02522;
	conc2[45] = rho * y[46] / 43.02522;
	conc2[46] = rho * y[47] / 42.01725;
	conc2[47] = rho * y[48] / 28.0134;
	conc2[48] = rho * y[49] / 39.948;
	conc2[49] = rho * y[50] / 43.08924;
	conc2[50] = rho * y[51] / 44.09721;
	conc2[51] = rho * y[52] / 43.04561;
	conc2[52] = rho * y[53] / 44.05358;

  	for(int j = 0; j < NSP; j++)
  	{
		for (int i = 0; i < NUM; i++)
		{
			if (i % 2 == 0)
				conc_arrays[i + j * padded_size] = conc1[j];
			else
				conc_arrays[i + j * padded_size] = conc2[j];
  		}
	}
}

#define tid (threadIdx.x + (blockDim.x * blockIdx.x))
__global__
void rxnrates_driver (const int NUM, const Real* T_global, const Real* conc, Real* fwd_rates, Real* rev_rates) {
#ifndef GLOBAL_MEM
	Real local_fwd_rates[FWD_RATES];
	Real local_rev_rates[REV_RATES];
	// local array with initial values
	Real local_conc[NSP];

	// load local array with initial values from global array
	#pragma unroll
	for (int i = 0; i < NSP; i++)
	{
		local_conc[i] = conc[tid + i * NUM];
	}
	#pragma unroll
	for (int repeat = 0; repeat < REPEATS; repeat++)
	{
		if (tid < NUM)
		{
			eval_rxn_rates (T_global[tid], local_conc, local_fwd_rates, local_rev_rates);
		}
	}
	//copy back
	for (int i = 0; i < FWD_RATES; i++)
	{
		fwd_rates[tid + i * NUM] = local_fwd_rates[i];
	}

	//copy back
	for (int i = 0; i < REV_RATES; i++)
	{
		rev_rates[tid + i * NUM] = local_rev_rates[i];
	}
#else
	#pragma unroll
	for (int repeat = 0; repeat < REPEATS; repeat++)
	{
		if (tid < NUM)
		{
			eval_rxn_rates (T_global[tid], conc, fwd_rates, rev_rates);
		}
	}
#endif
}

__global__
void presmod_driver (const int NUM, const Real* T_global, const Real* pr_global, const Real* conc, Real* pres_mod) {
#ifndef GLOBAL_MEM
	Real local_pres_mod[PRES_MOD_RATES];
	// local array with initial values
	Real local_conc[NSP];

	// load local array with initial values from global array
	#pragma unroll
	for (int i = 0; i < NSP; i++)
	{
		local_conc[i] = conc[tid + i * NUM];
	}

	#pragma unroll
	for (int repeat = 0; repeat < REPEATS; repeat++)
	{
		if (tid < NUM)
		{
			get_rxn_pres_mod (T_global[tid], pr_global[tid], local_conc, local_pres_mod);
		}
	}
	//copy back
	for (int i = 0; i < PRES_MOD_RATES; i++)
	{
		pres_mod[tid + i * NUM] = local_pres_mod[i];
	}
#else
	#pragma unroll
	for (int repeat = 0; repeat < REPEATS; repeat++)
	{
		if (tid < NUM)
		{
			get_rxn_pres_mod (T_global[tid], pr_global[tid], conc, pres_mod);
		}
	}
#endif
}

__global__
void specrates_driver (const int NUM, const Real* fwd_rates, const Real* rev_rates, const Real* pres_mod, Real* spec_rates) {
#ifndef GLOBAL_MEM
	Real local_fwd_rates[FWD_RATES];
	Real local_rev_rates[REV_RATES];
	Real local_pres_mod[PRES_MOD_RATES];
	// local array with initial values
	Real local_spec_rates[NSP];

	// load local array with initial values from global array
	//copy in
	for (int i = 0; i < FWD_RATES; i++)
	{
		local_fwd_rates[i] = fwd_rates[tid + i * NUM];
	}
	//copy in
	for (int i = 0; i < REV_RATES; i++)
	{
		local_rev_rates[i] = rev_rates[tid + i * NUM];
	}
	//copy in
	for (int i = 0; i < PRES_MOD_RATES; i++)
	{
		local_pres_mod[i] = pres_mod[tid + i * NUM];
	}
	#pragma unroll
	for (int repeat = 0; repeat < REPEATS; repeat++)
	{
		if (tid < NUM)
		{
			eval_spec_rates (local_fwd_rates, local_rev_rates, local_pres_mod, local_spec_rates);
		}
	}
	//copy back
	for (int i = 0; i < NSP; i++)
	{
		spec_rates[tid + i * NUM] = local_spec_rates[i];
	}
#else
	#pragma unroll
	for (int repeat = 0; repeat < REPEATS; repeat++)
	{
		if (tid < NUM)
		{
			eval_spec_rates (fwd_rates, rev_rates, pres_mod, spec_rates);
		}
	}
#endif
}

/////////////////////////////////////////////////////////////////////////////
int main (int argc, char *argv[]) {

  int NUM = 1024;
  srand(1);

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

	int padded_size = max(NUM, g_num * BLOCK_SIZE);

	// size of data array in bytes
	size_t size = padded_size * sizeof(Real);

	//temperature array
	Real* T_host = (Real *) malloc (size);
	// pressure/volume arrays
	Real* pres_host = (Real *) malloc (size);
	// conc array
	Real* conc_host = (Real *) malloc (NSP * size);

#ifdef DEBUG
	//define print arrays
	Real* fwd_host = (Real*)malloc(FWD_RATES * size);
	Real* rev_host = (Real*)malloc(FWD_RATES * size);
	Real* pres_mod_host = (Real*)malloc(FWD_RATES * size);
	Real* spec_host = (Real*)malloc(FWD_RATES * size);
#endif
  
  	Real sum = 0;
  	Real y_dummy[NN];
  	//come up with a random state
  	for (int i = 1; i < NN; i++)
  	{
  		Real r = 1;
  		sum += r * r;
  		y_dummy[i] = r;
  	}
  	//normalize
  	sum = sqrt(sum);
  	 for (int i = 1; i < NN; i++)
  	{
  		y_dummy[i] /= sum;
  	}
  	y_dummy[0] = LOW_T;
  	Real P = 101325;
  	for (int j = 0; j < NUM; j++)
  	{
  		pres_host[j] = P;
  		if (j % 2 == 0)
  			T_host[j] = LOW_T;
  		else
  			T_host[j] = HI_T;
  	}

  	populate(NUM, padded_size, P, y_dummy, conc_host);

  	//bump up shared mem bank size
	errorCheck(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));
	//and L1 size
	errorCheck(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));

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
  
  	// initialize GPU
  
	// Allocate device memory
	Real* T_device;
	errorCheck(cudaMalloc (&T_device, size));
	// transfer memory to GPU
	errorCheck(cudaMemcpy (T_device, T_host, size, cudaMemcpyHostToDevice));
  
	// device array for pressure or density
	Real* pr_device;
	errorCheck(cudaMalloc (&pr_device, size));
	errorCheck(cudaMemcpy (pr_device, pres_host, size, cudaMemcpyHostToDevice));

	Real* conc_device;
	// device array for concentrations
	errorCheck(cudaMalloc (&conc_device, NSP * size));
	errorCheck(cudaMemcpy (conc_device, conc_host, NSP * size, cudaMemcpyHostToDevice));

  	//allocate fwd and reverse rate arrays
  	Real* fwd_rates;
  	errorCheck(cudaMalloc (&fwd_rates, FWD_RATES * size));

  	Real* rev_rates;
  	errorCheck(cudaMalloc (&rev_rates, REV_RATES * size));

  	//pres_mod
  	Real* pres_mod;
  	errorCheck(cudaMalloc (&pres_mod, PRES_MOD_RATES * size));

  	//finally species rates
  	Real* spec_rates;
  	errorCheck(cudaMalloc (&spec_rates, NSP * size));
	cuProfilerStart();
	//////////////////////////////
	rxnrates_driver <<<dimGrid, dimBlock>>> (NUM, T_device, conc_device, fwd_rates, rev_rates);
	cuProfilerStop();
#ifdef DEBUG
	//copy back and print
	errorCheck(cudaMemcpy (fwd_host, fwd_rates, FWD_RATES * size, cudaMemcpyDeviceToHost));
	printf("FWD_RATES:\n");
	for (int i = 0; i < FWD_RATES; i++)
		printf("%e\n", fwd_host[i * padded_size]);

	errorCheck(cudaMemcpy (rev_host, rev_rates, REV_RATES * size, cudaMemcpyDeviceToHost));
	printf("REV_RATES:\n");
	for (int i = 0; i < REV_RATES; i++)
		printf("%e\n", rev_host[i * padded_size]);
#endif
	/////////////////////////////////

	cuProfilerStart();
	//////////////////////////////
	presmod_driver <<<dimGrid, dimBlock>>> (NUM, T_device, pr_device, conc_device, pres_mod);
	cuProfilerStop();
#ifdef DEBUG
	//copy back and print
	errorCheck(cudaMemcpy (pres_mod_host, pres_mod, PRES_MOD_RATES * size, cudaMemcpyDeviceToHost));
	printf("PRES_MOD:\n");
	for (int i = 0; i < PRES_MOD_RATES; i++)
		printf("%e\n", pres_mod_host[i * padded_size]);
#endif
	/////////////////////////////////

	cuProfilerStart();
	//////////////////////////////
	specrates_driver <<<dimGrid, dimBlock>>> (NUM, fwd_rates, rev_rates, pres_mod, spec_rates);
	cuProfilerStop();
#ifdef DEBUG
	//copy back and print
	errorCheck(cudaMemcpy (spec_host, spec_rates, NSP * size, cudaMemcpyDeviceToHost));
	printf("SPEC_RATES:\n");
	for (int i = 0; i < NSP; i++)
		printf("%e\n", spec_host[i * padded_size]);
#endif
	/////////////////////////////////
  
  free (T_host);
  free (pres_host);
  free (conc_host);
#ifdef DEBUG
  //define print arrays
  free(fwd_host);
  free(rev_host);
  free(pres_mod_host);
  free(spec_host);
#endif
  
  cudaFree (T_device);
  cudaFree (pr_device);
  cudaFree (conc_device);
  cudaFree (fwd_rates);
  cudaFree (rev_rates);
  cudaFree (pres_mod);
  cudaFree (spec_rates);
  
  errorCheck(cudaDeviceReset());
	
	return 0;
}
