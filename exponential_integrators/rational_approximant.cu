/**
* \file rational_approximant.cu
* \brief The generic initialization file for poles/hosts for RA based evaulation of the matrix exponential
*
* \author Nicholas Curtis
* \date 03/09/2015
*
* Contains initialization and declaration of RA
*/

//cf
#include <cuComplex.h>
#include "header.cuh"
extern "C" {
#include "cf.h"
}
#include "solver_options.cuh"
#include "gpu_macros.cuh"

__device__ __constant__ cuDoubleComplex poles[N_RA];
__device__ __constant__ cuDoubleComplex res[N_RA];

/**
* \brief get poles and residues for rational approximant to matrix exponential
*/
void find_poles_and_residuals()
{
	// get poles and residues for rational approximant to matrix exponential
    double *poles_r = (double *) calloc (N_RA, sizeof(double));
    double *poles_i = (double *) calloc (N_RA, sizeof(double));
    double *res_r = (double *) calloc (N_RA, sizeof(double));
    double *res_i = (double *) calloc (N_RA, sizeof(double));

    cf (N_RA, poles_r, poles_i, res_r, res_i);

    cuDoubleComplex polesHost[N_RA];
    cuDoubleComplex resHost[N_RA];

    for (int i = 0; i < N_RA; ++i)
    {
        polesHost[i] = make_cuDoubleComplex(poles_r[i], poles_i[i]);
        resHost[i] = make_cuDoubleComplex(res_r[i], res_i[i]);
    }

    // free memory
    free (poles_r);
    free (poles_i);
    free (res_r);
    free (res_i);

    //copy to GPU memory
    cudaErrorCheck( cudaMemcpyToSymbol (poles, polesHost, N_RA * sizeof(cuDoubleComplex), 0, cudaMemcpyHostToDevice) );
    cudaErrorCheck( cudaMemcpyToSymbol (res, resHost, N_RA * sizeof(cuDoubleComplex), 0, cudaMemcpyHostToDevice) );
}