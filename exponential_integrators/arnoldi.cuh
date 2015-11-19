/* krylov.h
 * Implementation of the arnoldi iteration methods
 * \file arnoldi.h
 *
 * \author Nicholas Curtis
 * \date 03/09/2015
 *
 */

#ifndef ARNOLDI_CUH
#define ARNOLDI_CUH

#include <string.h>

#include "header.cuh"
#include "phiAHessenberg.cuh"
#include "exponential_linear_algebra.cuh"
#include "sparse_multiplier.cuh"
#include "solver_options.h"
#include "solver_props.cuh" 

__constant__ int index_list[23] = {1, 2, 3, 4, 5, 6, 7, 9, 11, 14, 17, 21, 27, 34, 42, 53, 67, 84, 106, 133, 167, 211, 265};

///////////////////////////////////////////////////////////////////////////////

/** Matrix-vector multiplication of a matrix sized MxM and a vector Mx1
 * 
 * Performs inline matrix-vector multiplication (with unrolled loops) 
 * 
 * \param[in, out]		m 		in - the starting size of the matrix, out - the ending size of the matrix
 * \param[in]			scale	the value to scale the timestep by
 * \param[in]			p		the order of the maximum phi function needed
 * \param[in]			h		the timestep
 * \param[in]			A 		the jacobian matrix
 * \param[in]  			v 		the vector to use for the krylov subspace
 * \param[in] 			sc 		the error scaling vector
 * \param[out] 			beta 	the norm of the v vector
 * \param[out]			Vm 		the arnoldi basis matrix
 * \param[out]			Hm 		the constructed Hm matrix, used in actual expoentials
 * \param[out] 			phiHm   the exponential matrix computed from h * scale * Hm
 */
__device__
int arnoldi(int* m, const double scale, const int p, const double h, const double* A, const double* v, const double* sc, double* beta, double* Vm, double* Hm, double* phiHm)
{
	//the temporary work array
	double w[NSP];

	//first place A*fy in the Vm matrix
	*beta = normalize(v, Vm);

	double store = 0;
	int index = 0;
	int j = 0;
	double err = 2.0;

	while (err >= 1.0 && j + p < M_MAX)
	{
		for (; j < index_list[index]; j++)
		{
			sparse_multiplier(A, &Vm[j * NSP], w);
			for (int i = 0; i <= j; i++)
			{
				Hm[j * STRIDE + i] = dotproduct(w, &Vm[i * NSP]);
				scale_subtract(Hm[j * STRIDE + i], &Vm[i * NSP], w);
			}
			Hm[j * STRIDE + j + 1] = two_norm(w);
			if (fabs(Hm[j * STRIDE + j + 1]) < ATOL)
			{
				//happy breakdown
				*m = j;
				break;
			}
			scale_mult(1.0 / Hm[j * STRIDE + j + 1], w, &Vm[(j + 1) * NSP]);
		}
		*m = index_list[index++];
		//resize Hm to be mxm, and store Hm(m, m + 1) for later
		store = Hm[(*m - 1) * STRIDE + *m];
		Hm[(*m - 1) * STRIDE + *m] = 0.0;

		//0. fill potentially non-empty memory first
		for (int i = 0; i < (*m + 1); ++i)
			Hm[*m * STRIDE + i] = 0;

		//get error
		//1. Construct augmented Hm (fill in identity matrix)
		Hm[(*m) * STRIDE] = 1.0;
		#pragma unroll 1
		for (int i = 1; i < p; i++)
		{
			//0. fill potentially non-empty memory first
			for (int j = 0; j < (*m + i + 1); ++j)
				Hm[(*m + i) * STRIDE + j] = 0;
			Hm[(*m + i) * STRIDE + (*m + i - 1)] = 1.0;
		}

#ifdef RB43
		//2. Get phiHm
		expAc_variable (*m + p, Hm, h * scale, phiHm);
#elif EXP4
		//2. Get phiHm
		phiAc_variable (*m + p, Hm, h * scale, phiHm);
#endif

		//3. Get error

		#ifdef USE_SMOOTHED_ERROR
			if (*m > 4)
			{
				//use the modified err from Hochbruck et al. 

				//setup I - h*Hm
				double* working = (double*)malloc((*m) * (*m) * sizeof(double));
				#pragma unroll 1
				for (int ind1 = 0; ind1 < *m; ind1++)
				{
					#pragma unroll 1
					for (int ind2 = 0; ind2 < *m; ind2++)
					{
						if (ind1 == ind2)
						{
							working[ind1 * (*m) + ind2] = 1.0 - h * scale * Hm[ind1 * STRIDE + ind2];
						}
						else
						{
							working[ind1 * (*m) + ind2] = -h * scale * Hm[ind1 * STRIDE + ind2];
						}
					}
				}
				getInverseHessenberg(*m, working);
				//get the value for the err (dot product of mth row of working and 1'st col of Hm)
				double val = 0;
				#pragma unroll 1
				for (int ind1 = 0; ind1 < *m; ind1++)
				{
					val += working[(*m) * ind1 + (*m - 1)] * Hm[ind1];
				}
				err = h * (*beta) * fabs(store * val) * sc_norm(&Vm[(*m) * NSP], sc);

				free(working);
			}
			else
			{
				err = h * (*beta) * fabs(store * phiHm[(*m) * STRIDE + (*m) - 1]) * sc_norm(&Vm[(*m) * NSP], sc);
			}
		#else
			err = h * (*beta) * fabs(store * phiHm[(*m) * STRIDE + (*m) - 1]) * sc_norm(&Vm[(*m) * NSP], sc);
		#endif

		//restore Hm(m, m + 1)
		Hm[(*m - 1) * STRIDE + *m] = store;

		//restore real Hm (NOTE: untested in RB43)
		Hm[(*m) * STRIDE] = 0.0;
	}

	return j;
}

#endif