/* arnoli.cuh
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
int arnoldi(int* m, const double scale,
			const int p, const double h,
			const double* __restrict__ A,
			const solver_memory* __restrict__ solver,
			const double* __restrict__ v, double* __restrict__ beta,
			double * __restrict__ work,
			cuDoubleComplex* __restrict__ work2)
{
	const double* __restrict__ sc = solver->sc;
	double* __restrict__ Vm = solver->Vm;
	double* __restrict__ Hm = solver->Hm;
	double* __restrict__ phiHm = solver->phiHm;

	//first place A*fy in the Vm matrix
	*beta = normalize(v, Vm);

	double store = 0;
	int index = 0;
	int j = 0;
	double err = 2.0;
	int info = 0;

	while (err >= 1.0 && j + p < M_MAX)
	{
		for (; j < index_list[index]; j++)
		{
			sparse_multiplier(A, &Vm[GRID_DIM * (j * NSP)], work);
			for (int i = 0; i <= j; i++)
			{
				Hm[INDEX(j * STRIDE + i)] = dotproduct(work, &Vm[GRID_DIM * (i * NSP)]);
				scale_subtract(Hm[INDEX(j * STRIDE + i)], &Vm[GRID_DIM * (i * NSP)], work);
			}
			Hm[INDEX(j * STRIDE + j + 1)] = two_norm(work);
			if (fabs(Hm[INDEX(j * STRIDE + j + 1)]) < ATOL)
			{
				//happy breakdown
				*m = j;
				break;
			}
			scale_mult(1.0 / Hm[INDEX(j * STRIDE + j + 1)], work, &Vm[GRID_DIM * ((j + 1) * NSP)]);
		}
		*m = index_list[index++];
		//resize Hm to be mxm, and store Hm(m, m + 1) for later
		store = Hm[INDEX((*m - 1) * STRIDE + *m)];
		Hm[INDEX((*m - 1) * STRIDE + *m)] = 0.0;

		//0. fill potentially non-empty memory first
		for (int i = 0; i < (*m + 1); ++i)
			Hm[INDEX(*m * STRIDE + i)] = 0;

		//get error
		//1. Construct augmented Hm (fill in identity matrix)
		Hm[INDEX((*m) * STRIDE)] = 1.0;
		#pragma unroll
		for (int i = 1; i < p; i++)
		{
			//0. fill potentially non-empty memory first
			for (int j = 0; j < (*m + i + 1); ++j)
				Hm[INDEX((*m + i) * STRIDE + j)] = 0;
			//1. Construct augmented Hm (fill in identity matrix)
			Hm[INDEX((*m + i) * STRIDE + (*m + i - 1))] = 1.0;
		}

#ifdef RB43
		//2. Get phiHm
		info = expAc_variable (*m + p, Hm, h * scale, phiHm, solver, work2);
#elif EXP4
		//2. Get phiHm
		info = phiAc_variable (*m + p, Hm, h * scale, phiHm, solver, work2);
#endif
		if (info != 0)
			return -info;

		//3. Get error
		err = h * (*beta) * fabs(store * phiHm[INDEX((*m) * STRIDE + (*m) - 1)]) * sc_norm(&Vm[GRID_DIM * ((*m) * NSP)], sc);

		//restore Hm(m, m + 1)
		Hm[INDEX((*m - 1) * STRIDE + *m)] = store;

		//restore real Hm
		Hm[INDEX((*m) * STRIDE)] = 0.0;
	}

	return j;
}

#endif