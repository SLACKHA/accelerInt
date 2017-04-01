/*!
 * \file
 * \brief Implementation of the GPU arnoldi iteration methods
 *
 * \author Nicholas Curtis
 * \date 03/09/2015
 *
 * Note: turn on EXACT_KRYLOV krylov definition to use the use the "happy breakdown" criteria in determining end of krylov iteration
 */

#ifndef ARNOLDI_CUH
#define ARNOLDI_CUH

#include <string.h>

#include "header.cuh"
#include "phiAHessenberg.cuh"
#include "exponential_linear_algebra.cuh"
#include "sparse_multiplier.cuh"
#include "solver_options.cuh"
#include "solver_props.cuh"

//#define EXACT_KRYLOV

//! The list of indicies to check the Krylov projection error at
__constant__ int index_list[23] = {1, 2, 3, 4, 5, 6, 7, 9, 11, 14, 17, 21, 27, 34, 42, 53, 67, 84, 106, 133, 167, 211, 265};

///////////////////////////////////////////////////////////////////////////////

/*!
 * \fn int arnoldi(const double scale,
			const int p, const double h,
			const double* __restrict__ A,
			const solver_memory* __restrict__ solver,
			const double* __restrict__ v, double* __restrict__ beta,
			double * __restrict__ work,
			cuDoubleComplex* __restrict__ work2)
 * \brief Runs the arnoldi iteration to calculate the Krylov projection
 * \returns				m - the ending size of the matrix
 * \param[in]			scale	the value to scale the timestep by
 * \param[in]			p		the order of the maximum phi function needed
 * \param[in]			h		the timestep
 * \param[in]			A 		the jacobian matrix
 * \param[in,out]		solver  the solver memory struct
 * \param[in]  			v 		the vector to use for the krylov subspace
 * \param[out] 			beta 	the norm of the v vector
 * \param[in,out]		work    A work vector
 * \param[in,out]		work2   A complex work vector
 */
__device__
int arnoldi(const double scale,
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

	while (err >= 1.0)
	{
		for (; j < index_list[index] && j + p < STRIDE; j++)
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
				j++;
				break;
			}
			scale_mult(1.0 / Hm[INDEX(j * STRIDE + j + 1)], work, &Vm[GRID_DIM * ((j + 1) * NSP)]);
		}
#ifndef CONST_TIME_STEP
		if (j + p >= STRIDE)
			return j;
#else
		if (j + p >= STRIDE)
			j = STRIDE - p - 1;
#endif
		index++;
		//resize Hm to be mxm, and store Hm(m, m + 1) for later
		store = Hm[INDEX((j - 1) * STRIDE + j)];
		Hm[INDEX((j - 1) * STRIDE + j)] = 0.0;

		//0. fill potentially non-empty memory first
		for (int i = 0; i < (j + 2); ++i)
			Hm[INDEX(j * STRIDE + i)] = 0;

		//get error
		//1. Construct augmented Hm (fill in identity matrix)
		Hm[INDEX((j) * STRIDE)] = 1.0;
		#pragma unroll
		for (int i = 1; i < p; i++)
		{
			//0. fill potentially non-empty memory first
			for (int k = 0; k < (j + i + 2); ++k)
				Hm[INDEX((j + i) * STRIDE + k)] = 0;
			//1. Construct augmented Hm (fill in identity matrix)
			Hm[INDEX((j + i) * STRIDE + (j + i - 1))] = 1.0;
		}

#ifdef RB43
		//2. Get phiHm
		info = expAc_variable (j + p, Hm, h * scale, phiHm, solver, work2);
#elif EXP4
		//2. Get phiHm
		info = phiAc_variable (j + p, Hm, h * scale, phiHm, solver, work2);
#endif
		if (info != 0)
			return -info;

		//3. Get error
		err = h * (*beta) * fabs(store * phiHm[INDEX((j) * STRIDE + (j) - 1)]) * sc_norm(&Vm[GRID_DIM * ((j) * NSP)], sc);

		//restore Hm(m, m + 1)
		Hm[INDEX((j - 1) * STRIDE + j)] = store;

		//restore real Hm
		Hm[INDEX((j) * STRIDE)] = 0.0;
#if defined(LOG_OUTPUT) && defined(EXACT_KRYLOV)
		//kill the error such that we will continue
		//and greatly reduce the subspace approximation error
		if (index_list[index] + p < STRIDE  && fabs(Hm[INDEX((j - 1) * STRIDE + j)]) >= ATOL)
		{
			err = 10;
		}
#endif
#ifdef CONST_TIME_STEP
		if (j == STRIDE - p - 1)
			break;
#endif
	}

	return j;
}

#endif