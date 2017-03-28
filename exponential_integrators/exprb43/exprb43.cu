/*!
 * \file exprb43.cu
 *
 * \author Nicholas J. Curtis
 * \date 09/02/2014
 *
 * \brief A krylov subspace integrator using a 4th order (3rd-order embedded)
 * 		  exponential Rosenbrock method of Hochbruck et al. (2009)
 *
 * See full reference:
 * M. Hochbruck, A. Ostermann, J. Schweitzer, Exponential Rosenbrock-type methods, SIAM J. Numer. Anal. 47 (1) (2009) 786–803. doi:10.1137/080717717.
 *
 * NOTE: all matricies stored in column major format!
 *
 */

/** Include common code. */
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include <cuComplex.h>

//various mechanism/solver defns
//these should be included first
#include "header.cuh"
#include "solver_options.cuh"
#include "solver_props.cuh"

#include "dydt.cuh"
#ifndef FINITE_DIFFERENCE
#include "jacob.cuh"
#else
#include "fd_jacob.cuh"
#endif
#include "exprb43_props.cuh"
#include "arnoldi.cuh"
#include "exponential_linear_algebra.cuh"
#include "solver_init.cuh"
#include "gpu_macros.cuh"

#ifdef GENERATE_DOCS
namespace exprb43cu {
#endif

#ifdef LOG_KRYLOV_AND_STEPSIZES
 	extern __device__ double err_log[MAX_STEPS];
 	extern __device__ int m_log[MAX_STEPS];
 	extern __device__ int m1_log[MAX_STEPS];
 	extern __device__ int m2_log[MAX_STEPS];
 	extern __device__ double t_log[MAX_STEPS];
 	extern __device__ double h_log[MAX_STEPS];
 	extern __device__ bool reject_log[MAX_STEPS];
 	extern __device__ int num_integrator_steps;
#endif
#ifdef DIVERGENCE_TEST
 	extern __device__ int integrator_steps[DIVERGENCE_TEST];
#endif

///////////////////////////////////////////////////////////////////////////////

/*!
 * \fn int integrate(const double t_start, const double t_end, const double pr, double* y)
 * \param t_start The initial integration time
 * \param t_end The final integration timestep
 * \param pr User data passed to the RHS function dydt() - commonly used for the Pressure term
 * \param y The state vector
 * \param mech The mechanism memory struct
 * \param solver The solver memory struct
 * \brief 4th-order exponential integrator function w/ adaptive Kyrlov subspace approximation
 * \returns The result of this integration step @see exprb43cu_ErrCodes
 */
__device__ void integrate (const double t_start, const double t_end, const double pr,
							double* __restrict__ y, const mechanism_memory* __restrict__ mech,
							const solver_memory* __restrict__ solver) {

	//initial time
#ifdef CONST_TIME_STEP
	double h = t_end - t_start;
#else
	double h = fmin(1.0e-8, t_end - t_start);
#endif
	double h_new;

	double err_old = 1.0;
	double h_old = h;

	bool reject = false;

	double t = t_start;

	// get scaling for weighted norm
	double * const __restrict__ sc = solver->sc;
	scale_init(y, sc);

#ifdef LOG_KRYLOV_AND_STEPSIZES
	if (T_ID == 0)
	{
		num_integrator_steps = 0;
	}
#endif

	double beta = 0;

	//arrays
	double * const __restrict__ work1 = solver->work1;
	double * const __restrict__ work2 = solver->work2;
	double * const __restrict__ y1 = solver->work3;
	cuDoubleComplex * const __restrict__ work4 = solver->work4;
	double * const __restrict__ fy = mech->dy;
	double * const __restrict__ A = mech->jac;
	double * const __restrict__ Vm = solver->Vm;
	double * const __restrict__ phiHm = solver->phiHm;
	double * const __restrict__ savedActions = solver->savedActions;
	double * const __restrict__ gy = solver->gy;
	int * const __restrict__ result = solver->result;

	//vectors for scaling operations
	double * in[5] = {0, 0, 0, savedActions, y};
	double * out[3] = {0, 0, work1};
	double scale_vec[3] = {0, 0, 0};

	double err = 0.0;
	int failures = 0;
	int steps = 0;
	while (t < t_end) {

		//error checking
		if (failures >= MAX_CONSECUTIVE_ERRORS)
		{
			result[T_ID] = EC_consecutive_steps;
			return;
		}
		if (steps++ >= MAX_STEPS)
		{
			result[T_ID] = EC_max_steps_exceeded;
			return;
		}
		if (t + h <= t)
		{
			result[T_ID] = EC_h_plus_t_equals_h;
			return;
		}

		if (!reject) {
			dydt (t, pr, y, fy, mech);
		#ifdef FINITE_DIFFERENCE
			eval_jacob (t, pr, y, A, mech, work1, work2);
		#else
			eval_jacob (t, pr, y, A, mech);
		#endif
			//gy = fy - A * y
			sparse_multiplier(A, y, gy);
			#pragma unroll
			for (int i = 0; i < NSP; ++i) {
				gy[INDEX(i)] = fy[INDEX(i)] - gy[INDEX(i)];
			}
		}

		#ifdef DIVERGENCE_TEST
		integrator_steps[T_ID]++;
		#endif
		int m = arnoldi(0.5, 1, h, A, solver, fy, &beta, work2, work4);
		if (m + 1 >= STRIDE || m < 0)
		{
			//failure: too many krylov vectors required or singular matrix encountered
			//need to reduce h and try again
			h /= 5.0;
			reject = true;
			failures++;
			continue;
		}

		// Un2 to be stored in work1
		//Un2 is partially in the mth column of phiHm
		//Un2 = y + ** 0.5 * h * phi_1(0.5 * h * A)*fy **
		//Un2 = y + ** beta * Vm * phiHm(:, m) **

		//store h * beta * Vm * phi_1(h * Hm) * e1 in savedActions
		matvec_m_by_m_plusequal(m, phiHm, &phiHm[GRID_DIM * (m * STRIDE)], work1);
		matvec_n_by_m_scale(m, beta, Vm, work1, savedActions);

		//store 0.5 * h *  beta * Vm * phi_1(0.5 * h * Hm) * fy + y in work1
		matvec_n_by_m_scale_add(m, beta, Vm, &phiHm[GRID_DIM * (m * STRIDE)], work1, y);
		//work1 is now equal to Un2

		//next compute Dn2
		//Dn2 = (F(Un2) - Jn * Un2) - gy

		dydt(t, pr, work1, &savedActions[GRID_DIM * NSP], mech);
		sparse_multiplier(A, work1, work2);

		#pragma unroll
		for (int i = 0; i < NSP; ++i) {
			work1[INDEX(i)] = savedActions[INDEX(NSP + i)] - work2[INDEX(i)] - gy[INDEX(i)];
		}
		//work1 is now equal to Dn2

		//partially compute Un3 as:
		//Un3 = y + ** h * phi_1(hA) * fy ** + h * phi_1(hA) * Dn2
		//Un3 = y + ** h * beta * Vm * phiHm(:, m) **

		//now we need the action of the exponential on Dn2
		int m1 = arnoldi(1.0, 4, h, A, solver, work1, &beta, work2, work4);
		if (m1 + 4 >= STRIDE || m1 < 0)
		{
			//need to reduce h and try again
			h /= 5.0;
			reject = true;
			failures++;
			continue;
		}

		//save Phi3(h * A) * Dn2 to savedActions[0]
		//save Phi4(h * A) * Dn2 to savedActions[NSP]
		//add the action of phi_1 on Dn2 to y and hn * phi_1(hA) * fy to get Un3
		in[0] = &phiHm[GRID_DIM * ((m1 + 2) * STRIDE)];
		in[1] = &phiHm[GRID_DIM * ((m1 + 3) * STRIDE)];
		in[2] = &phiHm[GRID_DIM * ((m1) * STRIDE)];
		out[0] = &savedActions[GRID_DIM * NSP];
		out[1] = &savedActions[GRID_DIM * 2 * NSP];
		scale_vec[0] = beta / (h * h);
		scale_vec[1] = beta / (h * h * h);
		scale_vec[2] = beta;
		matvec_n_by_m_scale_special(m1, scale_vec, Vm, in, out);
		//Un3 is now in work1

		//next compute Dn3
		//Dn3 = F(Un3) - A * Un3 - gy
		dydt(t, pr, work1, &savedActions[GRID_DIM * 3 * NSP], mech);
		sparse_multiplier(A, work1, work2);

		#pragma unroll
		for (int i = 0; i < NSP; ++i) {
			work1[INDEX(i)] = savedActions[INDEX(3 * NSP + i)] - work2[INDEX(i)] - gy[INDEX(i)];
		}
		//work1 is now equal to Dn3

		//finally we need the action of the exponential on Dn3
		int m2 = arnoldi(1.0, 4, h, A, solver, work1, &beta, work2, work4);
		if (m2 + 4 >= STRIDE || m2 < 0)
		{
			//need to reduce h and try again
			h /= 5.0;
			reject = true;
			failures++;
			continue;
		}
		out[0] = &savedActions[GRID_DIM * 3 * NSP];
		out[1] = &savedActions[GRID_DIM * 4 * NSP];
		in[0] = &phiHm[GRID_DIM * (m2 + 2) * STRIDE];
		in[1] = &phiHm[GRID_DIM * (m2 + 3) * STRIDE];
		scale_vec[0] = beta / (h * h);
		scale_vec[1] = beta / (h * h * h);
		matvec_n_by_m_scale_special2(m2, scale_vec, Vm, in, out);

		//construct y1 and error vector
		#pragma unroll
		for (int i = 0; i < NSP; ++i) {
			//y1 = y + h * phi1(h * A) * fy + h * sum(bi * Dni)
			y1[INDEX(i)] = y[INDEX(i)] + savedActions[INDEX(i)] + 16.0 * savedActions[INDEX(NSP + i)] - 48.0 * savedActions[INDEX(2 * NSP + i)] + -2.0 * savedActions[INDEX(3 * NSP + i)] + 12.0 * savedActions[INDEX(4 * NSP + i)];
			//error vec
			work1[INDEX(i)] = 48.0 * savedActions[INDEX(2 * NSP + i)] - 12.0 * savedActions[INDEX(4 * NSP + i)];
		}


		//scale and find err
		scale (y, y1, work2);
		err = fmax(EPS, sc_norm(work1, work2));

		// classical step size calculation
		h_new = pow(err, -1.0 / ORD);

#ifdef LOG_KRYLOV_AND_STEPSIZES
		if (T_ID == 0 && num_integrator_steps >= 0) {
			err_log[num_integrator_steps] = err;
			m_log[num_integrator_steps] = m;
			m1_log[num_integrator_steps] = m1;
			m2_log[num_integrator_steps] = m2;
			t_log[num_integrator_steps] = t;
			h_log[num_integrator_steps] = h;
			reject_log[num_integrator_steps] = err > 1.0;
			num_integrator_steps++;
			if (num_integrator_steps >= MAX_STEPS)
			{
				printf("Number of steps out of bounds! Overwriting\n");
				num_integrator_steps = -1;
			}
		}
#endif

#ifndef CONST_TIME_STEP
		failures = 0;
		if (err <= 1.0) {
			// update y, scale vector and t
			#pragma unroll
			for (int i = 0; i < NSP; ++i)
			{
				sc[INDEX(i)] = work2[INDEX(i)];
				y[INDEX(i)] = y1[INDEX(i)];
			}
			t += h;

			// minimum of classical and Gustafsson step size prediction
			h_new = fmin(h_new, (h / h_old) * pow((err_old / (err * err)), (1.0 / ORD)));

			// limit to 0.2 <= (h_new/8) <= 8.0
			h_new = h * fmax(fmin(0.9 * h_new, 8.0), 0.2);

			// store time step and error
			err_old = fmax(1.0e-2, err);
			h_old = h;

			// check if last step rejected
			if (reject) {
				h_new = fmin(h, h_new);
				reject = false;
			}
			h = fmin(h_new, t_end - t);

		} else {
			// limit to 0.2 <= (h_new/8) <= 8.0
			h_new = h * fmax(fmin(0.9 * h_new, 8.0), 0.2);
			h_new = fmin(h_new, t_end - t);

			reject = true;
			h = fmin(h, h_new);
		}
#else
		//constant time stepping
		//update y & t
		#pragma unroll
		for (int i = 0; i < NSP; ++i)
		{
			y[INDEX(i)] = y1[INDEX(i)];
		}
		t += h;
#endif

	} // end while

	result[T_ID] = EC_success;

}

#ifdef GENERATE_DOCS
}
#endif
