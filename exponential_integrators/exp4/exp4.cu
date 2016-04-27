/** 
 * \file krylov.c
 *
 * \author Nicholas J. Curtis
 * \date 09/02/2014
 *
 * A krylov subspace integrator using the EXP4 method
 * based on the work of Niesen and Wright (2012)
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
#include "arnoldi.cuh"
#include "exponential_linear_algebra.cuh"
#include "solver_init.cuh"

#define T_ID (threadIdx.x + blockIdx.x * blockDim.x)
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

/** 4th-order exponential integrator function w/ adaptive Kyrlov subspace approximation
 * 
 * 
 */
__device__
void integrate (const double t_start, const double t_end, const double pr,
				double* __restrict__ y, const mechanism_memory* __restrict__ mech,
				const solver_memory* __restrict__ solver) {
	
	//initial time
	double h = fmin(1.0e-8, t_end - t_start);
	double h_new;

	double err_old = 1.0;
	double h_old = h;
	double beta = 0;
	double err = 0.0;
	
	bool reject = false;
	int failures = 0;
	int steps = 0;

	double t = t_start;

	//arrays
	double * const __restrict__ sc = solver->sc;
	double * const __restrict__ work1 = solver->work1;
	double * const __restrict__ work2 = solver->work2; 
	double * const __restrict__ y1 = solver->work3;
	cuDoubleComplex * const __restrict__ work4 = solver->work4;
	double * const __restrict__ fy = mech->dy;
	double * const __restrict__ A = mech->jac;
	double * const __restrict__ Hm = solver->Hm;
	double * const __restrict__ Vm = solver->Vm;
	double * const __restrict__ phiHm = solver->phiHm;
	double * const __restrict__ k1 = solver->k1;
	double * const __restrict__ k2 = solver->k2;
	double * const __restrict__ k3 = solver->k3;
	double * const __restrict__ k4 = solver->k4;
	double * const __restrict__ k5 = solver->k5;
	double * const __restrict__ k6 = solver->k6;
	double * const __restrict__ k7 = solver->k7;
	int * const __restrict__ result = solver->result;

	// get scaling for weighted norm
	scale_init(y, sc);
			
	//initial krylov subspace sizes
	while (t < t_end) {

		//error checking
		if (failures >= 5)
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
		}

		#ifdef DIVERGENCE_TEST
		integrator_steps[T_ID]++;
		#endif
		int m = arnoldi(1.0 / 3.0, P, h, A, solver, fy, &beta, work1, work4);
		if (m >= M_MAX || m < 0)
		{
			//need to reduce h and try again
			h /= 5.0;
			failures++;
			continue;
		}

		// k1
		//k1 is partially in the first column of phiHm
		//k1 = beta * Vm * phiHm(:, 1)
		matvec_n_by_m_scale(m, beta, Vm, phiHm, k1);
	
		// k2
		//computing phi(2h * A)
		matvec_m_by_m (m, phiHm, phiHm, work1);
		//note: work2 will contain hm * phi * phi * e1 for later use
		matvec_m_by_m (m, Hm, work1, work2);
		matvec_n_by_m_scale_add(m, beta * (h / 6.0), Vm, work2, k2, k1);
	
		// k3
		//use the stored hm * phi * phi * e1 to get phi(3h * A)
		matvec_m_by_m (m, phiHm, work2, work1);
		matvec_m_by_m (m, Hm, work1, work2);
		matvec_n_by_m_scale_add_subtract(m, beta * (h * h / 27.0), Vm, work2, k3, k2, k1);
			
		// d4
		#pragma unroll
		for (int i = 0; i < NSP; ++i) {
			// f4
			work2[INDEX(i)] = h * ((-7.0 / 300.0) * k1[INDEX(i)] + (97.0 / 150.0) * k2[INDEX(i)] - (37.0 / 300.0) * k3[INDEX(i)]);
		
			k4[INDEX(i)] = y[INDEX(i)] + work2[INDEX(i)];
		}
		
		dydt (t, pr, k4, work1, mech);
		sparse_multiplier (A, work2, k4);
	
		#pragma unroll
		for (int i = 0; i < NSP; ++i) {
			k4[INDEX(i)] = work1[INDEX(i)] - fy[INDEX(i)] - k4[INDEX(i)];
		}

		//do arnoldi
		int m1 = arnoldi(1.0 / 3.0, P, h, A, solver, k4, &beta, work1, work4);
		if (m1 >= M_MAX || m1 < 0)
		{
			//need to reduce h and try again
			h /= 5.0;
			failures++;
			continue;
		}
		//k4 is partially in the m'th column of phiHm
		matvec_n_by_m_scale(m1, beta, Vm, phiHm, k4);
	
		// k5
		//computing phi(2h * A)
		matvec_m_by_m (m1, phiHm, phiHm, work1);
		//note: work2 will contain hm * phi * phi * e1 for later use
		matvec_m_by_m (m1, Hm, work1, work2);
		matvec_n_by_m_scale_add(m1, beta * (h / 6.0), Vm, work2, k5, k4);
			
		// k6
		//use the stored hm * phi * phi * e1 to get phi(3h * A)
		matvec_m_by_m (m1, phiHm, work2, work1);
		matvec_m_by_m (m1, Hm, work1, work2);
		matvec_n_by_m_scale_add_subtract(m1, beta * (h * h / 27.0), Vm, work2, k6, k5, k4);
			
		// k7
		#pragma unroll
		for (int i = 0; i < NSP; ++i) {
			// f7
			work2[INDEX(i)] = h * ((59.0 / 300.0) * k1[INDEX(i)] - (7.0 / 75.0) * k2[INDEX(i)] + (269.0 / 300.0) * k3[INDEX(i)] + (2.0 / 3.0) * (k4[INDEX(i)] + k5[INDEX(i)] + k6[INDEX(i)]));
		
			k7[INDEX(i)] = y[INDEX(i)] + work2[INDEX(i)];
		}
	
		dydt (t, pr, k7, work1, mech);
		sparse_multiplier (A, work2, k7);
	
		#pragma unroll
		for (int i = 0; i < NSP; ++i) {
			k7[INDEX(i)] = work1[INDEX(i)] - fy[INDEX(i)] - k7[INDEX(i)];
		}
	
		int m2 = arnoldi(1.0 / 3.0, P, h, A, solver, k7, &beta, work1, work4);
		if (m2 >= M_MAX || m2 < 0)
		{
			//need to reduce h and try again
			h /= 5.0;
			failures++;
			continue;
		}
		//k7 is partially in the m'th column of phiHm
		matvec_n_by_m_scale(m2, beta / (h / 3.0), Vm, &phiHm[GRID_DIM * m2 * STRIDE], k7);
				
		// y_n+1
		#pragma unroll
		for (int i = 0; i < NSP; ++i) {
			y1[INDEX(i)] = y[INDEX(i)] + h * (k3[INDEX(i)] + k4[INDEX(i)] - (4.0 / 3.0) * k5[INDEX(i)] + k6[INDEX(i)] + (1.0 / 6.0) * k7[INDEX(i)]);
		}
		
		scale (y, y1, work2);	
		
		///////////////////
		// calculate errors
		///////////////////
	
		// error of embedded order 3 method
		#pragma unroll
		for (int i = 0; i < NSP; ++i) {
			work1[INDEX(i)] = k3[INDEX(i)] - (2.0 / 3.0) * k5[INDEX(i)] + 0.5 * (k6[INDEX(i)] + k7[INDEX(i)] - k4[INDEX(i)]) - (y1[INDEX(i)] - y[INDEX(i)]) / h;
		}	
		err = h * sc_norm(work1, work2);
		
		// error of embedded W method
		#pragma unroll
		for (int i = 0; i < NSP; ++i) {
			work1[INDEX(i)] = -k1[INDEX(i)] + 2.0 * k2[INDEX(i)] - k4[INDEX(i)] + k7[INDEX(i)] - (y1[INDEX(i)] - y[INDEX(i)]) / h;
		}
		//double err_W = h * sc_norm(temp, sc);
		err = fmax(EPS, fmin(err, h * sc_norm(work1, work2)));
		
		// classical step size calculation
		h_new = pow(err, -1.0 / ORD);	
		
		failures = 0;
		if (err <= 1.0) {
			// update y, t and scale
			#pragma unroll
			for (int i = 0; i < NSP; ++i)
			{
				y[INDEX(i)] = y1[INDEX(i)];
				sc[INDEX(i)] = work2[INDEX(i)];
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
				reject = false;
				h_new = fmin(h, h_new);
			}
			h = fmin(h_new, t_end - t);
						
		} else {

			// limit to 0.2 <= (h_new/8) <= 8.0
			h_new = h * fmax(fmin(0.9 * h_new, 8.0), 0.2);
			h_new = fmin(h_new, t_end - t);
			
			reject = true;
			h = fmin(h, h_new);
		}

	} // end while

	result[T_ID] = EC_success;
}