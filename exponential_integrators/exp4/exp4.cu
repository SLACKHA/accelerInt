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

#include "header.cuh"
#include "dydt.cuh"
#include "jacob.cuh"
#include "exp4_props.cuh"
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
void integrate (const double t_start, const double t_end, const double pr, double* y) {
	
	//initial time
	double h = fmin(1.0e-8, t_end - t_start);
	double h_new;

	double err_old = 1.0;
	double h_old = h;
	
	bool reject = false;

	double t = t_start;

	// get scaling for weighted norm
	double sc[NSP];
	scale_init(y, sc);
	
	double beta = 0;
	// source vector	
	double fy[NSP];
	// Jacobian matrix
	double A[NSP * NSP] = {0.0};

	// temporary arrays
	double temp[NSP];
	double f_temp[NSP];
	double y1[NSP];
	double Hm[STRIDE * STRIDE];// = {0.0};
	double Vm[NSP * STRIDE];
	double phiHm[STRIDE * STRIDE];
	double err = 0.0;
			
	// i-vectors
	double k1[NSP];
	double k2[NSP];
	double k3[NSP];
	double k4[NSP];
	double k5[NSP];
	double k6[NSP];
	double k7[NSP];
	//initial krylov subspace sizes
	int m, m1, m2;
	while ((t < t_end) && (t + h > t)) {
		
		if (!reject) {
			//memset(fy, 0,  NSP * sizeof(double));
			//memset(A, 0, NSP * NSP * sizeof(double));
			dydt (t, pr, y, fy);
			eval_jacob (t, pr, y, A);
		}

		#ifdef DIVERGENCE_TEST
		integrator_steps[T_ID]++;
		#endif
		if (arnoldi(&m, 1.0 / 3.0, P, h, A, fy, sc, &beta, Vm, Hm, phiHm) >= M_MAX)
		{
			h /= 3;
			continue;
		}

		// k1
		double k1[NSP];
		//k1 is partially in the first column of phiHm
		//k1 = beta * Vm * phiHm(:, 1)
		matvec_n_by_m_scale(m, beta, Vm, phiHm, k1);
	
		// k2
		double k2[NSP];
		//computing phi(2h * A)
		matvec_m_by_m (m, phiHm, phiHm, temp);
		//note: f_temp will contain hm * phi * phi * e1 for later use
		matvec_m_by_m (m, Hm, temp, f_temp);
		matvec_n_by_m_scale_add(m, beta * (h / 6.0), Vm, f_temp, k2, k1);
	
		// k3
		double k3[NSP];
		//use the stored hm * phi * phi * e1 to get phi(3h * A)
		matvec_m_by_m (m, phiHm, f_temp, temp);
		matvec_m_by_m (m, Hm, temp, f_temp);
		matvec_n_by_m_scale_add_subtract(m, beta * (h * h / 27.0), Vm, f_temp, k3, k2, k1);
			
		// d4
		double k4[NSP];
		#pragma unroll
		for (int i = 0; i < NSP; ++i) {
			// f4
			f_temp[i] = h * ((-7.0 / 300.0) * k1[i] + (97.0 / 150.0) * k2[i] - (37.0 / 300.0) * k3[i]);
		
			k4[i] = y[i] + f_temp[i];
		}
		
		dydt (t, pr, k4, temp);
		sparse_multiplier (A, f_temp, k4);
	
		#pragma unroll
		for (int i = 0; i < NSP; ++i) {
			k4[i] = temp[i] - fy[i] - k4[i];
		}

		//do arnoldi
		arnoldi(&m1, 1.0 / 3.0, P, h, A, k4, sc, &beta, Vm, Hm, phiHm);
		//k4 is partially in the m'th column of phiHm
		matvec_n_by_m_scale(m1, beta, Vm, phiHm, k4);
	
		// k5
		double k5[NSP];
		//computing phi(2h * A)
		matvec_m_by_m (m1, phiHm, phiHm, temp);
		//note: f_temp will contain hm * phi * phi * e1 for later use
		matvec_m_by_m (m1, Hm, temp, f_temp);
		matvec_n_by_m_scale_add(m1, beta * (h / 6.0), Vm, f_temp, k5, k4);
			
		// k6
		double k6[NSP];
		//use the stored hm * phi * phi * e1 to get phi(3h * A)
		matvec_m_by_m (m1, phiHm, f_temp, temp);
		matvec_m_by_m (m1, Hm, temp, f_temp);
		matvec_n_by_m_scale_add_subtract(m1, beta * (h * h / 27.0), Vm, f_temp, k6, k5, k4);
			
		// k7
		double k7[NSP];
		#pragma unroll
		for (int i = 0; i < NSP; ++i) {
			// f7
			f_temp[i] = h * ((59.0 / 300.0) * k1[i] - (7.0 / 75.0) * k2[i] + (269.0 / 300.0) * k3[i] + (2.0 / 3.0) * (k4[i] + k5[i] + k6[i]));
		
			k7[i] = y[i] + f_temp[i];
		}
	
		dydt (t, pr, k7, temp);
		sparse_multiplier (A, f_temp, k7);
	
		#pragma unroll
		for (int i = 0; i < NSP; ++i) {
			k7[i] = temp[i] - fy[i] - k7[i];
		}
	
		arnoldi(&m2, 1.0 / 3.0, P, h, A, k7, sc, &beta, Vm, Hm, phiHm);
		//k7 is partially in the m'th column of phiHm
		matvec_n_by_m_scale(m2, beta / (h / 3.0), Vm, &phiHm[m2 * STRIDE], k7);
				
		// y_n+1
		#pragma unroll
		for (int i = 0; i < NSP; ++i) {
			y1[i] = y[i] + h * (k3[i] + k4[i] - (4.0 / 3.0) * k5[i] + k6[i] + (1.0 / 6.0) * k7[i]);
		}
		
		scale (y, y1, f_temp);	
		
		///////////////////
		// calculate errors
		///////////////////
	
		// error of embedded order 3 method
		#pragma unroll
		for (int i = 0; i < NSP; ++i) {
			temp[i] = k3[i] - (2.0 / 3.0) * k5[i] + 0.5 * (k6[i] + k7[i] - k4[i]) - (y1[i] - y[i]) / h;
		}	
		err = h * sc_norm(temp, f_temp);
		
		// error of embedded W method
		#pragma unroll
		for (int i = 0; i < NSP; ++i) {
			temp[i] = -k1[i] + 2.0 * k2[i] - k4[i] + k7[i] - (y1[i] - y[i]) / h;
		}
		//double err_W = h * sc_norm(temp, sc);
		err = fmax(EPS, fmin(err, h * sc_norm(temp, f_temp)));
		
		// classical step size calculation
		h_new = pow(err, -1.0 / ORD);	
		
		if (err <= 1.0) {
			#pragma unroll
			for (int i = 0; i < NSP; ++i)
				sc[i] = f_temp[i];

			// minimum of classical and Gustafsson step size prediction
			h_new = fmin(h_new, (h / h_old) * pow((err_old / (err * err)), (1.0 / ORD)));
			
			// limit to 0.2 <= (h_new/8) <= 8.0
			h_new = h * fmax(fmin(0.9 * h_new, 8.0), 0.2);
			
			// update y and t
			#pragma unroll
			for (int i = 0; i < NSP; ++i) {
				y[i] = y1[i];
			}
			
			t += h;
			
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
}