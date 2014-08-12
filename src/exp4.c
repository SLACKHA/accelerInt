/** 
 * \file exp4.c
 *
 * \author Kyle E. Niemeyer
 * \date 07/23/2012
 *
 * 
 */
 
/** Include common code. */
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdbool.h>

#include "header.h"
#include "exp4.h"
#include "derivs.h"
#include "phiA.h"

/** Order of embedded methods for time step control. */
#define ORD 3

static inline void matvec (const Real*, const Real*, Real*);
static inline void scale (const Real*, const Real*, Real*);
static inline Real sc_norm (const Real*, const Real*);

///////////////////////////////////////////////////////////////////////////////

/** Matrix-vector multiplication
 * 
 * Performs inline matrix-vector multiplication (with unrolled loops).
 * 
 * \param[in]		A		matrix
 * \param[in]		v		vector
 * \param[out]	Av	vector that is A * v
 */
static inline
void matvec (const Real * A, const Real * v, Real * Av) {
	#pragma unroll
	for (uint i = 0; i < NN; ++i) {
		Av[i] = ZERO;
		
		#pragma unroll
		for (uint j = 0; j < NN; ++j) {
			Av[i] += A[i + (j * NN)] * v[j];
		}
	}
}

///////////////////////////////////////////////////////////////////////////////

/** Get scaling for weighted norm
 * 
 * \param[in]		y0		values at current timestep
 * \param[in]		y1		values at next timestep
 * \param[out]	sc	array of scaling values
 */
static inline
void scale (const Real * y0, const Real * y1, Real * sc) {
	#pragma unroll
	for (uint i = 0; i < NN; ++i) {
		sc[i] = ATOL + fmax(fabs(y0[i]), fabs(y1[i])) * RTOL;
	}
}

///////////////////////////////////////////////////////////////////////////////

/** Perform weighted norm
 * 
 * \param[in]		nums	values to be normed
 * \param[in]		sc		scaling array for norm
 * \return			norm	weighted norm
 */
static inline
Real sc_norm (const Real * nums, const Real * sc) {
	Real norm = ZERO;
	
	#pragma unroll
	for (uint i = 0; i < NN; ++i) {
		norm += nums[i] * nums[i] / (sc[i] * sc[i]);
	}
	
	norm = sqrt(norm / NN);
	
	return norm;
}

///////////////////////////////////////////////////////////////////////////////

/** 4th-order exponential integrator function
 * 
 * 
 */
void exp4_int (const Real t_start, const Real t_end, const Real pr, Real* y) {
	
	Real h = 1.0e-8;
	Real h_max, h_new;
	
	Real err_old = 1.0;
	Real h_old = h;
	
	bool reject = false;
	
	Real t = t_start;
	
	while ((t < t_end) && (t + h > t)) {
    
		// temporary arrays
		Real temp[NN];
		Real f_temp[NN];
		Real y1[NN];
		
		h_max = t_end - t;
	
		// source vector
		Real fy[NN];
		dydt (t, pr, y, fy);
	
		// Jacobian matrix
		Real A[NN * NN] = {ZERO};
		//eval_jacob (t, pr, y, A);
    	eval_fd_jacob (t, pr, y, A);
			
		// get phi(A * h/3)
		Real phiA[NN * NN];
		phiAc (A, h / 3.0, phiA);
		
		// k1
		Real k1[NN];
		matvec (phiA, fy, k1);
	
		// k2
		Real k2[NN];
		matvec (A, k1, temp);
		matvec (phiA, temp, k2);
	
		#pragma unroll
		for (uint i = 0; i < NN; ++i) {
			k2[i] = (h / 6.0) * k2[i] + k1[i];
		}
	
		// k3
		Real k3[NN];
		matvec (A, k2, temp);
		matvec (phiA, temp, k3);
	
		#pragma unroll
		for (uint i = 0; i < NN; ++i) {
			k3[i] = (2.0 / 9.0) * h * k3[i] + (2.0 / 3.0) * k2[i] + (k1[i] / 3.0);
		}
			
		// k4
		Real k4[NN];
	
		#pragma unroll
		for (uint i = 0; i < NN; ++i) {
			// f4
			f_temp[i] = h * ((-7.0 / 300.0) * k1[i] + (97.0 / 150.0) * k2[i] - (37.0 / 300.0) * k3[i]);
		
			k4[i] = y[i] + f_temp[i];
		}
		
		dydt (t, pr, k4, temp);
		
		matvec (A, f_temp, k4);
	
		#pragma unroll
		for (uint i = 0; i < NN; ++i) {
			temp[i] = temp[i] - fy[i] - k4[i];
		}
	
		matvec (phiA, temp, k4);
	
		// k5
		Real k5[NN];
		matvec (A, k4, temp);
		matvec (phiA, temp, k5);
	
		#pragma unroll
		for (uint i = 0; i < NN; ++i) {
			k5[i] = (h / 6.0) * k5[i] + k4[i];
		}
			
		// k6
		Real k6[NN];
		matvec (A, k5, temp);
		matvec (phiA, temp, k6);
	
		#pragma unroll
		for (uint i = 0; i < NN; ++i) {
			k6[i] = (2.0 / 9.0) * h * k6[i] + (2.0 / 3.0) * k5[i] + (k4[i] / 3.0);
		}
			
		// k7
		Real k7[NN];
		#pragma unroll
		for (uint i = 0; i < NN; ++i) {
			// f7
			f_temp[i] = h * ((59.0 / 300.0) * k1[i] - (7.0 / 75.0) * k2[i] + (269.0 / 300.0) * k3[i] + (2.0 / 3.0) * (k4[i] + k5[i] + k6[i]));
		
			k7[i] = y[i] + f_temp[i];
		}
	
		dydt (t, pr, k7, temp);
		matvec (A, f_temp, k7);
	
		#pragma unroll
		for (uint i = 0; i < NN; ++i) {
			temp[i] = temp[i] - fy[i] - k7[i];
		}
	
		matvec (phiA, temp, k7);
				
		// y_n+1
		#pragma unroll
		for (uint i = 0; i < NN; ++i) {
			y1[i] = y[i] + h * (k3[i] + k4[i] - (4.0 / 3.0) * k5[i] + k6[i] + (1.0 / 6.0) * k7[i]);
		}
		
		// get scaling for weighted norm
		Real sc[NN];
		scale (y, y1, sc);	
		
		///////////////////
		// calculate errors
		///////////////////
	
		// error of embedded order 3 method
		#pragma unroll
		for (uint i = 0; i < NN; ++i) {
			temp[i] = k3[i] - (2.0 / 3.0) * k5[i] + 0.5 * (k6[i] + k7[i] - k4[i]) - (y1[i] - y[i]) / h;
		}	
		Real err = h * sc_norm(temp, sc);
		
		// error of embedded W method
		#pragma unroll
		for (uint i = 0; i < NN; ++i) {
			temp[i] = -k1[i] + 2.0 * k2[i] - k4[i] + k7[i] - (y1[i] - y[i]) / h;
		}
		//Real err_W = h * sc_norm(temp, sc);
		err = fmax(EPS, fmin(err, h * sc_norm(temp, sc)));
		
		// classical step size calculation
		h_new = pow(err, -1.0 / ORD);	
		
		if (err <= ONE) {
			
			// minimum of classical and Gustafsson step size prediction
			h_new = fmin(h_new, (h / h_old) * pow((err_old / (err * err)), (1.0 / ORD)));
			
			// limit to 0.2 <= (h_new/8) <= 8.0
			h_new = h * fmax(fmin(0.9 * h_new, 8.0), 0.2);
			h_new = fmin(h_new, h_max);
			
			// update y and t
			#pragma unroll
			for (uint i = 0; i < NN; ++i) {
				y[i] = y1[i];
			}
			
			t += h;
			
			// print
      
			//printf("%18.15e %18.15e\n", t, y[0]);
      /*
			for (int i = 0; i < NN; ++i) {
				printf(" %18.15e", y[i]);
			}
			printf("\n");
      */
			
			// store time step and error
			err_old = fmax(1.0e-2, err);
			h_old = h;
			
			// check if last step rejected
			if (reject) {
				reject = false;
				h_new = fmin(h, h_new);
			}
			h = fmin(h_new, fabs(t_end - t));
						
		} else {
			// limit to 0.2 <= (h_new/8) <= 8.0
			h_new = h * fmax(fmin(0.9 * h_new, 8.0), 0.2);
			h_new = fmin(h_new, h_max);
			
			reject = true;
			h = fmin(h, h_new);
		}
		
	} // end while
	
}