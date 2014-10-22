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

#include "header.h"
#include "exp4.h"
#include "derivs.h"
#include "phiAHessenberg.h"
#include "sparse_multiplier.h"


static inline void matvec_m_by_m (const int, const Real *, const Real *, Real *);
static inline void matvec_n_by_m (const int, const Real *, const Real *, Real *);
static inline void scale (const Real*, const Real*, Real*);
static inline Real sc_norm (const Real*, const Real*);
static inline double dotproduct(const Real*, const Real*);
static inline Real normalize(const Real*, Real*);
static inline void scale_subtract(const Real, const Real*, Real*);
static inline Real two_norm(const Real*);
static inline void scale_mult(const Real, const Real*, Real*);
static inline Real arnoldi(int*, Real*, bool, const Real, const Real, const Real, const Real*, const Real*, Real*, Real*, Real*);

#ifdef COMPILE_TESTING_METHODS
void matvec_m_by_m_test (const int i , const Real * j, const Real * k, Real * l) {
	matvec_m_by_m(i, j, k, l);
}
void matvec_n_by_m_test (const int i, const Real * j, const Real * k, Real *l){
	matvec_n_by_m(i, j, k, l);
}
double dotproduct_test(const Real* i, const Real* j){
	return dotproduct(i, j);
}
Real normalize_test(const Real* i, Real* j){
	return normalize(i, j);
}
void scale_subtract_test(const Real i, const Real* j, Real* k) {
	scale_subtract(i, j, k);
}
Real two_norm_test(const Real* i){
	return two_norm(i);
}
void scale_mult_test(const Real i, const Real* j, Real* k){
	scale_mult(i, j, k);
}
#endif

//max order of the phi functions (i.e. for error estimation)
#define P 2
//max size of arrays
#define STRIDE (NN + P)
//safety factors
#define GAMMA 0.8
#define DELTA 1.2
//order of embedded methods
#define ORD 3
#define TOL 1e-7

///////////////////////////////////////////////////////////////////////////////

/** Matrix-vector multiplication of a matrix sized MxM and a vector Mx1
 * 
 * Performs inline matrix-vector multiplication (with unrolled loops) 
 * 
 * \param[in]		m 		size of the matrix
 * \param[in]		A		matrix of size MxM
 * \param[in]		V		vector of size Mx1
 * \param[out]		Av		vector that is A * v
 */
static inline
void matvec_m_by_m (const int m, const Real * A, const Real * V, Real * Av) {
	//for each row
	#pragma unroll
	for (int i = 0; i < m; ++i) {
		Av[i] = ZERO;
		
		//go across a row of A, multiplying by a column of phiHm
		#pragma unroll
		for (int j = 0; j < m; ++j) {
			Av[i] += A[j * STRIDE + i] * V[j];
		}
	}
}

///////////////////////////////////////////////////////////////////////////////

/** Matrix-vector multiplication of a matrix sized NNxM and a vector of size Mx1
 * 
 * Performs inline matrix-vector multiplication (with unrolled loops)
 * 
 * \param[in]		m 		size of the matrix
 * \param[in]		A		matrix
 * \param[in]		V		the vector
 * \param[out]		Av		vector that is A * V
 */
static inline
void matvec_n_by_m (const int m, const Real * A, const Real * V, Real * Av) {
	//for each row
	#pragma unroll
	for (int i = 0; i < NN; ++i) {
		Av[i] = ZERO;
		
		//go across a row of A, multiplying by a column of phiHm
		#pragma unroll
		for (int j = 0; j < m; ++j) {
			Av[i] += A[j * STRIDE + i] * V[j];
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

/** Computes and returns the two norm of a vector
 *
 *	\param[in]		v 		the vector
 */
static inline
Real two_norm(const Real* v)
{
	Real norm = ZERO;
	#pragma unroll
	for (uint i = 0; i < NN; ++i) {
		norm += v[i] * v[i];
	}
	return sqrt(norm);
}

/** Normalize the input vector using a 2-norm
 * 
 * \param[in]		v		vector to be normalized
 * \param[out]		v_out	where to stick the normalized part of v (in a column)
 */
static inline
Real normalize (const Real * v, Real* v_out) {
	
	Real norm = two_norm(v);

	#pragma unroll
	for (uint i = 0; i < NN; ++i) {
		v_out[i] = v[i] / norm;
	}
	return norm;
}


/** Performs the dot product of the w vector with the given Matrix
 * 
 * \param[in]		w   	the vector with with to dot
 * \param[in]		Vm		the subspace matrix
 * \out						the dot product of the specified vectors
 */
static inline
Real dotproduct(const Real* w, const Real* Vm)
{
	Real sum = 0;
	#pragma unroll
	for(int i = 0; i < NN; i++)
	{
		sum += w[i] * Vm[i];
	}
	return sum;
}

/** Subtracts column c of Vm scaled by s from w
 * 
 * \param[in]		s   	the scale multiplier to use
 * \param[in]		Vm		the subspace matrix
 * \param[out]		w 		the vector to subtract from
 */
static inline void scale_subtract(const Real s, const Real* Vm, Real* w)
{
	#pragma unroll
	for (int i = 0; i < NN; i++)
	{
		w[i] -= s * Vm[i];
	}
}

/** Sets column c of Vm to s * w
 * 
 * \param[in]		c 		the column of matrix Vm to use
 * \param[in]		stride 	number of columns in Vm
 * \param[in]		s   	the scale multiplier to use
 * \param[in]		w 		the vector to use as a base
 * \param[out]		Vm		the subspace matrix to set
 */
static inline void scale_mult(const Real s, const Real* w, Real* Vm)
{
	#pragma unroll
	for (int i = 0; i < NN; i++)
	{
		Vm[i] = w[i] * s;
	}
}

/** Performs the arnoldi iteration on the matrix A and the vector v using the Niesen scheme
 *
 *	Generates the subspace matrix (Vm) and Hessenberg matrix Hm 
 *
 * \param[in, out]	m 			the size of the subspace (variable)
 * \param[in, out]	h 			the timestep to use (may be variable)
 * \param[in]		h_variable	if true, h may change
 * \param[in]		A   		the jacobian matrix
 * \param[in]		v			the vector for which to determine the subspace
 * \param[out]		Vm			the subspace matrix to set
 * \param[out]		Hm			the hessenberg matrix to set
 */
static inline
Real arnoldi(int* m, Real* h, bool h_variable, const Real tau, const Real t, const Real t_end, const Real* A, const Real* v, Real* Vm, Real* Hm, Real* phiHm)
{
	//the temporary work array
	Real w[NN];
	Real err = 100;
	Real err_old = 0;
	Real h_old = -1;
	int m_old = -1;
	int j = 0;
	Real t_start = t;
	bool h_changed = false;
	bool m_changed = false;

	//first place A*fy in the Vm matrix
	Real beta = normalize(v, Vm);
	Real store = -1;

	//the niesen scheme
	while (err >= DELTA)
	{
		#pragma unroll
		for (; j < *m; j++)
		{
			sparse_multiplier(A, &Vm[j * STRIDE], w);
			for (int i = 0; i <= j; i++)
			{
				Hm[j * STRIDE + i] = dotproduct(w, &Vm[i * STRIDE]);
				scale_subtract(Hm[j * STRIDE + i], &Vm[i * STRIDE], w);
			}
			Hm[j * STRIDE + j + 1] = two_norm(w);
			if (fabs(Hm[j * STRIDE + j + 1]) < ATOL)
			{
				//happy breakdown
				*m = j;
				break;
			}
			scale_mult(ONE / Hm[j * STRIDE + j + 1], w, &Vm[(j + 1) * STRIDE]);
		}
		//resize Hm to be mxm, and store Hm(m, m + 1) for later
		store = Hm[(*m - 1) * STRIDE + *m];
		Hm[(*m - 1) * STRIDE + *m] = ZERO;

		//get error
		//1. Construct augmented Hm (fill in identity matrix)
		Hm[(*m) * STRIDE] = ONE;
		Hm[((*m) + 1) * STRIDE + *m] = ONE;

		//2. Get phiHm
		expAc_variable (*m + P, STRIDE, Hm, *h / 3.0, phiHm);

		//3. Rearrange phiHm
		err = store * phiHm[(*m + P - 1) * STRIDE + (*m - 1)];
		phiHm[(*m) * STRIDE + *m] = err;

		//4. Get error
		err = fabs(3.0 * beta * err / ((*h) * TOL));

		//test error
		if (err >= DELTA)
		{
			//restore Hm(m, m + 1)
			Hm[(*m - 1) * STRIDE + *m] = store;
			//restore Hm(m + 1, m + 2), Hm(0, m + 1) will be cleared automatically next iteration
			Hm[((*m) + 1) * STRIDE + *m] = ZERO;

			int m_new = -1;
			Real h_new = -1;
			//calculate next h and m

			//1. get k and q
			Real q = 0;
			Real k = 2;
			Real omega = (tau) * err / ((*h) * TOL);
			if (h_changed)
			{
				q = log(*h / h_old) / log(err / err_old) - 1.0;
			}
			else
			{
				q = 0.25 * (*m);
			}
			if (m_changed)
			{
				k = pow(err / err_old, 1.0 / (m_old - *m));
			}

			h_new = (*h) * pow(GAMMA / omega, 1.0 / (q + 1.0));
			m_new = (*m) + log(omega / GAMMA) / log(k);

			//store old
			h_old = *h;
			err_old = err;
			m_old = *m;

			if (h_variable)
			{
				//calculate cost functions
				Real C_new_m = ((m_new + P) * N_A + (m_new * m_new + 3.0 * P + 2.0) * NN + (m_new + P + 1.0) * (m_new + P + 1.0) * (m_new + P + 1.0))  * (t_end - t) / *h;
				Real C_new_h = (((*m) + P) * N_A + ((*m) * (*m) + 3.0 * P + 2.0) * NN + ((*m) + P + 1.0) * ((*m) + P + 1.0) * ((*m) + P + 1.0)) * (t_end - t) / h_new;

				if (C_new_m > C_new_h)
				{
					(*h) = fmax(fmin(h_new, 5.0 * (*h)), 0.2 * (*h));
					h_changed = true;
				}
				else
				{
					*m = round(fmin(m_new, 4.0 * (*m) / 3.0));
					m_changed = true;
				}
			}
			else
			{
				*m = round(fmin(m_new, 4.0 * (*m) / 3.0));
				m_changed = true;
			}
		}
	}
	return beta;
}

///////////////////////////////////////////////////////////////////////////////

/** 4th-order exponential integrator function w/ adaptive Kyrlov subspace approximation
 * 
 * 
 */
void exp4_krylov_int (const Real t_start, const Real t_end, const Real pr, Real* y) {
	
	//initial time
	Real h = 1.0e-8;
	Real h_max, h_new;
	
	Real err_old = 1.0;
	Real h_old = h;
	
	bool reject = false;

	Real t = t_start;
	
	while ((t < t_end) && (t + h > t)) {
		//initial krylov subspace sizes
		int m = NN / 3, m1 = NN / 5, m2 = NN / 5;

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

		Real Hm[STRIDE * STRIDE] = {ZERO};
		Real Vm[NN * STRIDE] = {ZERO};
		Real phiHm[STRIDE * STRIDE] = {ZERO};

		//do arnoldi
		Real beta = arnoldi(&m, &h, true, t_end - t_start, t, t_end, A, fy, Vm, Hm, phiHm);

		// k1
		Real k1[NN];
		//k1 is partially in the m'th column of phiHm
		matvec_n_by_m(m + 1, Vm, &phiHm[m * STRIDE], k1);
		#pragma unroll
		for (int i = 0; i < NN; ++i) {
			k1[i] = beta * k1[i] / (h / 3.0);
		}
	
		// k2
		Real k2[NN];
		//computing phi(2h * A)
		matvec_m_by_m (m, phiHm, &phiHm[m * STRIDE], temp);
		matvec_m_by_m (m, Hm, temp, f_temp);
		//note: f_temp contains hm * phi * phi * e1 for later use
		matvec_n_by_m(m, Vm, f_temp, k2);

		#pragma unroll
		for (uint i = 0; i < NN; ++i) {
			k2[i] = beta * (h / 6.0) * k2[i] + k1[i];
		}
	
		// k3
		Real k3[NN];
		//use the stored hm * phi * phi * e1 to get phi(3h * A)
		matvec_m_by_m (m, phiHm, f_temp, temp);
		matvec_m_by_m (m, Hm, temp, f_temp);
		matvec_n_by_m(m, Vm, f_temp, k3);
	
		#pragma unroll
		for (uint i = 0; i < NN; ++i) {
			k3[i] = (2.0 / 9.0) * beta * h * k3[i] + (2.0 / 3.0) * k2[i] + (k1[i] / 3.0);
		}
			
		// d4
		Real k4[NN];
		#pragma unroll
		for (uint i = 0; i < NN; ++i) {
			// f4
			f_temp[i] = h * ((-7.0 / 300.0) * k1[i] + (97.0 / 150.0) * k2[i] - (37.0 / 300.0) * k3[i]);
		
			k4[i] = y[i] + f_temp[i];
		}
		
		dydt (t, pr, k4, temp);
		sparse_multiplier (A, f_temp, k4);
	
		#pragma unroll
		for (uint i = 0; i < NN; ++i) {
			k4[i] = temp[i] - fy[i] - k4[i];
		}

		//do arnoldi
		beta = arnoldi(&m1, &h, false, t_end - t_start, t, t_end, A, k4, Vm, Hm, phiHm);
		//k4 is partially in the m'th column of phiHm
		matvec_n_by_m(m1, Vm, &phiHm[m1 * STRIDE], k4);
		#pragma unroll
		for (uint i = 0; i < NN; ++i) {
			k4[i] = beta * k4[i] / (h / 3.0);
		}
	
		// k5
		Real k5[NN];
		//computing phi(2h * A)
		matvec_m_by_m (m1, phiHm, &phiHm[m * STRIDE], temp);
		matvec_m_by_m (m1, Hm, temp, f_temp);
		//note: f_temp contains hm * phi * phi * e1 for later use
		matvec_n_by_m(m1, Vm, f_temp, k5);
	
		#pragma unroll
		for (uint i = 0; i < NN; ++i) {
			k5[i] = beta * (h / 6.0) * k5[i] + k4[i];
		}
			
		// k6
		Real k6[NN];
		//use the stored hm * phi * phi * e1 to get phi(3h * A)
		matvec_m_by_m (m1, phiHm, f_temp, temp);
		matvec_m_by_m (m1, Hm, temp, f_temp);
		matvec_n_by_m(m1, Vm, f_temp, k6);
	
		#pragma unroll
		for (uint i = 0; i < NN; ++i) {
			k6[i] = (2.0 / 9.0) * beta * h * k6[i] + (2.0 / 3.0) * k5[i] + (k4[i] / 3.0);
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
		sparse_multiplier (A, f_temp, k7);
	
		#pragma unroll
		for (uint i = 0; i < NN; ++i) {
			k7[i] = temp[i] - fy[i] - k7[i];
		}
	
		beta = arnoldi(&m2, &h, false, t_end - t_start, t, t_end, A, k7, Vm, Hm, phiHm);
		//k7 is partially in the m'th column of phiHm
		matvec_n_by_m(m2, Vm, &phiHm[m2 * STRIDE], k7);
		#pragma unroll
		for (uint i = 0; i < NN; ++i) {
			k7[i] = beta * k7[i] / (h / 3.0);
		}
				
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