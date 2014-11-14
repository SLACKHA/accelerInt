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
#include "derivs.h"
#include "phiAHessenberg.h"
#include "sparse_multiplier.h"


static inline void matvec_m_by_m (const int, const Real *, const Real *, Real *);
static inline void matvec_n_by_m_scale (const int, const Real, const Real *, const Real *, Real *);
static inline void matvec_n_by_m_scale_add (const int, const Real, const Real *, const Real *, Real *, const Real *);
static inline void matvec_n_by_m_scale_add_subtract (const int, const Real, const Real *, const Real *, Real *, const Real *, const Real *);
static inline void scale (const Real*, const Real*, Real*);
static inline void scale_init (const Real*, Real*);
static inline Real sc_norm (const Real*, const Real*);
static inline double dotproduct(const Real*, const Real*);
static inline Real normalize(const Real*, Real*);
static inline void scale_subtract(const Real, const Real*, Real*);
static inline Real two_norm(const Real*);
static inline void scale_mult(const Real, const Real*, Real*);
static inline Real arnoldi(int*, bool, const Real, const Real*, const Real*, const Real*, Real*, Real*, Real*, Real*);

#ifdef COMPILE_TESTING_METHODS
void matvec_m_by_m_test (const int i , const Real * j, const Real * k, Real * l) {
	matvec_m_by_m(i, j, k, l);
}
void matvec_n_by_m_scale_test (const int i, const Real m, const Real * j, const Real * k, Real *l){
	matvec_n_by_m_scale(i, m, j, k, l);
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

//whether to use S3 error
//#define USE_S3_ERR
//max order of the phi functions (i.e. for error estimation)
#ifdef USE_S3_ERR
	#define P 2
#else
	#define P 1
#endif
//order of embedded methods
#define ORD 3
//indexed list
static int index_list[23] = {1, 2, 3, 4, 5, 6, 7, 9, 11, 14, 17, 21, 27, 34, 42, 53, 67, 84, 106, 133, 167, 211, 265};
#define M_u 2
#define M_opt 8
#define M_MAX 20
//max size of arrays
#define STRIDE (M_MAX + P)

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

/** Matrix-vector multiplication of a matrix sized NNxM and a vector of size Mx1 scaled by a specified factor
 * 
 * Performs inline matrix-vector multiplication (with unrolled loops)
 * 
 * \param[in]		m 		size of the matrix
 * \param[in]		scale 	a number to scale the multplication by
 * \param[in]		A		matrix
 * \param[in]		V		the vector
 * \param[out]		Av		vector that is A * V
 */
static inline
void matvec_n_by_m_scale (const int m, const Real scale, const Real * A, const Real * V, Real * Av) {
	//for each row
	#pragma unroll
	for (int i = 0; i < NN; ++i) {
		Av[i] = ZERO;
		
		//go across a row of A, multiplying by a column of phiHm
		#pragma unroll
		for (int j = 0; j < m; ++j) {
			Av[i] += A[j * STRIDE + i] * V[j];
		}

		Av[i] *= scale;
	}
}

///////////////////////////////////////////////////////////////////////////////

/** Matrix-vector multiplication of a matrix sized NNxM and a vector of size Mx1 scaled by a specified factor and added to another vector
 * 
 * Performs inline matrix-vector multiplication (with unrolled loops)
 * 
 * \param[in]		m 		size of the matrix
 * \param[in]		scale 	a number to scale the multplication by
 * \param[in]		add 	the vector to add to the result
 * \param[in]		A		matrix
 * \param[in]		V		the vector
 * \param[out]		Av		vector that is A * V
 */
static inline
void matvec_n_by_m_scale_add (const int m, const Real scale, const Real * A, const Real * V, Real * Av, const Real* add) {
	//for each row
	#pragma unroll
	for (int i = 0; i < NN; ++i) {
		Av[i] = ZERO;
		
		//go across a row of A, multiplying by a column of phiHm
		#pragma unroll
		for (int j = 0; j < m; ++j) {
			Av[i] += A[j * STRIDE + i] * V[j];
		}

		Av[i] = Av[i] * scale + add[i];
	}
}

///////////////////////////////////////////////////////////////////////////////

/** Matrix-vector multiplication of a matrix sized NNxM and a vector of size Mx1 scaled by a specified factor and adds and subtracts the specified vectors
 *  note, the addition is twice the specified vector
 * 
 * Performs inline matrix-vector multiplication (with unrolled loops)
 * 
 * \param[in]		m 		size of the matrix
 * \param[in]		scale 	a number to scale the multplication by
 * \param[in]		add 	the vector to add to the result
 * \param[]
 * \param[in]		A		matrix
 * \param[in]		V		the vector
 * \param[out]		Av		vector that is A * V
 */
static inline
void matvec_n_by_m_scale_add_subtract (const int m, const Real scale, const Real * A, const Real * V, Real * Av, const Real* add, const Real * sub) {
	//for each row
	#pragma unroll
	for (int i = 0; i < NN; ++i) {
		Av[i] = ZERO;
		
		//go across a row of A, multiplying by a column of phiHm
		#pragma unroll
		for (int j = 0; j < m; ++j) {
			Av[i] += A[j * STRIDE + i] * V[j];
		}

		Av[i] = Av[i] * scale + 2.0 * add[i] - sub[i];
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

/** Get scaling for weighted norm for the initial timestep (used in krylov process)
 * 
 * \param[in]		y0		values at current timestep
 * \param[out]	sc	array of scaling values
 */
static inline
void scale_init (const Real * y0, Real * sc) {
	#pragma unroll
	for (uint i = 0; i < NN; ++i) {
		sc[i] = ATOL + fabs(y0[i]) * RTOL;
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
 *  Returns h_kry the necessary step size to maintain accuracy given a maximum Krylov subspace size of M_MAX
 *
 * \param[in, out]	m 			the size of the subspace (variable)
 * \param[in]		h_changable	true if this is the first krylov projection, and h can be changed
 * \param[in]		h 			the timestep to use
 * \param[in]		A   		the jacobian matrix
 * \param[in]		v			the vector for which to determine the subspace
 * \param[in]		sc 			the scaled weighted norm vector to use for error control
 * \param[out]		beta		the two norm of the v vector
 * \param[out]		Vm			the subspace matrix to set
 * \param[out]		Hm			the hessenberg matrix to set
 * \param[out]		phiHm		the resulting phi function of the hessenberg matrix
 */
static inline
Real arnoldi(int* m, bool h_changable, const Real h, const Real* A, const Real* v, const Real* sc, Real* beta, Real* Vm, Real* Hm, Real* phiHm)
{
	//the temporary work array
	Real w[NN];

	//first place A*fy in the Vm matrix
	*beta = normalize(v, Vm);

	Real store = 0;
	int index = 0;
	int j = 0;
	Real err = 0;
	Real h_kry = h;

	do
	{
		if (j >= M_MAX && h_changable) //need to modify h_kry and restart
		{
			h_kry /= 3.0;
			j = 0;
		}
		#pragma unroll
		for (; j < index_list[index]; j++)
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
		*m = index_list[index++];
		//resize Hm to be mxm, and store Hm(m, m + 1) for later
		store = Hm[(*m - 1) * STRIDE + *m];
		Hm[(*m - 1) * STRIDE + *m] = ZERO;

		//get error
		//1. Construct augmented Hm (fill in identity matrix)
		Hm[(*m) * STRIDE] = ONE;

		#ifdef USE_S3_ERR
			if (*m < 5)
			{
				//construct identity matrix
				Hm[(*m + 1) * STRIDE + (*m)] = ONE;
			}
		#endif

		//2. Get phiHm
		phiAc_variable (*m + P, STRIDE, Hm, h / 3.0, phiHm);

		//3. Get error
		err = h * (*beta) * fabs(store * phiHm[(*m) * STRIDE + (*m) - 1]) * sc_norm(&Vm[(*m) * STRIDE], sc);

		#ifdef USE_S3_ERR
			if (*m < 5)
			{
				//restore
				Hm[(*m + 1) * STRIDE + (*m)] = ZERO;
				//get s3 err
				sparse_multiplier(A, &Vm[(*m) * STRIDE], w);
				err = fmax(h * (*beta) * fabs(store * phiHm[(*m + 1) * STRIDE + (*m) - 1]) * sc_norm(w, sc), err);
			}
		#endif

		//restore Hm(m, m + 1)
		Hm[(*m - 1) * STRIDE + *m] = store;
		//restore real Hm
		Hm[(*m) * STRIDE] = ZERO;
	} while (err >= ONE);

	return h_kry;
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

	// get scaling for weighted norm
	Real sc[NN];
	scale_init(y, sc);

	#ifdef LOG_KRYLOV_AND_STEPSIZES
	  //file for krylov logging
	  FILE *logFile;
	  //open and clear
	  logFile = fopen("log.txt", "a");
  	#endif
	
	int small_count = 0, mu_count = 0;
	Real beta = 0;
	while ((t < t_end) && (t + h > t)) {
		//initial krylov subspace sizes
		int m, m1, m2;

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
		Real err = ZERO;

		do
		{
			Real h_kry = h;
			//do arnoldi
			h_kry = arnoldi(&m, true, h, A, fy, sc, &beta, Vm, Hm, phiHm);
			bool h_change = h_kry != h;
			if (h_change)
				h = h_kry;

			// k1
			Real k1[NN];
			//k1 is partially in the first column of phiHm
			//k1 = beta * Vm * phiHm(:, 1)
			matvec_n_by_m_scale(m, beta, Vm, phiHm, k1);
		
			// k2
			Real k2[NN];
			//computing phi(2h * A)
			matvec_m_by_m (m, phiHm, phiHm, temp);
			//note: f_temp will contain hm * phi * phi * e1 for later use
			matvec_m_by_m (m, Hm, temp, f_temp);
			matvec_n_by_m_scale_add(m, beta * (h / 6.0), Vm, f_temp, k2, k1);
		
			// k3
			Real k3[NN];
			//use the stored hm * phi * phi * e1 to get phi(3h * A)
			matvec_m_by_m (m, phiHm, f_temp, temp);
			matvec_m_by_m (m, Hm, temp, f_temp);
			matvec_n_by_m_scale_add_subtract(m, beta * (h * h / 27.0), Vm, f_temp, k3, k2, k1);
				
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
			arnoldi(&m1, false, h, A, k4, sc, &beta, Vm, Hm, phiHm);
			//k4 is partially in the m'th column of phiHm
			matvec_n_by_m_scale(m1, beta, Vm, phiHm, k4);
		
			// k5
			Real k5[NN];
			//computing phi(2h * A)
			matvec_m_by_m (m1, phiHm, phiHm, temp);
			//note: f_temp will contain hm * phi * phi * e1 for later use
			matvec_m_by_m (m1, Hm, temp, f_temp);
			matvec_n_by_m_scale_add(m1, beta * (h / 6.0), Vm, f_temp, k5, k4);
				
			// k6
			Real k6[NN];
			//use the stored hm * phi * phi * e1 to get phi(3h * A)
			matvec_m_by_m (m1, phiHm, f_temp, temp);
			matvec_m_by_m (m1, Hm, temp, f_temp);
			matvec_n_by_m_scale_add_subtract(m1, beta * (h * h / 27.0), Vm, f_temp, k6, k5, k4);
				
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
		
			arnoldi(&m2, false, h, A, k7, sc, &beta, Vm, Hm, phiHm);
			//k7 is partially in the m'th column of phiHm
			matvec_n_by_m_scale(m2, beta / (h / 3.0), Vm, &phiHm[m2 * STRIDE], k7);
					
			// y_n+1
			#pragma unroll
			for (uint i = 0; i < NN; ++i) {
				y1[i] = y[i] + h * (k3[i] + k4[i] - (4.0 / 3.0) * k5[i] + k6[i] + (1.0 / 6.0) * k7[i]);
			}
			
			scale (y, y1, sc);	
			
			///////////////////
			// calculate errors
			///////////////////
		
			// error of embedded order 3 method
			#pragma unroll
			for (uint i = 0; i < NN; ++i) {
				temp[i] = k3[i] - (2.0 / 3.0) * k5[i] + 0.5 * (k6[i] + k7[i] - k4[i]) - (y1[i] - y[i]) / h;
			}	
			err = h * sc_norm(temp, sc);
			
			// error of embedded W method
			#pragma unroll
			for (uint i = 0; i < NN; ++i) {
				temp[i] = -k1[i] + 2.0 * k2[i] - k4[i] + k7[i] - (y1[i] - y[i]) / h;
			}
			//Real err_W = h * sc_norm(temp, sc);
			err = fmax(EPS, fmin(err, h * sc_norm(temp, sc)));
			
			// classical step size calculation
			h_new = pow(err, -1.0 / ORD);	

			//take care of h_kry updating

			Real temp_kry = 0;
			//m < mu step
			if (m < M_u)
				mu_count++;
			else
				mu_count = 0;
			if (mu_count > 1)
			{
				temp_kry = h * pow(((Real)M_opt) / ((Real)m), 1.0 / 3.0);
			}

			//m small step
			if (m < 4)
				small_count++;
			else
				small_count = 0;
			if (small_count > 1)
			{
				temp_kry = fmax(temp_kry, (1 << (small_count - 1)) * h);
			}

			if (temp_kry == 0)
				temp_kry = h_kry;

			if (!h_change)
				h_kry = temp_kry;
			else
			{
				h_kry = fmin(h_kry, temp_kry);
			}

			#ifdef LOG_KRYLOV_AND_STEPSIZES
				fprintf (logFile, "%e\t%e\t%e\t%d\t%d\t%d\n", t, h, err, m, m1, m2);
	  		#endif
			
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
				//krylov step
				h_new = fmin(h_new, h_kry);
				h = fmin(h_new, fabs(t_end - t));
							
			} else {

				// limit to 0.2 <= (h_new/8) <= 8.0
				h_new = h * fmax(fmin(0.9 * h_new, 8.0), 0.2);
				h_new = fmin(h_new, h_max);

				//krylov step
				h_new = fmin(h_new, h_kry);
				
				reject = true;
				h = fmin(h, h_new);
			}
		} while(err >= ONE);

	} // end while

	#ifdef LOG_KRYLOV_AND_STEPSIZES
		fclose(logFile);
	#endif
	
}