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
#include <string.h>

#include "header.h"
#include "derivs.h"
#include "phiAHessenberg.h"
#include "sparse_multiplier.h"
#include "inverse.h"


static inline void matvec_m_by_m (const int, const Real *, const Real *, Real *);
static inline void matvec_m_by_m_plusequal (const int, const Real *, const Real *, Real *);
static inline void matvec_n_by_m_scale (const int, const Real, const Real *, const Real *, Real *);
static inline void matvec_n_by_m_scale_add (const int, const Real, const Real *, const Real *, Real *, const Real *);
static inline void matvec_n_by_m_scale_add_subtract (const int, const Real, const Real *, const Real *, Real *, const Real *, const Real *);
static inline void matvec_n_by_m_scale_special (const int, const Real[], const Real*, const Real* [], Real* []);
static inline void matvec_n_by_m_scale_special2 (const int, const Real[], const Real*, const Real* [], Real* []);
static inline void scale (const Real*, const Real*, Real*);
static inline void scale_init (const Real*, Real*);
static inline Real sc_norm (const Real*, const Real*);
static inline double dotproduct(const Real*, const Real*);
static inline Real normalize(const Real*, Real*);
static inline void scale_subtract(const Real, const Real*, Real*);
static inline Real two_norm(const Real*);
static inline void scale_mult(const Real, const Real*, Real*);
static inline Real arnoldi(int*, const Real, const int, const Real, const Real*, const Real*, const Real*, Real*, Real*, Real*, Real*);

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
//max order of the phi functions (i.e. for error estimation)
#define P 4
//order of embedded methods
#define ORD 3
//indexed list
static int index_list[23] = {1, 2, 3, 4, 5, 6, 7, 9, 11, 14, 17, 21, 27, 34, 42, 53, 67, 84, 106, 133, 167, 211, 265};
#define M_u 2
#define M_opt 8
#define M_MAX 20
//max size of arrays
#define STRIDE (M_MAX + P)
//if defined, uses (I - h * Hm)^-1 to smooth the krylov error vector
//#define USE_SMOOTHED_ERROR

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

/** Matrix-vector plus equals for a matrix of size MxM and vector of size Mx1
 * 
 *  That is, it returns (A + I) * v
 *
 * Performs inline matrix-vector multiplication (with unrolled loops) 
 * 
 * \param[in]		m 		size of the matrix
 * \param[in]		A		matrix of size MxM
 * \param[in]		V		vector of size Mx1
 * \param[out]		Av		vector that is (A + I) * v
 */
static inline void matvec_m_by_m_plusequal (const int m, const Real * A, const Real * V, Real * Av)
{
	//for each row
	#pragma unroll
	for (int i = 0; i < m; ++i) {
		Av[i] = ZERO;
		
		//go across a row of A, multiplying by a column of phiHm
		#pragma unroll
		for (int j = 0; j < m; ++j) {
			Av[i] += A[j * STRIDE + i] * V[j];
		}

		Av[i] += V[i];
	}
}

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
			Av[i] += A[j * NN + i] * V[j];
		}

		Av[i] *= scale;
	}
}


/** Matrix-vector multiplication of a matrix sized NNxM and a vector of size Mx1 scaled by a specified factor
 *
 *  Computes the following:
 *  Av1 = A * V1 * scale[0]
 *  Av2 = A * V2 * scale[1]
 *  Av3 = A * V3 * scale[2] + V4 + V5
 * 
 * Performs inline matrix-vector multiplication (with unrolled loops)
 * 
 * \param[in]		m 		size of the matrix
 * \param[in]		scale 	a list of numbers to scale the multplication by
 * \param[in]		A		matrix
 * \param[in]		V		a list of 5 pointers corresponding to V1, V2, V3, V4, V5
 * \param[out]		Av		a list of 3 pointers corresponding to Av1, Av2, Av3
 */
static inline
void matvec_n_by_m_scale_special (const int m, const Real scale[], const Real * A, const Real* V[], Real* Av[]) {
	//for each row
	#pragma unroll
	for (int i = 0; i < NN; ++i) {
		#pragma unroll
		for (int k = 0; k < 3; k++)
		{
			Av[k][i] = ZERO;
		}
		
		//go across a row of A, multiplying by a column of phiHm
		#pragma unroll
		for (int j = 0; j < m; ++j) {
			#pragma unroll
			for (int k = 0; k < 3; k++)
			{
				Av[k][i] += A[j * NN + i] * V[k][j];
			}
		}

		#pragma unroll
		for (int k = 0; k < 3; k++)
		{
			Av[k][i] *= scale[k];
		}
		Av[2][i] += V[3][i];
		Av[2][i] += V[4][i];
	}
}

/** Matrix-vector multiplication of a matrix sized NNxM and a vector of size Mx1 scaled by a specified factor
 *
 *  Computes the following:
 *  Av1 = A * V1 * scale[0]
 *  Av2 = A * V2 * scale[1]
 * 
 * Performs inline matrix-vector multiplication (with unrolled loops)
 * 
 * \param[in]		m 		size of the matrix
 * \param[in]		scale 	a list of numbers to scale the multplication by
 * \param[in]		A		matrix
 * \param[in]		V		a list of 2 pointers corresponding to V1, V2
 * \param[out]		Av		a list of 2 pointers corresponding to Av1, Av2
 */
static inline
void matvec_n_by_m_scale_special2 (const int m, const Real scale[], const Real * A, const Real* V[], Real* Av[]) {
	//for each row
	#pragma unroll
	for (int i = 0; i < NN; ++i) {
		#pragma unroll
		for (int k = 0; k < 2; k++)
		{
			Av[k][i] = ZERO;
		}
		
		//go across a row of A, multiplying by a column of phiHm
		#pragma unroll
		for (int j = 0; j < m; ++j) {
			#pragma unroll
			for (int k = 0; k < 2; k++)
			{
				Av[k][i] += A[j * NN + i] * V[k][j];
			}
		}

		#pragma unroll
		for (int k = 0; k < 2; k++)
		{
			Av[k][i] *= scale[k];
		}
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
			Av[i] += A[j * NN + i] * V[j];
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
 * \param[in]		scale  		the factor to scale the timestep by
 * \param[in]		h 			the timestep to use
 * \param[in]		p 			the maximum phi order needed (note, order 1 is used to evaluate error)
 * \param[in]		A   		the jacobian matrix
 * \param[in]		v			the vector for which to determine the subspace
 * \param[in]		sc 			the scaled weighted norm vector to use for error control
 * \param[out]		beta		the two norm of the v vector
 * \param[out]		Vm			the subspace matrix to set
 * \param[out]		Hm			the hessenberg matrix to set
 * \param[out]		phiHm		the resulting phi function of the hessenberg matrix
 */
static inline
Real arnoldi(int* m, const Real scale, const int p, const Real h, const Real* A, const Real* v, const Real* sc, Real* beta, Real* Vm, Real* Hm, Real* phiHm)
{
	//the temporary work array
	Real w[NN];

	//first place A*fy in the Vm matrix
	*beta = normalize(v, Vm);

	Real store = 0;
	int index = 0;
	int j = 0;
	Real err = 0;
	int order = p < 1 ? 1 : p;

	do
	{
		if (j >= M_MAX) //need to modify h_kry and restart
		{
			break;
		}
		#pragma unroll
		for (; j < index_list[index]; j++)
		{
			sparse_multiplier(A, &Vm[j * NN], w);
			for (int i = 0; i <= j; i++)
			{
				Hm[j * STRIDE + i] = dotproduct(w, &Vm[i * NN]);
				scale_subtract(Hm[j * STRIDE + i], &Vm[i * NN], w);
			}
			Hm[j * STRIDE + j + 1] = two_norm(w);
			if (fabs(Hm[j * STRIDE + j + 1]) < ATOL)
			{
				//happy breakdown
				*m = j;
				break;
			}
			scale_mult(ONE / Hm[j * STRIDE + j + 1], w, &Vm[(j + 1) * NN]);
		}
		*m = index_list[index++];
		//resize Hm to be mxm, and store Hm(m, m + 1) for later
		store = Hm[(*m - 1) * STRIDE + *m];
		Hm[(*m - 1) * STRIDE + *m] = ZERO;

		//0. fill potentially non-empty memory first
		memset(&Hm[*m * STRIDE], 0, (*m + 1) * sizeof(Real)); 

		//get error
		//1. Construct augmented Hm (fill in identity matrix)
		Hm[(*m) * STRIDE] = ONE;
		#pragma unroll
		for (int i = 1; i < order; i++)
		{
			//0. fill potentially non-empty memory first
			memset(&Hm[(*m + i) * STRIDE], 0, (*m + i + 1) * sizeof(Real));
			Hm[(*m + i) * STRIDE + (*m + i - 1)] = ONE;
		}

		//2. Get phiHm
		expAc_variable (*m + order, STRIDE, Hm, h * scale, phiHm);

		//3. Get error

		#ifdef USE_SMOOTHED_ERROR
			if (*m > 4)
			{
				//use the modified err from Hochbruck et al. 

				//setup I - h*Hm
				Real* working = (Real*)malloc((*m) * (*m) * sizeof(Real));
				#pragma unroll
				for (int ind1 = 0; ind1 < *m; ind1++)
				{
					#pragma unroll
					for (int ind2 = 0; ind2 < *m; ind2++)
					{
						if (ind1 == ind2)
						{
							working[ind1 * (*m) + ind2] = ONE - h * scale * Hm[ind1 * STRIDE + ind2];
						}
						else
						{
							working[ind1 * (*m) + ind2] = -h * scale * Hm[ind1 * STRIDE + ind2];
						}
					}
				}
				getInverseHessenberg(*m, working);
				//get the value for the err (dot product of mth row of working and 1'st col of Hm)
				Real val = 0;
				#pragma unroll
				for (uint ind1 = 0; ind1 < *m; ind1++)
				{
					val += working[(*m) * ind1 + (*m - 1)] * Hm[ind1];
				}
				err = h * (*beta) * fabs(store * val) * sc_norm(&Vm[(*m) * NN], sc);

				free(working);
			}
			else
			{
				err = h * (*beta) * fabs(store * phiHm[(*m) * STRIDE + (*m) - 1]) * sc_norm(&Vm[(*m) * NN], sc);
			}
		#else
			err = h * (*beta) * fabs(store * phiHm[(*m) * STRIDE + (*m) - 1]) * sc_norm(&Vm[(*m) * NN], sc);
		#endif

		//restore Hm(m, m + 1)
		Hm[(*m - 1) * STRIDE + *m] = store;

	} while (err >= ONE);

	return j;
}

///////////////////////////////////////////////////////////////////////////////

/** 4th-order exponential integrator function w/ adaptive Kyrlov subspace approximation
 * 
 * 
 */
void exprb43_int (const Real t_start, const Real t_end, const Real pr, Real* y) {
	
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
    	Real gy[NN];
    	//gy = fy - A * y
    	sparse_multiplier(A, y, gy);
    	#pragma unroll
    	for (uint i = 0; i < NN; ++i) {
    		gy[i] = fy[i] - gy[i];
    	}

		Real Hm[STRIDE * STRIDE] = {ZERO};
		Real Vm[NN * STRIDE] = {ZERO};
		Real phiHm[STRIDE * STRIDE] = {ZERO};
		Real err = ZERO;
		Real savedActions[NN * 5];

		do
		{
			//do arnoldi
			if (arnoldi(&m, 0.5, 1, h, A, fy, sc, &beta, Vm, Hm, phiHm) >= M_MAX)
			{
				//need to reduce h and try again
				h /= 3;
				continue;
			}

			// Un2 to be stored in temp
			//Un2 is partially in the mth column of phiHm
			//Un2 = y + ** 0.5 * h * phi_1(0.5 * h * A)*fy **
			//Un2 = y + ** beta * Vm * phiHm(:, m) **

			//store h * beta * Vm * phi_1(h * Hm) * e1 in savedActions
			matvec_m_by_m_plusequal(m, phiHm, &phiHm[m * STRIDE], temp);
			matvec_n_by_m_scale(m, beta, Vm, temp, savedActions);

			//store 0.5 * h *  beta * Vm * phi_1(0.5 * h * Hm) * fy in temp
			matvec_n_by_m_scale_add(m, beta, Vm, &phiHm[m * STRIDE], temp, y);
			//temp is now equal to Un2

			//next compute Dn2
			//Dn2 = (F(Un2) - Jn * Un2) - gy

			dydt(t, pr, temp, &savedActions[NN]);
			sparse_multiplier(A, temp, f_temp);

			#pragma unroll
			for (uint i = 0; i < NN; ++i) {
				temp[i] = savedActions[NN + i] - f_temp[i] - gy[i]; 
			}
			//temp is now equal to Dn2

			//partially compute Un3 as:
			//Un3 = y + ** h * phi_1(hA) * fy ** + h * phi_1(hA) * Dn2
			//Un3 = y + ** h * beta * Vm * phiHm(:, m) **

			//now we need the action of the exponential on Dn2
			if (arnoldi(&m1, 1.0, 4, h, A, temp, sc, &beta, Vm, Hm, phiHm) >= M_MAX)
			{
				//need to reduce h and try again
				h /= 3;
				continue;
			}

			//save Phi3(h * A) * Dn2 to savedActions[0]
			//save Phi4(h * A) * Dn2 to savedActions[NN]
			//add the action of phi_1 on Dn2 to y and hn * phi_1(hA) * fy to get Un3
			const Real* in[5] = {&phiHm[(m1 + 2) * STRIDE], &phiHm[(m1 + 3) * STRIDE], &phiHm[m1 * STRIDE], savedActions, y};
			Real* out[3] = {&savedActions[NN], &savedActions[2 * NN], temp};
			Real scale_vec[3] = {beta / (h * h), beta / (h * h * h), beta};
			matvec_n_by_m_scale_special(m1, scale_vec, Vm, in, out);
			//Un3 is now in temp

			//next compute Dn3
			//Dn3 = F(Un3) - A * Un3 - gy
			dydt(t, pr, temp, &savedActions[3 * NN]);
			sparse_multiplier(A, temp, f_temp);

			#pragma unroll
			for (uint i = 0; i < NN; ++i) {
				temp[i] = savedActions[3 * NN + i] - f_temp[i] - gy[i]; 
			}
			//temp is now equal to Dn3

			//finally we need the action of the exponential on Dn3
			if (arnoldi(&m2, 1.0, 4, h, A, temp, sc, &beta, Vm, Hm, phiHm) >= M_MAX)
			{
				//need to reduce h and try again
				h /= 3;
				continue;
			}
			out[0] = &savedActions[3 * NN];
			out[1] = &savedActions[4 * NN];
			in[0] = &phiHm[(m2 + 2) * STRIDE];
			in[1] = &phiHm[(m2 + 3) * STRIDE];
			scale_vec[0] = beta / (h * h);
			scale_vec[1] = beta / (h * h * h);
			matvec_n_by_m_scale_special2(m2, scale_vec, Vm, in, out);

			//construct y1 and error vector
			#pragma unroll
			for (uint i = 0; i < NN; ++i) {
				//y1 = y + h * phi1(h * A) * fy + h * sum(bi * Dni)
				y1[i] = y[i] + savedActions[i] + 16.0 * savedActions[NN + i] - 48.0 * savedActions[2 * NN + i] + -2.0 * savedActions[3 * NN + i] + 12.0 * savedActions[4 * NN + i];
				//error vec
				temp[i] = y[i] + 16.0 * savedActions[NN + i] -2.0 * savedActions[3 * NN + i];
			}


			//scale and find err
			scale (y, y1, sc);
			err = fmax(EPS, h * sc_norm(temp, sc));
			
			// classical step size calculation
			h_new = pow(err, -1.0 / ORD);	

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
				h = fmin(h_new, fabs(t_end - t));
							
			} else {

				// limit to 0.2 <= (h_new/8) <= 8.0
				h_new = h * fmax(fmin(0.9 * h_new, 8.0), 0.2);
				h_new = fmin(h_new, h_max);
				
				reject = true;
				h = fmin(h, h_new);
			}
		} while(err >= ONE);

	} // end while

	#ifdef LOG_KRYLOV_AND_STEPSIZES
		fclose(logFile);
	#endif
	
}