/*!
 * \file exprb43.c
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
#include <string.h>

#include "header.h"
#include "dydt.h"
#include "jacob.h"
#include "exprb43_props.h"
#include "arnoldi.h"
#include "exponential_linear_algebra.h"
#include "solver_init.h"

#ifdef GENERATE_DOCS
namespace exprb43 {
#endif

///////////////////////////////////////////////////////////////////////////////

/*!
 * \fn int integrate(const double t_start, const double t_end, const double pr, double* y)
 * \param t_start The initial integration time
 * \param t_end The final integration timestep
 * \param pr User data passed to the RHS function dydt() - commonly used for the Pressure term
 * \param y The state vector
 * \brief 4th-order exponential integrator function w/ adaptive Kyrlov subspace approximation
 * \returns The result of this integration step @see exprb43_ErrCodes
 */
int integrate (const double t_start, const double t_end, const double pr, double* y) {

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
	int failures = 0;
	int steps = 0;

	double t = t_start;

	// get scaling for weighted norm
	double sc[NSP];
	scale_init(y, sc);

	#ifdef LOG_KRYLOV_AND_STEPSIZES
	  //file for krylov logging
	  FILE *logFile;
	  //open and clear
	  const char* f_name = solver_name();
	  int len = strlen(f_name);
      char out_name[len + 17];
      sprintf(out_name, "log/%s-kry-log.txt", f_name);
      logFile = fopen(out_name, "a");

      char out_reject_name[len + 23];
      sprintf(out_reject_name, "log/%s-kry-reject.txt", f_name);
	  //file for krylov logging
	  FILE *rFile;
	  //open and clear
	  rFile = fopen(out_reject_name, "a");
  	#endif

	double beta = 0;

	// temporary arrays
	double temp[NSP];
	double f_temp[NSP];
	double y1[NSP];

	// source vector
	double fy[NSP];

	// Jacobian matrix
	double A[NSP * NSP] = {0.0};
	double gy[NSP];

	double Hm[STRIDE * STRIDE] = {0.0};
	double Vm[NSP * STRIDE];
	double phiHm[STRIDE * STRIDE];
	double err = 0.0;
	double savedActions[NSP * 5];
	int numSteps = 0;
	while ((t < t_end) && (t + h > t)) {

		//error checking
		if (failures >= 5)
		{
			return EC_consecutive_steps;
		}
		if (steps++ >= MAX_STEPS)
		{
			return EC_max_steps_exceeded;
		}
		if (t + h <= t)
		{
			return EC_h_plus_t_equals_h;
		}

		if (!reject) {
			dydt (t, pr, y, fy);
			eval_jacob (t, pr, y, A);
			//gy = fy - A * y
			sparse_multiplier(A, y, gy);

			for (int i = 0; i < NSP; ++i) {
				gy[i] = fy[i] - gy[i];
			}
		}

		//do arnoldi
		int m = arnoldi(0.5, 1, h, A, fy, sc, &beta, Vm, Hm, phiHm);
		if (m + 1 >= STRIDE || m < 0)
		{
			//need to reduce h and try again
			h /= 5.0;
			failures++;
			reject = true;
			continue;
		}

		// Un2 to be stored in temp
		//Un2 is partially in the mth column of phiHm
		//Un2 = y + ** 0.5 * h * phi_1(0.5 * h * A)*fy **
		//Un2 = y + ** beta * Vm * phiHm(:, m) **

		//store h * beta * Vm * phi_1(h * Hm) * e1 in savedActions
		matvec_m_by_m_plusequal(m, phiHm, &phiHm[m * STRIDE], temp);
		matvec_n_by_m_scale(m, beta, Vm, temp, savedActions);

		//store 0.5 * h *  beta * Vm * phi_1(0.5 * h * Hm) * fy + y in temp
		matvec_n_by_m_scale_add(m, beta, Vm, &phiHm[m * STRIDE], temp, y);
		//temp is now equal to Un2

		//next compute Dn2
		//Dn2 = (F(Un2) - Jn * Un2) - gy

		dydt(t, pr, temp, &savedActions[NSP]);
		sparse_multiplier(A, temp, f_temp);


		for (int i = 0; i < NSP; ++i) {
			temp[i] = savedActions[NSP + i] - f_temp[i] - gy[i];
		}
		//temp is now equal to Dn2

		//partially compute Un3 as:
		//Un3 = y + ** h * phi_1(hA) * fy ** + h * phi_1(hA) * Dn2
		//Un3 = y + ** h * beta * Vm * phiHm(:, m) **

		//now we need the action of the exponential on Dn2
		int m1 = arnoldi(1.0, 4, h, A, temp, sc, &beta, Vm, Hm, phiHm);
		if (m1 + 4 >= STRIDE || m1 < 0)
		{
			//need to reduce h and try again
			h /= 5.0;
			failures++;
			reject = true;
			continue;
		}

		//save Phi3(h * A) * Dn2 to savedActions[0]
		//save Phi4(h * A) * Dn2 to savedActions[NSP]
		//add the action of phi_1 on Dn2 to y and hn * phi_1(hA) * fy to get Un3
		const double* in[5] = {&phiHm[(m1 + 2) * STRIDE], &phiHm[(m1 + 3) * STRIDE], &phiHm[m1 * STRIDE], savedActions, y};
		double* out[3] = {&savedActions[NSP], &savedActions[2 * NSP], temp};
		double scale_vec[3] = {beta / (h * h), beta / (h * h * h), beta};
		matvec_n_by_m_scale_special(m1, scale_vec, Vm, in, out);
		//Un3 is now in temp

		//next compute Dn3
		//Dn3 = F(Un3) - A * Un3 - gy
		dydt(t, pr, temp, &savedActions[3 * NSP]);
		sparse_multiplier(A, temp, f_temp);


		for (int i = 0; i < NSP; ++i) {
			temp[i] = savedActions[3 * NSP + i] - f_temp[i] - gy[i];
		}
		//temp is now equal to Dn3

		//finally we need the action of the exponential on Dn3
		int m2 = arnoldi(1.0, 4, h, A, temp, sc, &beta, Vm, Hm, phiHm);
		if (m2 + 4 >= STRIDE || m2 < 0)
		{
			//need to reduce h and try again
			h /= 5.0;
			failures++;
			reject = true;
			continue;
		}

		out[0] = &savedActions[3 * NSP];
		out[1] = &savedActions[4 * NSP];
		in[0] = &phiHm[(m2 + 2) * STRIDE];
		in[1] = &phiHm[(m2 + 3) * STRIDE];
		scale_vec[0] = beta / (h * h);
		scale_vec[1] = beta / (h * h * h);
		matvec_n_by_m_scale_special2(m2, scale_vec, Vm, in, out);

		//construct y1 and error vector

		for (int i = 0; i < NSP; ++i) {
			//y1 = y + h * phi1(h * A) * fy + h * sum(bi * Dni)
			y1[i] = y[i] + savedActions[i] + 16.0 * savedActions[NSP + i] - 48.0 * savedActions[2 * NSP + i] + -2.0 * savedActions[3 * NSP + i] + 12.0 * savedActions[4 * NSP + i];
			//error vec
			temp[i] = 48.0 * savedActions[2 * NSP + i] - 12.0 * savedActions[4 * NSP + i];
		}


#ifndef CONST_TIME_STEP
		//scale and find err
		scale (y, y1, f_temp);
		err = fmax(EPS, sc_norm(temp, f_temp));

		// classical step size calculation
		h_new = pow(err, -1.0 / ORD);

		failures = 0;
		if (err <= 1.0) {

			#ifdef LOG_KRYLOV_AND_STEPSIZES
				fprintf (logFile, "%.15le\t%.15le\t%.15le\t%d\t%d\t%d\n", t, h, err, m, m1, m2);
  			#endif

			memcpy(sc, f_temp, NSP * sizeof(double));

			// minimum of classical and Gustafsson step size prediction
			h_new = fmin(h_new, (h / h_old) * pow((err_old / (err * err)), (1.0 / ORD)));

			// limit to 0.2 <= (h_new/8) <= 8.0
			h_new = h * fmax(fmin(0.9 * h_new, 8.0), 0.2);

			// update y and t

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
			numSteps++;

		} else {

			#ifdef LOG_KRYLOV_AND_STEPSIZES
				fprintf (rFile, "%.15le\t%.15le\t%.15le\t%d\t%d\t%d\n", t, h, err, m, m1, m2);
  			#endif

			// limit to 0.2 <= (h_new/8) <= 8.0
			h_new = h * fmax(fmin(0.9 * h_new, 8.0), 0.2);
			h_new = fmin(h_new, t_end - t);

			reject = true;
			h = fmin(h, h_new);
		}
#else
		//constant time stepping
		// update y and t
		for (int i = 0; i < NSP; ++i) {
			y[i] = y1[i];
		}

		t += h;
#endif

	} // end while

	#ifdef LOG_KRYLOV_AND_STEPSIZES
		fclose(logFile);
		fclose(rFile);
	#endif

	return EC_success;
}

#ifdef GENERATE_DOCS
}
#endif