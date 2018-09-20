/**
* \file
*
* \author Nicholas J. Curtis
* \date 03/16/2015
*
* \brief A Radau2A IRK implementation for C
* Adapted from Hairer and Wanner's [RADAU5 code](http://www.unige.ch/~hairer/prog/stiff/radau5.f)
* and the [FATODE](http://people.cs.vt.edu/~asandu/Software/FATODE/index.html) ODE integration library
*
* For full reference see:\n
* G. Wanner, E. Hairer, Solving Ordinary Differential Equations II: Stiff and DifferentialAlgebraic
Problems, 2nd Edition, Springer-Verlag, Berlin, 1996. doi:10.1007/978-3-642-
05221-7.
*
* NOTE: all matricies stored in column major format!
*
*/


#include "header.h"
#include "radau2a_solver.hpp"
#include "lapack_dfns.h"
#include "dydt.h"
#include "jacob.h"
#include <complex.h>
#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>

namespace c_solvers
{


	/**
	 * \brief Computes error weight scaling from initial and current state
	 * \param[in]		y0			the initial state vector to use
	 * \param[in]		y			the current state vector
	 * \param[out]		sc			the populated error weight scalings
	 */
	static inline void scale (const double * __restrict__ y0,
							  const double* __restrict__ y,
							  double * __restrict__ sc) {

		for (int i = 0; i < NSP; ++i) {
			sc[i] = 1.0 / (ATOL + fmax(fabs(y0[i]), fabs(y[i])) * RTOL);
		}
	}

	/**
	 * \brief Computes error weight scaling from initial state
	 * \param[in]		y0			the initial state vector to use
	 * \param[out]		sc			the populated error weight scalings
	 */
	static inline void scale_init (const double * __restrict__ y0,
								   double * __restrict__ sc) {

		for (int i = 0; i < NSP; ++i) {
			sc[i] = 1.0 / (ATOL + fabs(y0[i]) * RTOL);
		}
	}

	///////////////////////////////////////////////////////////////////////////////

	/**
	* \brief Compute E1 & E2 matricies and their LU Decomposition
	* \param[in]			H				The timestep size
	* \param[in,out]		E1				The non-complex matrix system
	* \param[in,out]		E2				The complex matrix system
	* \param[in]			Jac				The Jacobian matrix
	* \param[out]			ipiv1			The pivot indicies for E1
	* \param[out]			ipiv2			The pivot indicies for E2
	* \param[out]			info			An indicator variable determining if an error occured.
	*/
	static void RK_Decomp(const double H, double* __restrict__ E1,
						  std::complex<double>* __restrict__ E2, const double* __restrict__ Jac,
						  int* __restrict__ ipiv1, int* __restrict__ ipiv2, int* __restrict__ info) {
		std::complex<double> temp2(Radau::Radau::rkAlpha/H, Radau::rkBeta/H);
		double temp1 = Radau::rkGamma / H;

		for (int i = 0; i < NSP; i++)
		{

			for(int j = 0; j < NSP; j++)
			{
				E1[i + j * NSP] = -Jac[i + j * NSP];
				E2[i + j * NSP] = std::complex<double>(-Jac[i + j * NSP], 0);
			}
			E1[i + i * NSP] += temp1;
			E2[i + i * NSP] += temp2;
		}
		dgetrf_(&Radau::ARRSIZE, &Radau::ARRSIZE, E1, &Radau::ARRSIZE, ipiv1, info);
		if (*info != 0) {
			return;
		}
		zgetrf_(&Radau::ARRSIZE, &Radau::ARRSIZE, E2, &Radau::ARRSIZE, ipiv2, info);
	}

	/**
	* \brief Compute Quadaratic interpolate
	*/
	static void RK_Make_Interpolate(const double* __restrict__ Z1, const double* __restrict__ Z2,
									const double* __restrict__ Z3, double* __restrict__ CONT) {
		double den = (Radau::rkC[2] - Radau::rkC[1]) * (Radau::rkC[1] - Radau::rkC[0]) * (Radau::rkC[0] - Radau::rkC[2]);

		for (int i = 0; i < NSP; i++) {
			CONT[i] = ((-Radau::rkC[2] * Radau::rkC[2] * Radau::rkC[1] * Z1[i] + Z3[i] * Radau::rkC[1]* Radau::rkC[0] * Radau::rkC[0]
	                    + Radau::rkC[1] * Radau::rkC[1] * Radau::rkC[2] * Z1[i] - Radau::rkC[1] * Radau::rkC[1] * Radau::rkC[0] * Z3[i]
	                    + Radau::rkC[2] * Radau::rkC[2] * Radau::rkC[0] * Z2[i] - Z2[i] * Radau::rkC[2] * Radau::rkC[0] * Radau::rkC[0])
	                    /den)-Z3[i];
	        CONT[NSP + i] = -( Radau::rkC[0] * Radau::rkC[0] * (Z3[i] - Z2[i]) + Radau::rkC[1] * Radau::rkC[1] * (Z1[i] - Z3[i])
	        				 + Radau::rkC[2] * Radau::rkC[2] * (Z2[i] - Z1[i]) )/den;
	        CONT[NSP + NSP + i] = ( Radau::rkC[0] * (Z3[i] - Z2[i]) + Radau::rkC[1] * (Z1[i] - Z3[i])
	                           + Radau::rkC[2] * (Z2[i] - Z1[i]) ) / den;
		}
	}

	/**
	* \brief Apply quadaratic interpolate to get initial values
	*/
	static void RK_Interpolate(const double H, const double Hold, double* __restrict__ Z1,
							   double* __restrict__ Z2, double* __restrict__ Z3, const double* __restrict__ CONT) {
		double r = H / Hold;
		double x1 = 1.0 + Radau::rkC[0] * r;
		double x2 = 1.0 + Radau::rkC[1] * r;
		double x3 = 1.0 + Radau::rkC[2] * r;

		for (int i = 0; i < NSP; i++) {
			Z1[i] = CONT[i] + x1 * (CONT[NSP + i] + x1 * CONT[NSP + NSP + i]);
			Z2[i] = CONT[i] + x2 * (CONT[NSP + i] + x2 * CONT[NSP + NSP + i]);
			Z3[i] = CONT[i] + x2 * (CONT[NSP + i] + x3 * CONT[NSP + NSP + i]);
		}
	}


	/**
	* \brief Performs \f$Z:= X + Y\f$ with unrolled (or at least bounds known at compile time) loops
	*/
	static inline void WADD(const double* __restrict__ X, const double* __restrict__ Y,
							double* __restrict__ Z) {

		for (int i = 0; i < NSP; i++)
		{
			Z[i] = X[i] + Y[i];
		}
	}

	/**
	* \brief Sepcialization of DAXPY with unrolled (or at least bounds known at compile time) loops
	*
	*
	* Performs:
	*    *  \f$DY1:= DA1 * DX\f$
	*    *  \f$DY2:= DA2 * DX\f$
	*    *  \f$DY3:= DA3 * DX\f$
	*/
	static inline void DAXPY3(const double DA1, const double DA2, const double DA3,
							  const double* __restrict__ DX, double* __restrict__ DY1,
							  double* __restrict__ DY2, double* __restrict__ DY3) {

		for (int i = 0; i < NSP; i++) {
			DY1[i] += DA1 * DX[i];
			DY2[i] += DA2 * DX[i];
			DY3[i] += DA3 * DX[i];
		}
	}

	/**
	*	\brief Prepare the right-hand side for Newton iterations:
	*     \f$R = Z - hA * F\f$
	*/
	static void RK_PrepareRHS(const double t, const  double pr, const  double H,
							  const double* __restrict__ Y, const double* __restrict__ Z1,
							  const double* __restrict__ Z2, const double* __restrict__ Z3,
							  double* __restrict__ R1, double* __restrict__ R2, double* __restrict__ R3) {
		double TMP[NSP];
		double F[NSP];

		for (int i = 0; i < NSP; i++) {
			R1[i] = Z1[i];
			R2[i] = Z2[i];
			R3[i] = Z3[i];
		}

		// TMP = Y + Z1
		WADD(Y, Z1, TMP);
		dydt(t + Radau::rkC[0] * H, pr, TMP, F);
		//R[:] -= -h * Radau::rkA[:][0] * F[:]
		DAXPY3(-H * Radau::rkA[0][0], -H * Radau::rkA[1][0], -H * Radau::rkA[2][0], F, R1, R2, R3);

		// TMP = Y + Z2
		WADD(Y, Z2, TMP);
		dydt(t + Radau::rkC[1] * H, pr, TMP, F);
		//R[:] -= -h * Radau::rkA[:][1] * F[:]
		DAXPY3(-H * Radau::rkA[0][1], -H * Radau::rkA[1][1], -H * Radau::rkA[2][1], F, R1, R2, R3);

		// TMP = Y + Z3
		WADD(Y, Z3, TMP);
		dydt(t + Radau::rkC[2] * H, pr, TMP, F);
		//R[:] -= -h * Radau::rkA[:][2] * F[:]
		DAXPY3(-H * Radau::rkA[0][2], -H * Radau::rkA[1][2], -H * Radau::rkA[2][2], F, R1, R2, R3);
	}

	/**
	 * \brief Solves for the RHS values in the Newton iteration
	 */
	static void RK_Solve(const double H, double* __restrict__ E1,
						 std::complex<double>* __restrict__ E2, double* __restrict__ R1,
						 double* __restrict__ R2, double* __restrict__ R3, int* __restrict__ ipiv1,
						 int* __restrict__ ipiv2) {
		// Z = (1/h) T^(-1) A^(-1) * Z

		for(int i = 0; i < NSP; i++)
		{
			double x1 = R1[i] / H;
			double x2 = R2[i] / H;
			double x3 = R3[i] / H;
			R1[i] = Radau::Radau::rkTinvAinv[0][0] * x1 + Radau::Radau::rkTinvAinv[0][1] * x2 + Radau::Radau::rkTinvAinv[0][2] * x3;
			R2[i] = Radau::Radau::rkTinvAinv[1][0] * x1 + Radau::Radau::rkTinvAinv[1][1] * x2 + Radau::Radau::rkTinvAinv[1][2] * x3;
			R3[i] = Radau::Radau::rkTinvAinv[2][0] * x1 + Radau::Radau::rkTinvAinv[2][1] * x2 + Radau::Radau::rkTinvAinv[2][2] * x3;
		}
		int info = 0;
		dgetrs_ (&Radau::TRANS, &Radau::ARRSIZE, &Radau::NRHS, E1, &Radau::ARRSIZE, ipiv1, R1, &Radau::ARRSIZE, &info);
	#ifdef DEBUG
		if (info != 0) {
			printf("Error in back-substitution\n");
			exit(-1);
		}
	#endif
		std::complex<double> temp[NSP];

		for (int i = 0; i < NSP; ++i)
		{
			temp[i] = std::complex<double>(R2[i], R3[i]);
		}
		zgetrs_(&Radau::TRANS, &Radau::ARRSIZE, &Radau::NRHS, E2, &Radau::ARRSIZE, ipiv2, temp, &Radau::ARRSIZE, &info);
	#ifdef DEBUG
		if (info != 0) {
			printf("Error in back-substitution\n");
			exit(-1);
		}
	#endif

		for (int i = 0; i < NSP; ++i)
		{
			R2[i] = temp[i].real();
			R3[i] = temp[i].imag();
		}

		// Z = T * Z

		for (int i = 0; i < NSP; ++i) {
			double x1 = R1[i];
			double x2 = R2[i];
			double x3 = R3[i];
			R1[i] = Radau::rkT[0][0] * x1 + Radau::rkT[0][1] * x2 + Radau::rkT[0][2] * x3;
			R2[i] = Radau::rkT[1][0] * x1 + Radau::rkT[1][1] * x2 + Radau::rkT[1][2] * x3;
			R3[i] = Radau::rkT[2][0] * x1 + Radau::rkT[2][1] * x2 + Radau::rkT[2][2] * x3;
		}
	}

	/**
	 * \brief Computes the scaled error norm from the given `scale` and `DY` vectors
	 */
	static inline double RK_ErrorNorm(const double* __restrict__ scale, double* __restrict__ DY) {

		double sum = 0;
		for (int i = 0; i < NSP; ++i){
			sum += (scale[i] * scale[i] * DY[i] * DY[i]);
		}
		return fmax(sqrt(sum / ((double)NSP)), 1e-10);
	}

	/**
	 * \brief Computes and returns the error estimate for this step
	 */
	static double RK_ErrorEstimate(const double H, const double t, const double pr,
								   const double* __restrict__ Y, const double* __restrict__ F0,
								   const double* __restrict__ Z1, const double* __restrict__ Z2, const double* __restrict__ Z3,
								   const double* __restrict__ scale, double* __restrict__ E1, int* __restrict__ ipiv1,
								   const bool FirstStep, const bool Reject) {
		double HrkE1  = Radau::rkE[1]/H;
	    double HrkE2  = Radau::rkE[2]/H;
	    double HrkE3  = Radau::rkE[3]/H;

	    double F1[NSP] = {0};
	    double F2[NSP] = {0};
	    double TMP[NSP] = {0};

	    for (int i = 0; i < NSP; ++i) {
	    	F2[i] = HrkE1 * Z1[i] + HrkE2 * Z2[i] + HrkE3 * Z3[i];
	    }

	    for (int i = 0; i < NSP; ++i) {
	    	TMP[i] = Radau::rkE[0] * F0[i] + F2[i];
	    }
	    int info = 0;
	    dgetrs_ (&Radau::TRANS, &Radau::ARRSIZE, &Radau::NRHS, E1, &Radau::ARRSIZE, ipiv1, TMP, &Radau::ARRSIZE, &info);
	#ifdef DEBUG
	    //this is only true on an incorrect call of dgetrs, hence ignore
	    if (info != 0) {
	    	printf("Error on back-substitution.");
	    	exit(-1);
	    }
	#endif
	    double Err = RK_ErrorNorm(scale, TMP);
	    if (Err >= 1.0 && (FirstStep || Reject)) {

	    	for (int i = 0; i < NSP; i++) {
	        	TMP[i] += Y[i];
	        }
	    	dydt(t, pr, TMP, F1);

	    	for (int i = 0; i < NSP; i++) {
	        	TMP[i] = F1[i] + F2[i];
	        }
	       	dgetrs_ (&Radau::TRANS, &Radau::ARRSIZE, &Radau::NRHS, E1, &Radau::ARRSIZE, ipiv1, TMP, &Radau::ARRSIZE, &info);
	#ifdef DEBUG
	       	if (info != 0) {
		    	printf("Error on back-substitution.");
		    	exit(-1);
	    	}
	#endif
	        Err = RK_ErrorNorm(scale, TMP);
	    }
	    return Err;
	}

	/**
	 *  \brief 5th-order Radau2A CPU implementation
	 *
	 *	\param[in]			t_start				The starting time
	 *  \param[in]			t_end				The end integration time
	 *  \param[in]			pr					The system constant variable (pressure/density)
	 *	\param[in,out]		y 					The system state vector at time `t_start`.
	 											Overwritten with the system state at time `t_end`
	 *  \returns Return code, @see RK_ErrCodes
	 */
	ErrorCode RadauIntegrator::integrate (const double t_start, const double t_end, const double pr, double* y) const {
		double Hmin = 0;
		double Hold = 0;
	#ifdef Gustafsson
		double Hacc = 0;
		double ErrOld = 0;
	#endif
	#ifdef CONST_TIME_STEP
		double H = t_end - t_start;
	#else
		double H = fmin(5e-7, t_end - t_start);
	#endif
		double Hnew;
		double t = t_start;
		bool Reject = false;
		bool FirstStep = true;
		bool SkipJac = false;
		bool SkipLU = false;
		double sc[NSP];
		double A[NSP * NSP] = {0.0};
		double E1[NSP * NSP] = {0};
		std::complex<double> E2[NSP * NSP] = {0};
		int ipiv1[NSP] = {0};
		int ipiv2[NSP] = {0};
		double Z1[NSP] = {0};
		double Z2[NSP] = {0};
		double Z3[NSP] = {0};
	#ifdef SDIRK_ERROR
		double Z4[NSP] = {0};
		double DZ4[NSP] = {0};
		double G[NSP] = {0};
		double TMP[NSP] = {0};
	#endif
		double DZ1[NSP] = {0};
		double DZ2[NSP] = {0};
		double DZ3[NSP] = {0};
		double CONT[NSP * 3] = {0};
		scale_init(y, sc);
		double y0[NSP];
		memcpy(y0, y, NSP * sizeof(double));
		double F0[NSP];
		int info = 0;
		int Nconsecutive = 0;
		int Nsteps = 0;
		double NewtonRate = pow(2.0, 1.25);

		while (t + Roundoff < t_end) {
			if (!Reject) {
				dydt (t, pr, y, F0);
			}
			if (!SkipLU) {
				//need to update Jac/LU
				if (!SkipJac) {
					eval_jacob (t, pr, y, A);
				}
				RK_Decomp(H, E1, E2, A, ipiv1, ipiv2, &info);
				if (info != 0) {
					Nconsecutive += 1;
					if (Nconsecutive >= Max_consecutive_errs)
					{
						return ErrorCode::MAX_CONSECUTIVE_ERRORS_EXCEEDED;
					}
					H *= 0.5;
					Reject = true;
					SkipJac = true;
					SkipLU = false;
					continue;
				}
				else
				{
					Nconsecutive = 0;
				}
			}
			Nsteps += 1;
			if (Nsteps >= Max_no_steps) {
				return ErrorCode::MAX_STEPS_EXCEEDED;
			}
			if (0.1 * fabs(H) <= fabs(t) * Roundoff) {
				return ErrorCode::H_PLUS_T_EQUALS_H;
			}
			if (FirstStep || !StartNewton) {
				memset(Z1, 0, NSP * sizeof(double));
				memset(Z2, 0, NSP * sizeof(double));
				memset(Z3, 0, NSP * sizeof(double));
			} else {
				RK_Interpolate(H, Hold, Z1, Z2, Z3, CONT);
			}
			bool NewtonDone = false;
			double NewtonIncrementOld = 0;
			double Fac = 0.5; //Step reduction if too many iterations
			int NewtonIter = 0;
			double Theta = 0;

			//reuse previous NewtonRate
			NewtonRate = pow(fmax(NewtonRate, EPS), 0.8);

			for (; NewtonIter < NewtonMaxit; NewtonIter++) {
				RK_PrepareRHS(t, pr, H, y, Z1, Z2, Z3, DZ1, DZ2, DZ3);
				RK_Solve(H, E1, E2, DZ1, DZ2, DZ3, ipiv1, ipiv2);
				double d1 = RK_ErrorNorm(sc, DZ1);
				double d2 = RK_ErrorNorm(sc, DZ2);
				double d3 = RK_ErrorNorm(sc, DZ3);
				double NewtonIncrement = sqrt((d1 * d1 + d2 * d2 + d3 * d3) / 3.0);
				Theta = ThetaMin;
				if (NewtonIter > 0)
				{
					Theta = NewtonIncrement / NewtonIncrementOld;
					if (Theta < 0.99) {
						NewtonRate = Theta / (1.0 - Theta);
					}
					else { // Non-convergence of Newton: Theta too large
						break;
					}
					if (NewtonIter < NewtonMaxit) {
						//Predict error at the end of Newton process
						double NewtonPredictedErr = (NewtonIncrement * pow(Theta, (NewtonMaxit - NewtonIter - 1))) / (1.0 - Theta);
						if (NewtonPredictedErr >= NewtonTol) {
							//Non-convergence of Newton: predicted error too large
							double Qnewton = fmin(10.0, NewtonPredictedErr / NewtonTol);
		                    Fac = 0.8 * pow(Qnewton, -1.0/((double)(NewtonMaxit-NewtonIter)));
		                    break;
						}
					}
				}

				NewtonIncrementOld = fmax(NewtonIncrement, Roundoff);
	            // Update solution

	            for (int i = 0; i < NSP; i++)
	            {
	            	Z1[i] -= DZ1[i];
	            	Z2[i] -= DZ2[i];
	            	Z3[i] -= DZ3[i];
	            }

	            NewtonDone = (NewtonRate * NewtonIncrement <= NewtonTol);
	            if (NewtonDone) break;
	            if (NewtonIter == NewtonMaxit - 1) {
	            	return ErrorCode::MAX_NEWTON_ITER_EXCEEDED;
	            }
			}
	#ifndef CONST_TIME_STEP
			if (!NewtonDone) {
				H = Fac * H;
				Reject = true;
				SkipJac = true;
				SkipLU = false;
				continue;
			}

			double Err = RK_ErrorEstimate(H, t, pr, y, F0, Z1, Z2, Z3, sc, E1, ipiv1, FirstStep, Reject);
			//~~~> Computation of new step size Hnew
			Fac = pow(Err, (-1.0 / Radau::rkELO)) * (1.0 + 2 * NewtonMaxit) / (NewtonIter + 1.0 + 2 * NewtonMaxit);
			Fac = fmin(FacMax, fmax(FacMin, Fac));
			Hnew = Fac * H;
			if (Err < 1.0) {
	#ifdef Gustafsson
				if (!FirstStep) {
					double FacGus = FacSafe * (H / Hacc) * pow(Err * Err / ErrOld, -0.25);
					FacGus = fmin(FacMax, fmax(FacMin, FacGus));
					Fac = fmin(Fac, FacGus);
					Hnew = Fac * H;
				}
				Hacc = H;
				ErrOld = fmax(1e-2, Err);
	#endif
				FirstStep = false;
				Hold = H;
				t += H;

				for (int i = 0; i < NSP; i++) {
					y[i] += Z3[i];
				}
				// Construct the solution quadratic interpolant Q(c_i) = Z_i, i=1:3
				if (StartNewton) {
					RK_Make_Interpolate(Z1, Z2, Z3, CONT);
				}
				scale(y, y0, sc);
				memcpy(y0, y, NSP * sizeof(double));
				Hnew = fmin(fmax(Hnew, Hmin), t_end - t);
				if (Reject) {
					Hnew = fmin(Hnew, H);
				}
				Reject = false;
				if (t + Hnew / Qmin - t_end >= 0.0) {
					H = t_end - t;
				} else {
					double Hratio = Hnew / H;
		            // Reuse the LU decomposition
		            SkipLU = (Theta <= ThetaMin) && (Hratio>=Qmin) && (Hratio<=Qmax);
		            if (!SkipLU) H = Hnew;
				}
				// If convergence is fast enough, do not update Jacobian
	         	SkipJac = NewtonIter == 1 || NewtonRate <= ThetaMin;
			}
			else {
				if (FirstStep || Reject) {
					H = FacRej * H;
				} else {
					H = Hnew;
				}
				Reject = true;
				SkipJac = true;
				SkipLU = false;
			}
	#else
			//constant time stepping
			//update y & t
			t += H;

			for (int i = 0; i < NSP; i++) {
				y[i] += Z3[i];
			}
	#endif
		}
		return ErrorCode::SUCCESS;
	}

}
