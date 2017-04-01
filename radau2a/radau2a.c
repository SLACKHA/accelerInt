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
#include "solver_props.h"
#include "solver_options.h"
#include "lapack_dfns.h"
#include "dydt.h"
#include "jacob.h"
#include <complex.h>
#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>

#ifdef GENERATE_DOCS
namespace radau2a {
#endif

//! Maximum number of allowed internal timesteps before error
#define Max_no_steps (200000)
//! Maximum number of allowed Newton iteration steps before error
#define NewtonMaxit (8)
//! Use quadratic interpolation from previous step if possible
#define StartNewton (true)
//! Use gustafsson time stepping control
#define Gustafsson
//! Smallist representable double precision number
#define Roundoff (EPS)
//! Controls maximum decrease in timestep size
#define FacMin (0.2)
//! Controls maximum increase in timestep size
#define FacMax (8)
//! Safety factor for Gustafsson time stepping control
#define FacSafe (0.9)
//! Time step factor on rejected step
#define FacRej (0.1)
//! Minimum Newton convergence rate
#define ThetaMin (0.001)
//! Newton convergence tolerance
#define NewtonTol (0.03)
//! Min Timestep ratio to skip LU decomposition
#define Qmin (1.0)
//! Max Timestep ratio to skip LU decomposition
#define Qmax (1.2)
//#define UNROLL (8)
//! Error allowed on this many consecutive internal timesteps before exit
#define Max_consecutive_errs (5)
//#define SDIRK_ERROR

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

/**
 * \defgroup RK_Params Various parameters for the RadauIIA method
 * @{
 */

const static double rkA[3][3] = { {
	 1.968154772236604258683861429918299e-1,
	-6.55354258501983881085227825696087e-2,
	 2.377097434822015242040823210718965e-2
	}, {
	 3.944243147390872769974116714584975e-1,
	 2.920734116652284630205027458970589e-1,
	-4.154875212599793019818600988496743e-2
	}, {
	 3.764030627004672750500754423692808e-1,
	 5.124858261884216138388134465196080e-1,
	 1.111111111111111111111111111111111e-1
	}
};

const static double rkB[3] = {
3.764030627004672750500754423692808e-1,
5.124858261884216138388134465196080e-1,
1.111111111111111111111111111111111e-1
};

const static double rkC[3] = {
1.550510257216821901802715925294109e-1,
6.449489742783178098197284074705891e-1,
1.0
};

#ifdef SDIRK_ERROR
	// Classical error estimator:
	// H* Sum (B_j-Bhat_j)*f(Z_j) = H*E(0)*f(0) + Sum E_j*Z_j
	const static double rkE[4] = {
	0.02,
	-10.04880939982741556246032950764708*0.02,
	1.382142733160748895793662840980412*0.02,
	-0.3333333333333333333333333333333333*0.02
	};
	// H* Sum Bgam_j*f(Z_j) = H*Bgam(0)*f(0) + Sum Theta_j*Z_j
	const static double rkTheta[3] = {
	-1.520677486405081647234271944611547 - 10.04880939982741556246032950764708*0.02,
	2.070455145596436382729929151810376 + 1.382142733160748895793662840980413*0.02,
	-0.3333333333333333333333333333333333*0.02 - 0.3744441479783868387391430179970741
	};
	// ! Sdirk error estimator
	const static double rkBgam[5] = {
	0.02,
	0.3764030627004672750500754423692807-1.558078204724922382431975370686279*0.02,
	0.8914115380582557157653087040196118*0.02+0.5124858261884216138388134465196077,
	-0.1637777184845662566367174924883037-0.3333333333333333333333333333333333*0.02,
	0.2748888295956773677478286035994148
	};
#else
	// Classical error estimator:
	// H* Sum (B_j-Bhat_j)*f(Z_j) = H*E(0)*f(0) + Sum E_j*Z_j
	const static double rkE[4] = {
	0.05,
	-10.04880939982741556246032950764708*0.05,
	1.382142733160748895793662840980412*0.05,
	-0.3333333333333333333333333333333333*0.05
	};
	// H* Sum Bgam_j*f(Z_j) = H*Bgam(0)*f(0) + Sum Theta_j*Z_j
	const static double rkTheta[3] = {
	-1.520677486405081647234271944611547 - 10.04880939982741556246032950764708*0.05,
	2.070455145596436382729929151810376 + 1.382142733160748895793662840980413*0.05,
	-0.3333333333333333333333333333333333*0.05 - 0.3744441479783868387391430179970741
	};
#endif

//Local order of error estimator
/*
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
!~~~> Diagonalize the RK matrix:
! rkTinv * inv(rkA) * rkT =
!           |  rkGamma      0           0     |
!           |      0      rkAlpha   -rkBeta   |
!           |      0      rkBeta     rkAlpha  |
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

const static double rkGamma = 3.637834252744495732208418513577775;
const static double rkAlpha = 2.681082873627752133895790743211112;
const static double rkBeta  = 3.050430199247410569426377624787569;

const static double rkT[3][3] = {
{9.443876248897524148749007950641664e-2,
-1.412552950209542084279903838077973e-1,
-3.00291941051474244918611170890539e-2},
{2.502131229653333113765090675125018e-1,
2.041293522937999319959908102983381e-1,
3.829421127572619377954382335998733e-1},
{1.0,
1.0,
0.0e0}
};

const static double rkTinv[3][3] =
{{4.178718591551904727346462658512057,
3.27682820761062387082533272429617e-1,
5.233764454994495480399309159089876e-1},
{-4.178718591551904727346462658512057,
-3.27682820761062387082533272429617e-1,
4.766235545005504519600690840910124e-1},
{-5.02872634945786875951247343139544e-1,
2.571926949855605429186785353601676e0,
-5.960392048282249249688219110993024e-1}
};

const static double rkTinvAinv[3][3] = {
{1.520148562492775501049204957366528e+1,
1.192055789400527921212348994770778,
1.903956760517560343018332287285119},
{-9.669512977505946748632625374449567,
-8.724028436822336183071773193986487,
3.096043239482439656981667712714881},
{-1.409513259499574544876303981551774e+1,
5.895975725255405108079130152868952,
-1.441236197545344702389881889085515e-1}
};

const static double rkAinvT[3][3] = {
{0.3435525649691961614912493915818282,
-0.4703191128473198422370558694426832,
0.3503786597113668965366406634269080},
{0.9102338692094599309122768354288852,
1.715425895757991796035292755937326,
0.4040171993145015239277111187301784},
{3.637834252744495732208418513577775,
2.681082873627752133895790743211112,
-3.050430199247410569426377624787569}
};

const static double rkELO = 4;

/**
 * @}
 */

//! Lapack - non-transpose
static char TRANS = 'N';
//! Lapack - 1 RHS solve
static int NRHS = 1;
//! Lapack - Array size
static int ARRSIZE = NSP;

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
					  double complex* __restrict__ E2, const double* __restrict__ Jac,
					  int* __restrict__ ipiv1, int* __restrict__ ipiv2, int* __restrict__ info) {
	double complex temp2 = rkAlpha/H + I * rkBeta/H;
	double temp1 = rkGamma / H;

	for (int i = 0; i < NSP; i++)
	{

		for(int j = 0; j < NSP; j++)
		{
			E1[i + j * NSP] = -Jac[i + j * NSP];
			E2[i + j * NSP] = -Jac[i + j * NSP] + 0 * I;
		}
		E1[i + i * NSP] += temp1;
		E2[i + i * NSP] += temp2;
	}
	dgetrf_(&ARRSIZE, &ARRSIZE, E1, &ARRSIZE, ipiv1, info);
	if (*info != 0) {
		return;
	}
	zgetrf_(&ARRSIZE, &ARRSIZE, E2, &ARRSIZE, ipiv2, info);
}

/**
* \brief Compute Quadaratic interpolate
*/
static void RK_Make_Interpolate(const double* __restrict__ Z1, const double* __restrict__ Z2,
								const double* __restrict__ Z3, double* __restrict__ CONT) {
	double den = (rkC[2] - rkC[1]) * (rkC[1] - rkC[0]) * (rkC[0] - rkC[2]);

	for (int i = 0; i < NSP; i++) {
		CONT[i] = ((-rkC[2] * rkC[2] * rkC[1] * Z1[i] + Z3[i] * rkC[1]* rkC[0] * rkC[0]
                    + rkC[1] * rkC[1] * rkC[2] * Z1[i] - rkC[1] * rkC[1] * rkC[0] * Z3[i]
                    + rkC[2] * rkC[2] * rkC[0] * Z2[i] - Z2[i] * rkC[2] * rkC[0] * rkC[0])
                    /den)-Z3[i];
        CONT[NSP + i] = -( rkC[0] * rkC[0] * (Z3[i] - Z2[i]) + rkC[1] * rkC[1] * (Z1[i] - Z3[i])
        				 + rkC[2] * rkC[2] * (Z2[i] - Z1[i]) )/den;
        CONT[NSP + NSP + i] = ( rkC[0] * (Z3[i] - Z2[i]) + rkC[1] * (Z1[i] - Z3[i])
                           + rkC[2] * (Z2[i] - Z1[i]) ) / den;
	}
}

/**
* \brief Apply quadaratic interpolate to get initial values
*/
static void RK_Interpolate(const double H, const double Hold, double* __restrict__ Z1,
						   double* __restrict__ Z2, double* __restrict__ Z3, const double* __restrict__ CONT) {
	double r = H / Hold;
	double x1 = 1.0 + rkC[0] * r;
	double x2 = 1.0 + rkC[1] * r;
	double x3 = 1.0 + rkC[2] * r;

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
	dydt(t + rkC[0] * H, pr, TMP, F);
	//R[:] -= -h * rkA[:][0] * F[:]
	DAXPY3(-H * rkA[0][0], -H * rkA[1][0], -H * rkA[2][0], F, R1, R2, R3);

	// TMP = Y + Z2
	WADD(Y, Z2, TMP);
	dydt(t + rkC[1] * H, pr, TMP, F);
	//R[:] -= -h * rkA[:][1] * F[:]
	DAXPY3(-H * rkA[0][1], -H * rkA[1][1], -H * rkA[2][1], F, R1, R2, R3);

	// TMP = Y + Z3
	WADD(Y, Z3, TMP);
	dydt(t + rkC[2] * H, pr, TMP, F);
	//R[:] -= -h * rkA[:][2] * F[:]
	DAXPY3(-H * rkA[0][2], -H * rkA[1][2], -H * rkA[2][2], F, R1, R2, R3);
}

/**
 * \brief Solves for the RHS values in the Newton iteration
 */
static void RK_Solve(const double H, double* __restrict__ E1,
					 double complex* __restrict__ E2, double* __restrict__ R1,
					 double* __restrict__ R2, double* __restrict__ R3, int* __restrict__ ipiv1,
					 int* __restrict__ ipiv2) {
	// Z = (1/h) T^(-1) A^(-1) * Z

	for(int i = 0; i < NSP; i++)
	{
		double x1 = R1[i] / H;
		double x2 = R2[i] / H;
		double x3 = R3[i] / H;
		R1[i] = rkTinvAinv[0][0] * x1 + rkTinvAinv[0][1] * x2 + rkTinvAinv[0][2] * x3;
		R2[i] = rkTinvAinv[1][0] * x1 + rkTinvAinv[1][1] * x2 + rkTinvAinv[1][2] * x3;
		R3[i] = rkTinvAinv[2][0] * x1 + rkTinvAinv[2][1] * x2 + rkTinvAinv[2][2] * x3;
	}
	int info = 0;
	dgetrs_ (&TRANS, &ARRSIZE, &NRHS, E1, &ARRSIZE, ipiv1, R1, &ARRSIZE, &info);
#ifdef DEBUG
	if (info != 0) {
		printf("Error in back-substitution\n");
		exit(-1);
	}
#endif
	double complex temp[NSP];

	for (int i = 0; i < NSP; ++i)
	{
		temp[i] = R2[i] + I * R3[i];
	}
	zgetrs_(&TRANS, &ARRSIZE, &NRHS, E2, &ARRSIZE, ipiv2, temp, &ARRSIZE, &info);
#ifdef DEBUG
	if (info != 0) {
		printf("Error in back-substitution\n");
		exit(-1);
	}
#endif

	for (int i = 0; i < NSP; ++i)
	{
		R2[i] = creal(temp[i]);
		R3[i] = cimag(temp[i]);
	}

	// Z = T * Z

	for (int i = 0; i < NSP; ++i) {
		double x1 = R1[i];
		double x2 = R2[i];
		double x3 = R3[i];
		R1[i] = rkT[0][0] * x1 + rkT[0][1] * x2 + rkT[0][2] * x3;
		R2[i] = rkT[1][0] * x1 + rkT[1][1] * x2 + rkT[1][2] * x3;
		R3[i] = rkT[2][0] * x1 + rkT[2][1] * x2 + rkT[2][2] * x3;
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
	double HrkE1  = rkE[1]/H;
    double HrkE2  = rkE[2]/H;
    double HrkE3  = rkE[3]/H;

    double F1[NSP] = {0};
    double F2[NSP] = {0};
    double TMP[NSP] = {0};

    for (int i = 0; i < NSP; ++i) {
    	F2[i] = HrkE1 * Z1[i] + HrkE2 * Z2[i] + HrkE3 * Z3[i];
    }

    for (int i = 0; i < NSP; ++i) {
    	TMP[i] = rkE[0] * F0[i] + F2[i];
    }
    int info = 0;
    dgetrs_ (&TRANS, &ARRSIZE, &NRHS, E1, &ARRSIZE, ipiv1, TMP, &ARRSIZE, &info);
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
       	dgetrs_ (&TRANS, &ARRSIZE, &NRHS, E1, &ARRSIZE, ipiv1, TMP, &ARRSIZE, &info);
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
int integrate (const double t_start, const double t_end, const double pr, double* y) {
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
	double complex E2[NSP * NSP] = {0};
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
					return EC_consecutive_steps;
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
			return EC_max_steps_exceeded;
		}
		if (0.1 * fabs(H) <= fabs(t) * Roundoff) {
			return EC_h_plus_t_equals_h;
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
            	return EC_newton_max_iterations_exceeded;
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
		Fac = pow(Err, (-1.0 / rkELO)) * (1.0 + 2 * NewtonMaxit) / (NewtonIter + 1.0 + 2 * NewtonMaxit);
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
	return EC_success;
}

#ifdef GENERATE_DOCS
}
#endif