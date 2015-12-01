/** 
* \file radau2a.cu
*
* \author Nicholas J. Curtis
* \date 03/16/2015
*
* A Radau2A IRK implementation for CUDA
* 
* NOTE: all matricies stored in column major format!
* 
*/

#include "header.cuh"
#include "solver_options.h"
#include "inverse.cuh"
#include "complexInverse_NSP.cuh"
#include "solver_options.h"
#include "jacob.cuh"
#include "dydt.cuh"
#include <cuComplex.h>

#ifdef NEWTON_UNROLL
	#define NEWTON_UNROLLER	#pragma unroll NEWTON_UNROLL
#else
	#define NEWTON_UNROLLER #pragma unroll
#endif
#ifdef WARP_VOTING
	#define ANY(X) (__any((X)))
	#define ALL(X) (__all((X)))
#else
	#define ANY(X) ((X))
	#define ALL(X) ((X))
#endif
#define Max_no_steps (200000)
#define NewtonMaxit (8)
#define StartNewton (true)
#define Gustafsson
#define Roundoff (EPS)
#define FacMin (0.2)
#define FacMax (8)
#define FacSafe (0.9)
#define FacRej (0.1)
#define ThetaMin (0.001)
#define NewtonTol (0.03)
#define Qmin (1.0)
#define Qmax (1.2)
#define UNROLL (8)
//#define SDIRK_ERROR

__device__
void scale (const double * y0, const double* y, double * sc) {
	#pragma unroll 1
	for (int i = 0; i < NSP; ++i) {
		sc[i] = 1.0 / (ATOL + fmax(fabs(y0[i]), fabs(y[i])) * RTOL);
	}
}

__device__
void scale_init (const double * y0, double * sc) {
	#pragma unroll 1
	for (int i = 0; i < NSP; ++i) {
		sc[i] = 1.0 / (ATOL + fabs(y0[i]) * RTOL);
	}
}

__device__
void safe_memcpy(double* dest, const double* source)
{
	#pragma unroll 1
	for (int i = 0; i < NSP; i++)
	{
		dest[i] = source[i];
	}
}
__device__
void safe_memset3(double* dest1, double* dest2, double* dest3, const double val)
{
	#pragma unroll 1
	for (int i = 0; i < NSP; i++)
	{
		dest1[i] = val;
		dest2[i] = val;
		dest3[i] = val;
	}
}

__constant__ double rkA[3][3] = { {
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

__constant__ double rkB[3] = {
3.764030627004672750500754423692808e-1,
5.124858261884216138388134465196080e-1,
1.111111111111111111111111111111111e-1
};

__constant__ double rkC[3] = {
1.550510257216821901802715925294109e-1,
6.449489742783178098197284074705891e-1,
1.0
};

#ifdef SDIRK_ERROR
	// Classical error estimator: 
	// H* Sum (B_j-Bhat_j)*f(Z_j) = H*E(0)*f(0) + Sum E_j*Z_j
	__constant__ double rkE[4] = {
	0.02,
	-10.04880939982741556246032950764708e0*0.02,
	1.382142733160748895793662840980412e0*0.02,
	-0.3333333333333333333333333333333333e0*0.02
	};
	// H* Sum Bgam_j*f(Z_j) = H*Bgam(0)*f(0) + Sum Theta_j*Z_j
	__constant__ double rkTheta[3] = {
	-1.520677486405081647234271944611547e0 - 10.04880939982741556246032950764708e0*0.02,
	2.070455145596436382729929151810376e0 + 1.382142733160748895793662840980413e0*0.02,
	-0.3333333333333333333333333333333333e0*0.02 - 0.3744441479783868387391430179970741
	};
	// ! Sdirk error estimator
	__constant__ double rkBgam[5] = {
	0.02,
	0.3764030627004672750500754423692807-1.558078204724922382431975370686279*0.02,
	0.8914115380582557157653087040196118*0.02+.5124858261884216138388134465196077,
	-0.1637777184845662566367174924883037-0.3333333333333333333333333333333333*0.02,
	0.2748888295956773677478286035994148
	};
#else
	// Classical error estimator: 
	// H* Sum (B_j-Bhat_j)*f(Z_j) = H*E(0)*f(0) + Sum E_j*Z_j
	__constant__ double rkE[4] = {
	0.05,
	-10.04880939982741556246032950764708e0*0.05,
	1.382142733160748895793662840980412e0*0.05,
	-0.3333333333333333333333333333333333e0*0.05
	};
	// H* Sum Bgam_j*f(Z_j) = H*Bgam(0)*f(0) + Sum Theta_j*Z_j
	__constant__ double rkTheta[3] = {
	-1.520677486405081647234271944611547e0 - 10.04880939982741556246032950764708e0*0.05,
	2.070455145596436382729929151810376e0 + 1.382142733160748895793662840980413e0*0.05,
	-0.3333333333333333333333333333333333e0*0.05 - 0.3744441479783868387391430179970741e0
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

__constant__ double rkGamma = 3.637834252744495732208418513577775e0;
__constant__ double rkAlpha = 2.681082873627752133895790743211112e0;
__constant__ double rkBeta  = 3.050430199247410569426377624787569e0;

__constant__ double rkT[3][3] = {
{9.443876248897524148749007950641664e-2,
-1.412552950209542084279903838077973e-1,
-3.00291941051474244918611170890539e-2},
{2.502131229653333113765090675125018e-1,
2.041293522937999319959908102983381e-1,
3.829421127572619377954382335998733e-1},
{1.0e0,
1.0e0,
0.0e0}
};

__constant__ double rkTinv[3][3] = {
{4.178718591551904727346462658512057e0,
3.27682820761062387082533272429617e-1,
5.233764454994495480399309159089876e-1},
{-4.178718591551904727346462658512057e0,
-3.27682820761062387082533272429617e-1,
4.766235545005504519600690840910124e-1},
{-5.02872634945786875951247343139544e-1,
2.571926949855605429186785353601676e0,
-5.960392048282249249688219110993024e-1}
};

__constant__ double rkTinvAinv[3][3] = {
{1.520148562492775501049204957366528e+1,
1.192055789400527921212348994770778e0,
1.903956760517560343018332287285119e0},
{-9.669512977505946748632625374449567e0,
-8.724028436822336183071773193986487e0,
3.096043239482439656981667712714881e0},
{-1.409513259499574544876303981551774e+1,
5.895975725255405108079130152868952e0,
-1.441236197545344702389881889085515e-1}
};

__constant__ double rkAinvT[3][3] = {
{0.3435525649691961614912493915818282e0,
-0.4703191128473198422370558694426832e0,
0.3503786597113668965366406634269080e0},
{0.9102338692094599309122768354288852e0,
1.715425895757991796035292755937326e0,
0.4040171993145015239277111187301784e0},
{3.637834252744495732208418513577775e0,
2.681082873627752133895790743211112e0,
-3.050430199247410569426377624787569e0}
};

__constant__ double rkELO = 4;

///////////////////////////////////////////////////////////////////////////////

/*
* calculate E1 & E2 matricies and their LU Decomposition
*/
__device__ void RK_Decomp(double H, double* E1, cuDoubleComplex* E2, const double* Jac, int* ipiv1, int* ipiv2, int* info) {
	cuDoubleComplex temp = make_cuDoubleComplex(rkAlpha/H, rkBeta/H);
	#pragma unroll 1
	for (int i = 0; i < NSP; i++)
	{
		#pragma unroll 1
		for(int j = 0; j < NSP; j++)
		{
			E1[i + j * NSP] = -Jac[i + j * NSP];
			E2[i + j * NSP] = make_cuDoubleComplex(-Jac[i + j * NSP], 0);
		}
		E1[i + i * NSP] += rkGamma / H;
		E2[i + i * NSP] = cuCadd(E2[i + i * NSP], temp); 
	}
	getLU(E1, ipiv1, info);
	if (*info != 0) {
		return;
	}
	getComplexLU(E2, ipiv2, info);
}

__device__ void RK_Make_Interpolate(const double* Z1, const double* Z2, const double* Z3, double* CONT) {
	double den = (rkC[2] - rkC[1]) * (rkC[1] - rkC[0]) * (rkC[0] - rkC[2]); 
	#pragma unroll 1
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

__device__ void RK_Interpolate(double H, double Hold, double* Z1, double* Z2, double* Z3, const double* CONT) {
	double r = H / Hold;
	register double x1 = 1.0 + rkC[0] * r;
	register double x2 = 1.0 + rkC[1] * r;
	register double x3 = 1.0 + rkC[2] * r;
	#pragma unroll 1
	for (int i = 0; i < NSP; i++) {
		Z1[i] = CONT[i] + x1 * (CONT[NSP + i] + x1 * CONT[NSP + NSP + i]);
		Z2[i] = CONT[i] + x2 * (CONT[NSP + i] + x2 * CONT[NSP + NSP + i]);
		Z3[i] = CONT[i] + x2 * (CONT[NSP + i] + x3 * CONT[NSP + NSP + i]);
	}
}


__device__ void WADD(const double* X, const double* Y, double* Z) {
	#pragma unroll 1
	for (int i = 0; i < NSP; i++)
	{
		Z[i] = X[i] + Y[i];
	}
}

__device__ void DAXPY3(double DA1, double DA2, double DA3, const double* DX, double* DY1, double* DY2, double* DY3) {
	#pragma unroll 1
	for (int i = 0; i < NSP; i++) {
		DY1[i] += DA1 * DX[i];
		DY2[i] += DA2 * DX[i];
		DY3[i] += DA3 * DX[i];
	}
}

/*
*Prepare the right-hand side for Newton iterations
*     R = Z - hA * F
*/
__device__ void RK_PrepareRHS(double t, double pr, double H, double* Y, double* F0, double* Z1, double* Z2, double* Z3, double* R1, double* R2, double* R3) {
	double TMP[NSP];
	double F[NSP];
	#pragma unroll 1
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

__device__ void dlaswp(double* A, int* ipiv) {
	#pragma unroll 1
	for (int i = 0; i < NSP; i++) {
		int ip = ipiv[i];
		if (ip != i) {
			double temp = A[i];
			A[i] = A[ip];
			A[ip] = temp;
		}
	}	
}

//diag == 'n' -> nounit = true
//upper == 'u' -> upper = true
__device__ void dtrsm(bool upper, bool nounit, double* A, double* b) {
	if (upper) {
		#pragma unroll 1
		for (int k = NSP - 1; k >= 0; --k)
		{
			if (nounit) {
				b[k] /= A[k + k * NSP];
			}
			#pragma unroll 1
			for (int i = 0; i < k; i++)
			{
				b[i] -= b[k] * A[i + k * NSP];
			}
		}
	}
	else{
		#pragma unroll 1
		for (int k = 0; k < NSP; k++) {
			if (fabs(b[k]) > 0) {
				if (nounit) {
					b[k] /= A[k + k * NSP];
				}
				#pragma unroll 1
				for (int i = k + 1; i < NSP; i++)
				{
					b[i] -= b[k] * A[i + k * NSP];
				}
			}
		}
	}
}

__device__ void dgetrs(double* A, double* B, int* ipiv) {
	dlaswp(B, ipiv);
	dtrsm(false, false, A, B);
	dtrsm(true, true, A, B);
}

__device__ void zlaswp(cuDoubleComplex* A, int* ipiv) {
	#pragma unroll 1
	for (int i = 0; i < NSP; i++) {
		int ip = ipiv[i];
		if (ip != i) {
			cuDoubleComplex temp = A[i];
			A[i] = A[ip];
			A[ip] = temp;
		}
	}	
}

//diag == 'n' -> nounit = true
//upper == 'u' -> upper = true
__device__ void ztrsm(bool upper, bool nounit, cuDoubleComplex* A, cuDoubleComplex* b) {
	if (upper) {
		#pragma unroll 1
		for (int k = NSP - 1; k >= 0; --k)
		{
			if (nounit) {
				b[k] = cuCdiv(b[k], A[k + k * NSP]);
			}
			#pragma unroll 1
			for (int i = 0; i < k; i++)
			{
				b[i] = cuCsub(b[i], cuCmul(b[k], A[i + k * NSP]));
			}
		}
	}
	else{
		#pragma unroll 1
		for (int k = 0; k < NSP; k++) {
			if (cuCabs(b[k]) > 0) {
				if (nounit) {
					b[k] = cuCdiv(b[k], A[k + k * NSP]);
				}
				#pragma unroll 1
				for (int i = k + 1; i < NSP; i++)
				{
					b[i] = cuCsub(b[i], cuCmul(b[k], A[i + k * NSP]));
				}
			}
		}
	}
}

__device__ void zgetrs(cuDoubleComplex* A, cuDoubleComplex* B, int* ipiv) {
	zlaswp(B, ipiv);
	ztrsm(false, false, A, B);
	ztrsm(true, true, A, B);
}

__device__ void RK_Solve(double H, double* E1, cuDoubleComplex* E2, double* R1, double* R2, double* R3, int* ipiv1, int* ipiv2) {
	// Z = (1/h) T^(-1) A^(-1) * Z
	#pragma unroll 1
	for(int i = 0; i < NSP; i++)
	{
		double x1 = R1[i] / H;
		double x2 = R2[i] / H;
		double x3 = R3[i] / H;
		R1[i] = rkTinvAinv[0][0] * x1 + rkTinvAinv[0][1] * x2 + rkTinvAinv[0][2] * x3;
		R2[i] = rkTinvAinv[1][0] * x1 + rkTinvAinv[1][1] * x2 + rkTinvAinv[1][2] * x3;
		R3[i] = rkTinvAinv[2][0] * x1 + rkTinvAinv[2][1] * x2 + rkTinvAinv[2][2] * x3;
	}
	dgetrs(E1, R1, ipiv1);
	cuDoubleComplex temp[NSP];
	#pragma unroll 1
	for (int i = 0; i < NSP; ++i)
	{
		temp[i] = make_cuDoubleComplex(R2[i], R3[i]);
	}
	zgetrs(E2, temp, ipiv2);
	#pragma unroll 1
	for (int i = 0; i < NSP; ++i)
	{
		R2[i] = cuCreal(temp[i]);
		R3[i] = cuCimag(temp[i]);
	}

	// Z = T * Z
	#pragma unroll 1
	for (int i = 0; i < NSP; ++i) {
		double x1 = R1[i];
		double x2 = R2[i];
		double x3 = R3[i];
		R1[i] = rkT[0][0] * x1 + rkT[0][1] * x2 + rkT[0][2] * x3;
		R2[i] = rkT[1][0] * x1 + rkT[1][1] * x2 + rkT[1][2] * x3;
		R3[i] = rkT[2][0] * x1 + rkT[2][1] * x2 + rkT[2][2] * x3;
	}
}

__device__ double RK_ErrorNorm(double* scale, double* DY) {
	double sum = 0;
	#pragma unroll 1
	for (int i = 0; i < NSP; ++i){
		sum += (scale[i] * scale[i] * DY[i] * DY[i]);
	}
	return fmax(sqrt(sum / ((double)NSP)), 1e-10);
}

__device__ double RK_ErrorEstimate(double H, double t, double pr, double* Y, double* F0, double* Z1, double* Z2, double* Z3, double* scale, double* E1, int* ipiv1, bool FirstStep, bool Reject) {
	double HrkE1  = rkE[1]/H;
    double HrkE2  = rkE[2]/H;
    double HrkE3  = rkE[3]/H;

    double F1[NSP];
    double F2[NSP];
    double TMP[NSP];
    #pragma unroll 1
    for (int i = 0; i < NSP; ++i) {
    	F2[i] = HrkE1 * Z1[i] + HrkE2 * Z2[i] + HrkE3 * Z3[i];
    }
    #pragma unroll 1
    for (int i = 0; i < NSP; ++i) {
    	TMP[i] = rkE[0] * F0[i] + F2[i];
    }
    dgetrs(E1, TMP, ipiv1);
    double Err = RK_ErrorNorm(scale, TMP);
    if (Err >= 1.0 && (FirstStep || Reject)) {
        #pragma unroll 1
    	for (int i = 0; i < NSP; i++) {
        	TMP[i] += Y[i];
        }
    	dydt(t, pr, TMP, F1);
    	#pragma unroll 1
    	for (int i = 0; i < NSP; i++) {
        	TMP[i] = F1[i] + F2[i];
        }
        dgetrs(E1, TMP, ipiv1);
        Err = RK_ErrorNorm(scale, TMP);
    }
    return Err;
}

/** 
 *  5th-order Radau2A implementation
 * 
 */
__device__ void integrate (const double t_start, const double t_end, const double pr, double* y) {
	double Hmin = 0;
	double Hold = 0;
#ifdef Gustafsson
	double Hacc = 0;
	double ErrOld = 0;
#endif
	double H = fmin(5e-7, t_end - t_start);
	double Hnew;
	double t = t_start;
	bool Reject = false;
	bool FirstStep = true;
	bool SkipJac = false;
	bool SkipLU = false;
	double sc[NSP];
	double A[NSP * NSP] = {0.0};
	double E1[NSP * NSP] = {0.0};
	cuDoubleComplex E2[NSP * NSP] = {make_cuDoubleComplex(0.0, 0.0)};
	int ipiv1[NSP] = {0};
	int ipiv2[NSP] = {0};
	double Z1[NSP] = {0.0};
	double Z2[NSP] = {0.0};
	double Z3[NSP] = {0.0};
#ifdef SDIRK_ERROR
	double Z4[NSP] = {0.0};
	double DZ4[NSP] = {0.0};
	double G[NSP] = {0.0};
	double TMP[NSP] = {0.0};
#endif
	double DZ1[NSP] = {0.0};
	double DZ2[NSP] = {0.0};
	double DZ3[NSP] = {0.0};
	double CONT[NSP * 3] = {0.0};
	scale_init(y, sc);
	double y0[NSP];
	safe_memcpy(y0, y);
	double F0[NSP];
	int info = 0;
	int Nconsecutive = 0;
	int Nsteps = 0;
	double NewtonRate = pow(2.0, 1.25);
	while (t + Roundoff < t_end) {
		if(!Reject) {
			dydt (t, pr, y, F0);
		}
		if(!SkipLU) { 
			//need to update Jac/LU
			if(!SkipJac) {
				eval_jacob (t, pr, y, A);
			}
			RK_Decomp(H, E1, E2, A, ipiv1, ipiv2, &info);
			if(info != 0) {
				Nconsecutive += 1;
				if (Nconsecutive >= 5)
				{
					y[0] = logf(-1);
					return;
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
		if (Nsteps >= Max_no_steps)
		{
			y[0] = logf(-1);
			return;
		}
		if (0.1 * fabs(H) <= fabs(t) * Roundoff)
		{
			y[0] = logf(-1);
			return;
		}
		if (FirstStep || !StartNewton) {
			safe_memset3(Z1, Z2, Z3, 0);
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

		NEWTON_UNROLLER
		for (; NewtonIter < NewtonMaxit; NewtonIter++) {
			RK_PrepareRHS(t, pr, H, y, F0, Z1, Z2, Z3, DZ1, DZ2, DZ3);
			RK_Solve(H, E1, E2, DZ1, DZ2, DZ3, ipiv1, ipiv2);
			double d1 = RK_ErrorNorm(sc, DZ1);
			double d2 = RK_ErrorNorm(sc, DZ2);
			double d3 = RK_ErrorNorm(sc, DZ3);
			double NewtonIncrement = sqrt((d1 * d1 + d2 * d2 + d3 * d3) / 3.0);

			Theta = ThetaMin;
			if (NewtonIter > 0) 
			{
				Theta = NewtonIncrement / NewtonIncrementOld;
				if(Theta >= 0.99) //! Non-convergence of Newton: Theta too large
					break;
				else
					NewtonRate = Theta / (1.0 - Theta);
				//Predict error at the end of Newton process 
				double NewtonPredictedErr = (NewtonIncrement * pow(Theta, (NewtonMaxit - NewtonIter - 1))) / (1.0 - Theta);
				if(NewtonPredictedErr >= NewtonTol) {
					//Non-convergence of Newton: predicted error too large
					double Qnewton = fmin(10.0, NewtonPredictedErr / NewtonTol);
                    Fac = 0.8 * pow(Qnewton, -1.0/((double)(NewtonMaxit-NewtonIter)));
                    break;
				}
			}

			NewtonIncrementOld = fmax(NewtonIncrement, Roundoff);
            // Update solution
            #pragma unroll 1
            for (int i = 0; i < NSP; i++)
            {
            	Z1[i] -= DZ1[i];
            	Z2[i] -= DZ2[i];
            	Z3[i] -= DZ3[i];
            }

            NewtonDone = (NewtonRate * NewtonIncrement <= NewtonTol);
#ifndef NEWTON_UNROLL
            if (NewtonDone) break;
#else //only break if it's at the end of the unroll
            if(NewtonDone && (NewtonIter + 1) % NEWTON_UNROLL == 0) break;
#endif
            if (NewtonIter >= NewtonMaxit)
            {
				y[0] = logf(-1);
				return;
			}
		}
		if(!NewtonDone) {
			H = Fac * H;
			Reject = true;
			SkipJac = true;
			SkipLU = false;
			continue;
		}
#ifdef SDIRK_ERROR
		//!~~~>   Prepare the loop-independent part of the right-hand side
		//!       G = H*rkBgam(0)*F0 + rkTheta(1)*Z1 + rkTheta(2)*Z2 + rkTheta(3)*Z3
		#pragma unroll 1
		for (int i = 0; i < NSP; i++) {
			Z4[i] = Z3[i];
			G[i] = rkBgam[0]*F0[i]*H + rkTheta[0] * Z1[i] + rkTheta[1] * Z2[i] + rkTheta[2] * Z3[i];
		}
		NewtonDone = false;
        Fac = 0.5; // ! Step reduction factor if too many iterations
        double NewtonIncrement = 0;
		
		NEWTON_UNROLLER
        for (int sNewtonIter = 0; sNewtonIter < NewtonMaxit; sNewtonIter++) {
        	//!~~~>   Prepare the loop-dependent part of the right-hand side
        	WADD(y, Z4, TMP);
        	dydt(t + H, pr, TMP, DZ4);
        	#pragma unroll 1
        	for(int i = 0; i < NSP; i++){
        		DZ4[i] += (rkGamma / H) * (G[i] - Z4[i]);
        	}
        	//Solve the linear system
        	dgetrs(E1, DZ4, ipiv1);
        	//Check convergence of Newton iterations
        	NewtonIncrement = RK_ErrorNorm(sc,DZ4);
        	double sNewtonRate = 2.0;
        	double ThetaSD = ThetaMin;
        	if (sNewtonIter > 0) {
            	ThetaSD = NewtonIncrement/NewtonIncrementOld;
            	if (ThetaSD < 0.99) {
            		sNewtonRate = ThetaSD/(1.0-ThetaSD);
                    //! Predict error at the end of Newton process 
                    double NewtonPredictedErr = (NewtonIncrement * pow(ThetaSD, (NewtonMaxit - sNewtonIter - 1))) / (1.0 - Theta);
                    if (NewtonPredictedErr >= NewtonTol) {
                    	//! Non-convergence of Newton: predicted error too large
						double Qnewton = fmin(10.0, NewtonPredictedErr / NewtonTol);
	                    Fac = 0.8 * pow(Qnewton, -1.0/((double)(NewtonMaxit-sNewtonIter)));
	                    break;
                    }
            	}
            	else
            	{
            		//! Non-convergence of Newton: predicted error too large
            		break;
            	}
            }
            NewtonIncrementOld = NewtonIncrement;
            //! Update solution: Z4 <-- Z4 + DZ4
            #pragma unroll 1
            for (int i = 0; i < NSP; i++) {
            	Z4[i] += DZ4[i];
            }

#ifndef NEWTON_UNROLL
            if (NewtonDone) break;
#else //only break if it's at the end of the unroll
            if (NewtonDone && (sNewtonIter + 1) % NEWTON_UNROLL == 0) break;
#endif
        }
        if (!NewtonDone) {
        	H = Fac*H;
        	Reject = true;
        	SkipJac = true;
        	SkipLU = false;
        	continue;
		}
#endif
#ifdef SDIRK_ERROR
		#pragma unroll 1
		for (int i = 0; i < NSP; i++) {
			DZ4[i] = Z3[i] - Z4[i];
		}
		double Err = RK_ErrorNorm(sc, DZ4);
#else
		double Err = RK_ErrorEstimate(H, t, pr, y, F0, Z1, Z2, Z3, sc, E1, ipiv1, FirstStep, Reject);
#endif
		//!~~~> Computation of new step size Hnew
		Fac = pow(Err, (-1.0 / rkELO)) * (1.0 + 2 * NewtonMaxit) / (NewtonIter + 1 + 2 * NewtonMaxit);
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
			#pragma unroll 1
			for (int i = 0; i < NSP; i++) {
				y[i] += Z3[i];
			}
			// Construct the solution quadratic interpolant Q(c_i) = Z_i, i=1:3
			if (StartNewton) {
				RK_Make_Interpolate(Z1, Z2, Z3, CONT);
			}
			scale(y, y0, sc);
			safe_memcpy(y0, y);
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
	}
}