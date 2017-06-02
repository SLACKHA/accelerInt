#ifndef __sdirk_h
#define __sdirk_h

#include <cl_macros.h>
#include <cklib.h>

#ifdef __cplusplus
extern "C"
{
#endif

typedef enum { S4a } sdirk_solverTags;
enum { sdirk_maxStages = 5 };

typedef struct
{
   int neq;

   int itol;
   double s_rtol, s_atol;
   //double v_rtol[__sdirk_max_neq], v_atol[__sdirk_max_neq];

   int max_iters, min_iters;

   double adaption_limit;

   double h_min, h_max;
   double t_round;
   double t_stop;

   sdirk_solverTags solverTag;
   int numStages;
   int ELO;
   double A[sdirk_maxStages][sdirk_maxStages];
   double B[sdirk_maxStages];
   double Bhat[sdirk_maxStages];
   double C[sdirk_maxStages];
   double D[sdirk_maxStages];
   double E[sdirk_maxStages];
   double Theta[sdirk_maxStages][sdirk_maxStages];
   double Alpha[sdirk_maxStages][sdirk_maxStages];
   double gamma;

   //double NewtonThetaMin;	// Minimum convergence rate for the Newton Iteration (0.001)
   //double NewtonTolerance;	// Convergence criteria (0.03
}
sdirk_t;

typedef struct
{
   int niters;
   int nst;
   int nfe;
   int nje;
   int nlu;
   int nni;
}
sdirk_counters_t;

#if defined(__OPENCL_VERSION__)
typedef void* SDIRK_Function_t;
typedef void* SDIRK_Jacobian_t;
#else
typedef int (*SDIRK_Function_t)(int neq, double tcur, double *y, double *fy, void *user_data);
typedef int (*SDIRK_Jacobian_t)(int neq, double tcur, double *y, double *Jy, void *user_data);
#endif

int sdirk_create (__global sdirk_t *sdirk, const int neq, sdirk_solverTags solver_tag);
int sdirk_lenrwk (__global const sdirk_t *sdirk);
int sdirk_leniwk (__global const sdirk_t *sdirk);
int sdirk_destroy (__global sdirk_t *sdirk); 
int sdirk_init (__global sdirk_t *sdirk, const double t0, const double t_stop);

//int sdirk_callback (int neq, double tcur, __global double *y, __global double *ydot, __private void *user_data);

int sdirk_solve (__global const sdirk_t *sdirk, double *tcur, double *hcur, __private sdirk_counters_t *counters, __global double y[], __global int iwk[], __global double rwk[], SDIRK_Function_t func, SDIRK_Jacobian_t jac, __private void *user_data);

#define SDIRK_SUCCESS		(0)
#define SDIRK_ERROR		(-1)
#define SDIRK_TOO_MUCH_WORK	(-11)
#define SDIRK_TDIST_TOO_SMALL	(-12)
#define SDIRK_HIN_MAX_ITERS	(-13)

#ifdef __cplusplus
}
#endif

#endif
