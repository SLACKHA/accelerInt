#ifndef __rk_h
#define __rk_h

#include <cl_macros.h>
#include <cklib.h>

#ifdef __cplusplus
extern "C"
{
#endif

//enum { __rk_max_neq = 53 };

typedef struct //_rk_s
{
   int neq;

   int itol;
   double s_rtol, s_atol;
   //double v_rtol[__rk_max_neq], v_atol[__rk_max_neq];

   int max_iters, min_iters;

   double adaption_limit;

   double h_min, h_max;
   double t_round;
   double t_stop;

   // Private for each ...
   //double h;
   //int iter, nst, nfe;

   //int lenrwk;
   //double *rwk;
}
rk_t;

typedef struct //_rk_counters_s
{
   int niters, nsteps;
}
rk_counters_t;

//typedef int (*RHS_Function_t)(int neq, double tcur, double *y, double *ydot, void *user_data);
typedef void* RHS_Function_t;


int rk_create (__global rk_t *rk, const int neq);
int rk_lenrwk (__global const rk_t *rk);
int rk_destroy (__global rk_t *rk); 
int rk_init (__global rk_t *rk, const double t0, const double t_stop);

//int rk_callback (int neq, double tcur, __global double *y, __global double *ydot, __private void *user_data);

int rk_solve (__global const rk_t *rk, double *tcur, double *hnext, __private rk_counters_t *counters, __global double y[], __global double rwk[], RHS_Function_t func, __private void *user_data);

#define RK_SUCCESS		(0)
#define RK_TOO_MUCH_WORK	(-1)
#define RK_TDIST_TOO_SMALL	(-2)
#define RK_HIN_MAX_ITERS	(-3)

#ifdef __cplusplus
}
#endif

#endif
