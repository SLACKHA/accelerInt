#ifndef __ros_h
#define __ros_h

#include <cl_macros.h>
#include <cklib.h>

#ifdef __cplusplus
extern "C"
{
#endif

typedef enum { Ros3, Ros4, Rodas3, Rodas4 } ros_solverTags;
enum { ros_maxStages = 6 };

typedef struct
{
   int neq;

   int itol;
   double s_rtol, s_atol;
   //double v_rtol[__ros_max_neq], v_atol[__ros_max_neq];

   int max_iters, min_iters;

   double adaption_limit;

   double h_min, h_max;
   double t_round;
   double t_stop;

   ros_solverTags solverTag;
   int numStages;
   int ELO;
   double A[ros_maxStages*(ros_maxStages-1)/2];
   double C[ros_maxStages*(ros_maxStages-1)/2];
   int newFunc[ros_maxStages];
   double E[ros_maxStages];
   double M[ros_maxStages];
   double alpha[ros_maxStages];
   double gamma[ros_maxStages];
}
ros_t;

typedef struct
{
   int niters;
   int nst;
   int nfe;
   int nje;
   int nlu;
}
ros_counters_t;

//#if defined(__OPENCL_VERSION__)
typedef void* ROS_Function_t;
typedef void* ROS_Jacobian_t;
//#else
//typedef int (*ROS_Function_t)(int neq, double tcur, double *y, double *fy, void *user_data);
//typedef int (*ROS_Jacobian_t)(int neq, double tcur, double *y, double *Jy, void *user_data);
//#endif

int ros_create (__global ros_t *ros, const int neq, ros_solverTags solver_tag);
int ros_lenrwk (__global const ros_t *ros);
int ros_leniwk (__global const ros_t *ros);
int ros_destroy (__global ros_t *ros); 
int ros_init (__global ros_t *ros, const double t0, const double t_stop);

//int ros_callback (int neq, double tcur, __global double *y, __global double *ydot, __private void *user_data);

int ros_solve (__global const ros_t *ros, double *tcur, double *hcur, __private ros_counters_t *counters, __global double y[], __global int iwk[], __global double rwk[], ROS_Function_t func, ROS_Jacobian_t jac, __private void *user_data);

#define ROS_SUCCESS		(0)
#define ROS_ERROR		(-1)
#define ROS_TOO_MUCH_WORK	(-11)
#define ROS_TDIST_TOO_SMALL	(-12)
#define ROS_HIN_MAX_ITERS	(-13)

#ifdef __cplusplus
}
#endif

#endif
