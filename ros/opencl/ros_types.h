#ifndef RK_TYPES_H
#define RK_TYPES_H

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

#define solver_type ros_t

typedef struct
{
   int niters;
   int nst;
   int nfe;
   int nje;
   int nlu;
}
ros_counters_t;

#define counter_type ros_counters_t

#ifdef __OPENCL_VERSION__
//! \brief struct containing information on steps / iterations
typedef struct
{
    int niters;
    __MaskType nst;
    __MaskType nfe;
    __MaskType nje;
    __MaskType nlu;
}
ros_counters_t_vec;

#define counter_type_vec ros_counters_t_vec

__IntType ros_solve (__global const rk_t * __restrict__ rk,
                     __private __ValueType const t_start,
                     __private __ValueType const t_end,
                     __private __ValueType hcur,
                     __private rk_counters_t_vec * __restrict__ counters,
                     __global __ValueType* __restrict__ y,
                     __global __ValueType* __restrict__ rwk,
                     __global __ValueType const * __restrict__ user_data);

#define solver_function ros_solve
#endif


#ifndef RK_TYPES_H
#define RK_TYPES_H

#ifdef __cplusplus
}
#endif
