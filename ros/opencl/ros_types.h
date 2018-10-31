#ifndef ROS_TYPES_H
#define ROS_TYPES_H

#ifdef __cplusplus
extern "C"
{
#endif

enum { ros_maxStages = 6 };

typedef struct
{
   int itol;
   double s_rtol, s_atol;
   //double v_rtol[__ros_max_neq], v_atol[__ros_max_neq];

   int max_iters, min_iters;

   double adaption_limit;

   int numStages;
   int ELO;
   double A[ros_maxStages*(ros_maxStages-1)/2];
   double C[ros_maxStages*(ros_maxStages-1)/2];
   int newFunc[ros_maxStages];
   double E[ros_maxStages];
   double M[ros_maxStages];
   double alpha[ros_maxStages];
   double gamma[ros_maxStages];
} ros_t;


typedef struct
{
   int niters;
   int nsteps;
   int nst;
   int nfe;
   int nje;
   int nlu;
} ros_counters_t;

#ifdef __OPENCL_VERSION__
//! \brief struct containing information on steps / iterations
typedef struct
{
    int niters;
    __MaskType nsteps;
    __MaskType nst;
    __MaskType nfe;
    __MaskType nje;
    __MaskType nlu;
} ros_counters_t_vec;

#define solver_type ros_t
#define counter_type ros_counters_t
#define counter_type_vec ros_counters_t_vec

__IntType ros_solve (__global const ros_t * __restrict__ rk,
                     __private __ValueType const t_start,
                     __private __ValueType const t_end,
                     __global __ValueType* hcur,
                     __private ros_counters_t_vec * __restrict__ counters,
                     __global __ValueType* __restrict__ y,
                     __global __ValueType* __restrict__ rwk,
                     __global __IntType* __restrict__ iwk,
                     __global __ValueType const * __restrict__ user_data,
                     const int driver_offset);

#define solver_function ros_solve
#endif


#ifdef __cplusplus
}
#endif

#endif
