#ifndef RK_TYPES_H
#define RK_TYPES_H

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct
{
    double s_rtol, s_atol;
    int max_iters, min_iters;
    double adaption_limit;
} rk_t;


typedef struct
{
    int niters;
    int nsteps;
} rk_counters_t;


#ifdef __OPENCL_VERSION__
//! \brief struct containing information on steps / iterations
typedef struct
{
    int niters;
    __MaskType nsteps;
} rk_counters_t_vec;

#define solver_type rk_t
#define counter_type rk_counters_t
#define counter_type_vec rk_counters_t_vec

__IntType rk_solve (__global const rk_t * __restrict__ rk,
                    __private __ValueType const t_start,
                    __private __ValueType const t_end,
                    __ValueType* hcur,
                    __private rk_counters_t_vec * __restrict__ counters,
                    __global __ValueType* __restrict__ y,
                    __global __ValueType* rwk,
                    __global __IntType* __restrict__ iwk,
                    __global __ValueType const * __restrict__ user_data,
                    const int driver_offset);

#define solver_function rk_solve
#endif

#ifdef __cplusplus
}
#endif

#endif
