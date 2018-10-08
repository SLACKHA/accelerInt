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

#ifdef __cplusplus
}
#endif

#endif
