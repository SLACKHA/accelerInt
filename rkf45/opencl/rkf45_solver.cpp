#include "rkf45_solver.hpp"

namespace opencl_solvers
{
    void RKF45Integrator::init(rk_t* rk)
    {
        // init the rk struct
        rk->max_iters = _options.maxIters();
        rk->min_iters = _options.minIters();
        rk->adaption_limit = 4;
        rk->s_rtol = _options.rtol();
        rk->s_atol = _options.atol();
    }
}
