namespace opencl_solvers
{
    void RKF45Integrator::init(rk_t *rk)
    {
        // init the rk struct
        rk_vals->max_iters = _options.maxIters();
        rk_vals->min_iters = _options.minIters();
        rk_vals->adaption_limit = 4;
        rk_vals->s_rtol = _options.rtol();
        rk_vals->s_atol = _options.atol();
    }
}
