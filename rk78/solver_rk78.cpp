/**
 * \file
 *
 * \author Nicholas J. Curtis
 * \date 04/29/2016
 *
 * \brief Defines an interface for boost's runge_kutta_fehlberg78 solver
 *
*/

//wrapper code
#include "rk78_typedefs.hpp"

//boost includes
#include <boost/numeric/odeint/integrate/integrate_adaptive.hpp>
extern "C" {
#include "solver.h"
}

#ifdef GENERATE_DOCS
namespace rk78 {
#endif

extern std::vector<state_type*> state_vectors;
extern std::vector<rhs_eval*> evaluators;
extern std::vector<controller> controllers;

extern "C" void intDriver(const int, const double, const double, const double*, double*);

#ifdef STIFFNESS_MEASURE
    extern std::vector<double> max_stepsize;
    controlled_step_result test_step(int index, const state_type& y, const double t, state_type& y_out, const double dt)
    {
        try
        {
            double t_copy = t;
            double dt_copy = dt;
            return controllers[index]->try_step(*evaluators[index], y, t_copy, y_out, dt_copy);
        }
        catch(...)
        {
            return fail;
        }
    }
#endif

/**
 * \brief Integration driver for the CPU integrators
 * \param[in]       NUM         the number of IVPs to solve
 * \param[in]       t           the current IVP time
 * \param[in]       t_end       the time to integrate the IVP to
 * \param[in]       pr_global   the pressure value for the IVPs
 * \param[in, out]  y_global    the state vectors
 *
 * The integration driver for the RK78 solver
 */
void intDriver (const int NUM, const double t, const double t_end,
                const double *pr_global, double *y_global)
{
    #ifdef STIFFNESS_MEASURE
    max_stepsize.clear();
    max_stepsize.resize(NUM, 0.0);
    #endif

	int tid = 0;
#ifdef STIFFNESS_MEASURE
    #pragma omp parallel for shared(state_vectors, evaluators, controllers, max_stepsize) private(tid)
#else
	#pragma omp parallel for shared(state_vectors, evaluators, controllers) private(tid)
#endif
    for (tid = 0; tid < NUM; ++tid) {
    	int index = omp_get_thread_num();

        // local array with initial values
        state_type& vec = *state_vectors[index];
        evaluators[index]->set_state_var(pr_global[tid]);

        // load local array with initial values from global array
        for (int i = 0; i < NSP; i++)
        {
            vec[i] = y_global[tid + i * NUM];
        }

#ifndef STIFFNESS_MEASURE
        integrate_adaptive(controllers[index],
            *evaluators[index], vec, t, t_end, t_end - t);
#else
        double tol = 1e-15;
        state_type y_copy(vec);
        //do a binary search to find the maximum stepsize
        double left_size = 1.0;
        while (test_step(index, vec, t, y_copy, left_size) == success)
        {
            left_size *= 10.0;
        }
        double right_size = 1e-20;
        while (test_step(index, vec, t, y_copy, right_size) == fail)
        {
            right_size /= 10.0;
        }
        double delta = 1.0;
        double mid = 0;
        while (delta > tol) {
            mid = (left_size + right_size) / 2.0;
            controlled_step_result result = test_step(index, vec, t, y_copy, mid);
            if (result == fail) {
                //mid becomes the new left
                delta = fabs(left_size - mid) / left_size;
                left_size = mid;
            }
            else{
                delta = fabs(right_size - mid) / right_size;
                right_size = mid;
            }
        }
        max_stepsize[tid] = mid;
#endif

        // update global array with integrated values
        for (int i = 0; i < NSP; i++)
        {
            y_global[tid + i * NUM] = vec[i];
        }

    }
}

#ifdef GENERATE_DOCS
}
#endif
