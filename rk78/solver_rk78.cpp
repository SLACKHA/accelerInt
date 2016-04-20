//wrapper code
#include "rk78_typedefs.hpp"

//boost includes
#include <boost/numeric/odeint/integrate/integrate_adaptive.hpp>
extern "C" {
#include "solver.h"
}
extern std::vector<state_type*> state_vectors;
extern std::vector<rhs_eval*> evaluators;
extern std::vector<stepper*> steppers;

extern "C" void intDriver(const int, const double, const double, const double*, double*);

void intDriver (const int NUM, const double t, const double t_end,
                const double *pr_global, double *y_global)
{
	int tid = 0;
	#pragma omp parallel for shared(state_vectors, evaluators, steppers) private(tid)
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

        integrate_adaptive(make_controlled(ATOL, RTOL, *steppers[index]),
            *evaluators[index], vec, t, t_end, t_end - t);

        // update global array with integrated values
        
        for (int i = 0; i < NSP; i++)
        {
            y_global[tid + i * NUM] = vec[i];
        }

    }
}