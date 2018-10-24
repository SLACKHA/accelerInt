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
#include "rk78_solver.hpp"

//boost includes
#include <boost/numeric/odeint/integrate/integrate_adaptive.hpp>

namespace c_solvers {


    /**
     *  \brief 5th-order Radau2A CPU implementation
     *
     *  \param[in]          t_start             The starting time
     *  \param[in]          t_end               The end integration time
     *  \param[in]          pr                  The system constant variable (pressure/density)
     *  \param[in,out]      y                   The system state vector at time `t_start`.
                                                Overwritten with the system state at time `t_end`
     *  \returns Return code, @see RK_ErrCodes
     */
    ErrorCode RK78Integrator::integrate (
        const double t_start, const double t_end, const double pr, double* __restrict__ y)
    {
        int index = omp_get_thread_num();

        // copy parameter into into evaluator
        evaluators[index]->set_state(pr, this->rwk(index));

        // copy into state vector
        std::memcpy(&state_vectors[index][0], y, _neq * sizeof(double));

        // and integrate
        double h = 0;
        hinit(t, t_end, y, pr, &h);
        integrate_adaptive(controllers[index], *evaluators[index], state_vectors[index], t_start, t_end, h);

        // copy out of state vector
        std::memcpy(y, &state_vectors[index][0], _neq * sizeof(double));

        return ErrorCode::SUCCESS;
    }
}
