/**
 * \file
 * \brief the generic integration driver for the CPU solvers
 *
 * \author Nicholas Curtis
 * \date 03/10/2015
 *
 */

#include "solver.hpp"

namespace c_solvers {

    /**
     * \brief Integration driver for the CPU integrators
     * \param[in]       NUM             The (non-padded) number of IVPs to integrate
     * \param[in]       t               The current system time
     * \param[in]       t_end           The IVP integration end time
     * \param[in]       pr_global       The system constant variable (pressures / densities)
     * \param[in,out]   y_global        The system state vectors at time t.
                                        Returns system state vectors at time t_end
     *
     * This is generic driver for CPU integrators
     */
    void Integrator::intDriver (const int NUM, const double t,
                                const double t_end, const double * __restrict__ pr_global,
                                double * __restrict__ y_global)
    {
        int ivp_index;

        #pragma omp parallel for shared(y_global, pr_global) private(ivp_index)
        for (ivp_index = 0; ivp_index < NUM; ++ivp_index) {

            // local array with initial values
            double* __restrict__ phi = this->_unique<double>(omp_get_thread_num(), _phi);
            double pr_local = pr_global[ivp_index];

            // load local array with initial values from global array

            for (int i = 0; i < this->_neq; i++)
            {
                phi[i] = y_global[ivp_index + i * NUM];
            }

            // call integrator for one time step
            ErrorCode err = this->integrate(t, t_end, pr_local, phi);
            this->checkError(ivp_index, err);

            // update global array with integrated values

            for (int i = 0; i < this->neq(); i++)
            {
                y_global[ivp_index + i * NUM] = phi[i];
            }

        } //end ivp_index loop

    } // end intDriver

    // template specializations
    template double* Integrator::_unique(int tid, std::size_t offset);
    template int* Integrator::_unique(int tid, std::size_t offset);
    template std::complex<double>* Integrator::_unique(int tid, std::size_t offset);

    //! return reference to the working memory allocated for the jacobian and source
    //! rates
    double* Integrator::rwk(int tid)
    {
        if (_ivp.passEntireWorkingBufferToIVP())
        {
            // pass whole working buffer
            return _unique<double>(0, _ivp_working);
        }
        else
        {
            return _unique<double>(tid, _ivp_working);
        }
    }

    ErrorCode Integrator::hinit(const double t, const double t_end,
                                const double* __restrict__ y,
                                const double user_data,
                                double* h0)
    {
        #define t_round ((t_end - t) * DBL_EPSILON)
        #define h_min (t_round * 100)
        #define h_max ((t_end - t) / minIters())

        if ((t_end - t) < 2 * t_round)
        {
            // requested time-step is smaller than roundoff
            return ErrorCode::TDIST_TOO_SMALL;
        }

        double* __restrict__ ydot = _unique<double>(tid, _ydot_hin);
        double* __restrict__ y1 = _unique<double>(tid, _y1_hin);
        double* __restrict__ ydot1 = _unique<double>(tid, _ydot1_hin);
        double* __restrict__ rwk = this->rwk(tid);
        double hlb = h_min;
        double hub = h_max;

        // Already done ...
        bool done = *h0 >= h_min;
        double hg = sqrt(hlb*hub);

        if (hub < hlb)
        {
            *h0 = done ? *h0 : hg;
            return ErrorCode::SUCCESS;
        }

        // Start iteration to find solution to ... {WRMS norm of (h0^2 y'' / 2)} = 1

        bool hnew_is_ok = false;
        double hnew = hg;
        const int miters = 10;
        int iter = 0;
        ErrorCode ierr = ErrorCode::SUCCESS;

        // compute ydot at t=t0
        dydt(t, user_data, y, ydot, rwk);

        // maximum of 2 iterations
        #define MAX_HINIT_ITERS (1)
        for(; iter <= MAX_HINIT_ITERS; ++iter)
        {
            // Estimate y'' with finite-difference ...
            //double t1 = hg;
            for (int k = 0; k < neq; k++)
            {
                y1[k] = y[k] + hg * ydot[k];
            }

            // compute y' at t1
            dydt(t, user_data, y1, ydot1, rwk);

            // Compute WRMS norm of y''
            for (int k = 0; k < neq; k++)
                y1[k] = (ydot1[k] - ydot[k]) / hg;

            __ValueType yddnrm = get_wnorm(solver, y1, y);

            // should we accept this?
                  // should we accept this?
            if (hnew_is_ok || iter == miters)
            {
                hnew = hg;
                ierr = (hnew_is_ok) ? ErrorCode::SUCCESS : ErrorCode::HIN_MAX_ITERS;
                break;
            }

            // Get the new value of h ...
            hnew = (yddnrm*hub*hub > 2.0) ? sqrt(2.0 / yddnrm) : sqrt(hg * hub);

            // test the stopping conditions.
            double hrat = hnew / hg;

            // Accept this value ... the bias factor should bring it within range.
            if ( (hrat > 0.5) && (hrat < 2.0) )
                hnew_is_ok = true;

            // If y'' is still bad after a few iterations, just accept h and give up.
            if (iter >= MAX_HINIT_ITERS)
            {
                hnew = hg;
                hnew_is_ok = true;
            }

            hg = hnew;
        }

        // bound and bias estimate
        *h0 = hnew * 0.5;
        *h0 = fmax(*h0, hlb);
        *h0 = fmin(*h0, hub);

        #undef t_round
        #undef h_min
        #undef h_max

        return ierr;
    }

}
