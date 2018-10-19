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
            double* __restrict__ phi = this->_unique<double>(_phi, omp_get_thread_num());
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

}
