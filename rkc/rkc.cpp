/**
 * \file
 * \author Kyle Niemeyer
 * \date 06/28/2017
 * \brief Defines interface for Runge-Kutta-Chebyshev (RKC) solver
 *
 * Code originally written for KE Niemeyer and CJ Sung,
 * "Accelerating moderately stiff chemical kinetics in reactive-flow simulations
 * using GPUs," J Comput Phys 256 (2014) 854-871.
 * [doi:10.1016/j.jcp.2013.09.025](https://doi.org/10.1016/j.jcp.2013.09.025)
 *
 * Modified to fit into accelerInt by Nicholas Curtis, 09/20/18
 *
 * Original sources:
 * - BP Sommeijer, LF Shampine, JG Verwer, "RKC: An explicit solver for parabolic PDEs,"
 *   J Comput Appl Math 88 (1997) 315-326.
 *   [doi:10.1016/S0377-0427(97)00219-7](https://doi.org/10.1016/S0377-0427(97)00219-7)
 * - JG Verwer, BP Sommeijer, W Hundsdorfer, "RKC time-stepping for advection–diffusion–reaction
 *   problems,"" J Comput Phys 201 (2004) 61-79.
 *   [doi:10.1016/j.jcp.2004.05.002](https://doi.org/10.1016/j.jcp.2004.05.002)
 *
 */

#include "rkc_solver.hpp"
#include <cstring>

namespace c_solvers
{

    /**
     * \brief Function to estimate spectral radius.
     *
     * \param[in] t     the time.
     * \param[in] pr    A parameter used for pressure or density to pass to the derivative function.
     * \param[in] hmax  Max time step size.
     * \param[in] y     Array of dependent variable.
     * \param[in] F     Derivative evaluated at current state
     * \param[in,out] v Array for eigenvectors
     * \param[out] Fv   Array for derivative evaluations
     */
    double RKCIntegrator::rkc_spec_rad (const double t, const double pr, const double hmax,
                                        const double* __restrict__ y, const double* __restrict__ F,
                                        double* __restrict__ v, double* __restrict__ Fv) {

        const int itmax = 50;
        double small = ONE / hmax;

        double nrm1 = ZERO;
        double nrm2 = ZERO;
        for (int i = 0; i < _neq; ++i) {
            nrm1 += (y[i] * y[i]);
            nrm2 += (v[i] * v[i]);
        }
        nrm1 = std::sqrt(nrm1);
        nrm2 = std::sqrt(nrm2);

        double dynrm;
        if ((nrm1 != ZERO) && (nrm2 != ZERO)) {
            dynrm = nrm1 * std::sqrt(UROUND);
            for (int i = 0; i < _neq; ++i) {
                v[i] = y[i] + v[i] * (dynrm / nrm2);
            }
        } else if (nrm1 != ZERO) {
            dynrm = nrm1 * std::sqrt(UROUND);
            for (int i = 0; i < _neq; ++i) {
                v[i] = y[i] * (ONE + std::sqrt(UROUND));
            }
        } else if (nrm2 != ZERO) {
            dynrm = UROUND;
            for (int i = 0; i < _neq; ++i) {
                v[i] *= (dynrm / nrm2);
            }
        } else {
            dynrm = UROUND;
            for (int i = 0; i < _neq; ++i) {
                v[i] = UROUND;
            }
        }

        // now iterate using nonlinear power method
        double sigma = ZERO;
        for (int iter = 1; iter <= itmax; ++iter) {

            dydt (t, pr, v, Fv);

            nrm1 = ZERO;
            for (int i = 0; i < _neq; ++i) {
                nrm1 += ((Fv[i] - F[i]) * (Fv[i] - F[i]));
            }
            nrm1 = std::sqrt(nrm1);
            nrm2 = sigma;
            sigma = nrm1 / dynrm;
            if ((iter >= 2) && (std::fabs(sigma - nrm2) <= (std::fmax(sigma, small) * P01))) {
                for (int i = 0; i < _neq; ++i) {
                    v[i] = v[i] - y[i];
                }
                return (ONEP2 * sigma);
            }

            if (nrm1 != ZERO) {
                for (int i = 0; i < _neq; ++i) {
                    v[i] = y[i] + ((Fv[i] - F[i]) * (dynrm / nrm1));
                }
            } else {
                int ind = (iter % _neq);
                v[ind] = y[ind] - (v[ind] - y[ind]);
            }

        }
        return (ONEP2 * sigma);
    }

    ///////////////////////////////////////////////////////

    /**
     * \brief Function to take a single RKC integration step
     *
     * \param[in] t    the starting time.
     * \param[in] pr   A parameter used for pressure or density to pass to the derivative function.
     * \param[in] h    Time-step size.
     * \param[in] y_0  Initial conditions.
     * \param[in] F_0  Derivative function at initial conditions.
     * \param[in] s    number of steps.
     * \param[out] y_j  Integrated variables.
     */
    void RKCIntegrator::rkc_step (const double t, const double pr, const double h,
                                  const double* y_0, const double* F_0, const int s,
                                  double* __restrict__ y_j, double* __restrict__ y_jm1,
                                  double* __restrict__ y_jm2) {

        const double w0 = ONE + TWO / (13.0 * (double)(s * s));
        double temp1 = (w0 * w0) - ONE;
        double temp2 = std::sqrt(temp1);
        double arg = (double)(s) * std::log(w0 + temp2);
        const double w1 = std::sinh(arg) * temp1 / (std::cosh(arg) * (double)(s) * temp2 - w0 * std::sinh(arg));

        double b_jm1 = ONE / (FOUR * (w0 * w0));
        double b_jm2 = b_jm1;

          // calculate y_1
        double mu_t = w1 * b_jm1;
        for (int i = 0; i < _neq; ++i) {
            y_jm2[i] = y_0[i];
            y_jm1[i] = y_0[i] + (mu_t * h * F_0[i]);
        }

        double c_jm2 = ZERO;
        double c_jm1 = mu_t;
        double zjm1 = w0;
        double zjm2 = ONE;
        double dzjm1 = ONE;
        double dzjm2 = ZERO;
        double d2zjm1 = ZERO;
        double d2zjm2 = ZERO;

        for (int j = 2; j <= s; ++j) {

            double zj = TWO * w0 * zjm1 - zjm2;
            double dzj = TWO * w0 * dzjm1 - dzjm2 + TWO * zjm1;
            double d2zj = TWO * w0 * d2zjm1 - d2zjm2 + FOUR * dzjm1;
            double b_j = d2zj / (dzj * dzj);
            double gamma_t = ONE - (zjm1 * b_jm1);

            double nu = -b_j / b_jm2;
            double mu = TWO * b_j * w0 / b_jm1;
            mu_t = mu * w1 / w0;

              // calculate derivative, use y array for temporary storage
            dydt (t + (h * c_jm1), pr, y_jm1, y_j);

            for (int i = 0; i < _neq; ++i) {
                y_j[i] = (ONE - mu - nu) * y_0[i] + (mu * y_jm1[i]) + (nu * y_jm2[i])
                     + h * mu_t * (y_j[i] - (gamma_t * F_0[i]));
            }
            double c_j = (mu * c_jm1) + (nu * c_jm2) + mu_t * (ONE - gamma_t);

            if (j < s) {
                for (int i = 0; i < _neq; ++i) {
                    y_jm2[i] = y_jm1[i];
                    y_jm1[i] = y_j[i];
                }
            }

            c_jm2 = c_jm1;
            c_jm1 = c_j;
            b_jm2 = b_jm1;
            b_jm1 = b_j;
            zjm2 = zjm1;
            zjm1 = zj;
            dzjm2 = dzjm1;
            dzjm1 = dzj;
            d2zjm2 = d2zjm1;
            d2zjm1 = d2zj;
        }

    } // rkc_step

    /////////////////////////////////////////////////////////////

    /**
     * \brief Driver function for RKC integrator.
     *
     * \param[in,out] t     The time (starting, then ending).
     * \param[in] tEnd      The desired end time.
     * \param[in] pr        A parameter used for pressure or density to pass to the derivative function.
     * \param[in,out] y     Dependent variable array, integrated values replace initial conditions.
     */
    ErrorCode RKCIntegrator::integrate (double t, const double tEnd, const double pr, double* y) {

        int nstep = 0;
        int tid = omp_get_thread_num();

        double* __restrict__ work = _unique<double>(tid, _work);
        std::memset(work, 0, (4 + _neq) * sizeof(double));

        int m_max = (int)(std::round(std::sqrt(rtol() / (10.0 * UROUND))));

        if (m_max < 2) {
            m_max = 2;
        }

        double* __restrict__ y_n = _unique<double>(tid, _y_n);
        for (int i = 0; i < _neq; ++i) {
            y_n[i] = y[i];
        }

        // calculate F_n for initial y
        double* __restrict__ F_n = _unique<double>(tid, _F_n);
        dydt (t, pr, y_n, F_n);

        // load initial estimate for eigenvector
        if (work[2] < UROUND) {
            for (int i = 0; i < _neq; ++i) {
                work[4 + i] = F_n[i];
            }
        }

        const double hmax = std::fabs(tEnd - t);
        double hmin = TEN * UROUND * std::fmax(std::fabs(t), hmax);

        double* __restrict__ temp_arr = _unique<double>(tid, _temp_arr);
        double* __restrict__ temp_arr2 = _unique<double>(tid, _temp_arr2);

        double* __restrict__ y_jm1 = _unique<double>(tid, _y_jm1);
        double* __restrict__ y_jm2 = _unique<double>(tid, _y_jm2);

        while (t < tEnd) {
            // use time step stored in work[2]
            double err;

            // estimate Jacobian spectral radius
            // only if 25 steps passed
            if ((nstep % 25) == 0) {
                work[3] = rkc_spec_rad (t, pr, hmax, y_n, F_n, &work[4], temp_arr2);
            }

            // first step, estimate step size
            if (work[2] < UROUND) {
                work[2] = hmax;
                if ((work[3] * work[2]) > ONE) {
                    work[2] = ONE / work[3];
                }
                work[2] = std::fmax(work[2], hmin);

                for (int i = 0; i < _neq; ++i) {
                    temp_arr[i] = y_n[i] + (work[2] * F_n[i]);
                }
                dydt (t + work[2], pr, temp_arr, temp_arr2);

                err = ZERO;
                for (int i = 0; i < _neq; ++i) {
                    double est = (temp_arr2[i] - F_n[i]) / (atol() + rtol() * std::fabs(y_n[i]));
                    err += est * est;
                }
                err = work[2] * std::sqrt(err / _neq);

                if ((P1 * work[2]) < (hmax * std::sqrt(err))) {
                    work[2] = std::fmax(P1 * work[2] / std::sqrt(err), hmin);
                } else {
                    work[2] = hmax;
                }
            }

            // check if last step
            if ((ONEP1 * work[2]) >= std::fabs(tEnd - t)) {
                work[2] = std::fabs(tEnd - t);
            }

            // calculate number of steps
            int m = 1 + (int)(std::sqrt(ONEP54 * work[2] * work[3] + ONE));

            if (m > m_max) {
                m = m_max;
                work[2] = (double)(m * m - 1) / (ONEP54 * work[3]);
            }

            hmin = TEN * UROUND * std::fmax(std::fabs(t), std::fabs(t + work[2]));

            // perform tentative time step
            rkc_step (t, pr, work[2], y_n, F_n, m, y, y_jm1, y_jm2);

            // calculate F_np1 with tenative y_np1
            dydt (t + work[2], pr, y, temp_arr);

            // estimate error
            err = ZERO;
            for (int i = 0; i < _neq; ++i) {
                double est = P8 * (y_n[i] - y[i]) + P4 * work[2] * (F_n[i] + temp_arr[i]);
                est /= (atol() + rtol() * std::fmax(std::fabs(y[i]), std::fabs(y_n[i])));
                err += est * est;
            }
            err = std::sqrt(err / ((double)_neq));

            if (err > ONE) {
                // error too large, step is rejected

                // select smaller step size
                work[2] = P8 * work[2] / (pow(err, ONE3RD));

                // reevaluate spectral radius
                work[3] = rkc_spec_rad (t, pr, hmax, y_n, F_n, &work[4], temp_arr2);
            } else {
                // step accepted
                t += work[2];
                nstep++;

                double fac = TEN;

                if (work[1] < UROUND) {
                    double temp2 = pow(err, ONE3RD);
                    if (P8 < (fac * temp2)) {
                        fac = P8 / temp2;
                    }
                } else {
                    double temp1 = P8 * work[2] * pow(work[0], ONE3RD);
                    double temp2 = work[1] * pow(err, TWO3RD);
                    if (temp1 < (fac * temp2)) {
                        fac = temp1 / temp2;
                    }
                }

                // set "old" values to those for current time step
                work[0] = err;
                work[1] = work[2];

                for (int i = 0; i < _neq; ++i) {
                    y_n[i] = y[i];
                    F_n[i] = temp_arr[i];
                }

                // store next time step
                work[2] *= std::fmax(P1, fac);
                work[2] = std::fmax(hmin, std::fmin(hmax, work[2]));

                /* currently not supported
                if (task == 0) {
                    // only perform one step
                    return;
                }*/
            }

        }

        return ErrorCode::SUCCESS;

    } // rkc_driver

}
