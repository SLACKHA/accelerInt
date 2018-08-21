/**
 * \file
 * \author Kyle Niemeyer
 * \date 06/28/2017
 * \brief Defines interface for Runge-Kutta-Chebyshev (RKC) solver
 *
 * Modified from code originally written for KE Niemeyer and CJ Sung,
 * "Accelerating moderately stiff chemical kinetics in reactive-flow simulations
 * using GPUs," J Comput Phys 256 (2014) 854-871.
 * [doi:10.1016/j.jcp.2013.09.025](https://doi.org/10.1016/j.jcp.2013.09.025)
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

#include "rkc.h"
#include "dydt.h"
#include "solver_options.h"

#ifdef GENERATE_DOCS
namespace rkc {
#endif

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
Real rkc_spec_rad (const Real t, const Real pr, const Real hmax, const Real* y,
                   const Real* F, Real* v, Real* Fv) {

    const int itmax = 50;
    Real small = ONE / hmax;

    Real nrm1 = ZERO;
    Real nrm2 = ZERO;
    for (int i = 0; i < NSP; ++i) {
        nrm1 += (y[i] * y[i]);
        nrm2 += (v[i] * v[i]);
    }
    nrm1 = sqrt(nrm1);
    nrm2 = sqrt(nrm2);

    Real dynrm;
    if ((nrm1 != ZERO) && (nrm2 != ZERO)) {
        dynrm = nrm1 * sqrt(UROUND);
        for (int i = 0; i < NSP; ++i) {
            v[i] = y[i] + v[i] * (dynrm / nrm2);
        }
    } else if (nrm1 != ZERO) {
        dynrm = nrm1 * sqrt(UROUND);
        for (int i = 0; i < NSP; ++i) {
            v[i] = y[i] * (ONE + sqrt(UROUND));
        }
    } else if (nrm2 != ZERO) {
        dynrm = UROUND;
        for (int i = 0; i < NSP; ++i) {
            v[i] *= (dynrm / nrm2);
        }
    } else {
        dynrm = UROUND;
        for (int i = 0; i < NSP; ++i) {
            v[i] = UROUND;
        }
    }

    // now iterate using nonlinear power method
    Real sigma = ZERO;
    for (int iter = 1; iter <= itmax; ++iter) {

        dydt (t, pr, v, Fv);

        nrm1 = ZERO;
        for (int i = 0; i < NSP; ++i) {
            nrm1 += ((Fv[i] - F[i]) * (Fv[i] - F[i]));
        }
        nrm1 = sqrt(nrm1);
        nrm2 = sigma;
        sigma = nrm1 / dynrm;
        if ((iter >= 2) && (fabs(sigma - nrm2) <= (fmax(sigma, small) * P01))) {
            for (int i = 0; i < NSP; ++i) {
                v[i] = v[i] - y[i];
            }
            return (ONEP2 * sigma);
        }

        if (nrm1 != ZERO) {
            for (int i = 0; i < NSP; ++i) {
                v[i] = y[i] + ((Fv[i] - F[i]) * (dynrm / nrm1));
            }
        } else {
            int ind = (iter % NSP);
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
void rkc_step (const Real t, const Real pr, const Real h, const Real* y_0, const Real* F_0,
               const int s, Real* y_j) {

    const Real w0 = ONE + TWO / (13.0 * (Real)(s * s));
    Real temp1 = (w0 * w0) - ONE;
    Real temp2 = sqrt(temp1);
    Real arg = (Real)(s) * log(w0 + temp2);
    const Real w1 = sinh(arg) * temp1 / (cosh(arg) * (Real)(s) * temp2 - w0 * sinh(arg));

    Real b_jm1 = ONE / (FOUR * (w0 * w0));
    Real b_jm2 = b_jm1;

    Real y_jm1[NSP];
    Real y_jm2[NSP];

      // calculate y_1
    Real mu_t = w1 * b_jm1;
    for (int i = 0; i < NSP; ++i) {
        y_jm2[i] = y_0[i];
        y_jm1[i] = y_0[i] + (mu_t * h * F_0[i]);
    }

    Real c_jm2 = ZERO;
    Real c_jm1 = mu_t;
    Real zjm1 = w0;
    Real zjm2 = ONE;
    Real dzjm1 = ONE;
    Real dzjm2 = ZERO;
    Real d2zjm1 = ZERO;
    Real d2zjm2 = ZERO;

    for (int j = 2; j <= s; ++j) {

        Real zj = TWO * w0 * zjm1 - zjm2;
        Real dzj = TWO * w0 * dzjm1 - dzjm2 + TWO * zjm1;
        Real d2zj = TWO * w0 * d2zjm1 - d2zjm2 + FOUR * dzjm1;
        Real b_j = d2zj / (dzj * dzj);
        Real gamma_t = ONE - (zjm1 * b_jm1);

        Real nu = -b_j / b_jm2;
        Real mu = TWO * b_j * w0 / b_jm1;
        mu_t = mu * w1 / w0;

          // calculate derivative, use y array for temporary storage
        dydt (t + (h * c_jm1), pr, y_jm1, y_j);

        for (int i = 0; i < NSP; ++i) {
            y_j[i] = (ONE - mu - nu) * y_0[i] + (mu * y_jm1[i]) + (nu * y_jm2[i])
                 + h * mu_t * (y_j[i] - (gamma_t * F_0[i]));
        }
        Real c_j = (mu * c_jm1) + (nu * c_jm2) + mu_t * (ONE - gamma_t);

        if (j < s) {
            for (int i = 0; i < NSP; ++i) {
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
// * \param[in] task      0 to take a single integration step, 1 to integrate to tEnd.
 * \param[in,out] y     Dependent variable array, integrated values replace initial conditions.
 */
int integrate (Real t, const Real tEnd, const Real pr, Real* y) {

    int nstep = 0;
    Real work[4 + NSP] = {0};

    int m_max = (int)(round(sqrt(RTOL / (10.0 * UROUND))));

    if (m_max < 2) {
        m_max = 2;
    }

    Real y_n[NSP];
    for (int i = 0; i < NSP; ++i) {
        y_n[i] = y[i];
    }

    // calculate F_n for initial y
    Real F_n[NSP];
    dydt (t, pr, y_n, F_n);

    // load initial estimate for eigenvector
    if (work[2] < UROUND) {
        for (int i = 0; i < NSP; ++i) {
            work[4 + i] = F_n[i];
        }
    }

    const Real hmax = fabs(tEnd - t);
    Real hmin = TEN * UROUND * fmax(fabs(t), hmax);

    while (t < tEnd) {
        // use time step stored in work[2]

        Real temp_arr[NSP];
        Real temp_arr2[NSP];
        Real err;

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
            work[2] = fmax(work[2], hmin);

            for (int i = 0; i < NSP; ++i) {
                temp_arr[i] = y_n[i] + (work[2] * F_n[i]);
            }
            dydt (t + work[2], pr, temp_arr, temp_arr2);

            err = ZERO;
            for (int i = 0; i < NSP; ++i) {
                Real est = (temp_arr2[i] - F_n[i]) / (ATOL + RTOL * fabs(y_n[i]));
                err += est * est;
            }
            err = work[2] * sqrt(err / NSP);

            if ((P1 * work[2]) < (hmax * sqrt(err))) {
                work[2] = fmax(P1 * work[2] / sqrt(err), hmin);
            } else {
                work[2] = hmax;
            }
        }

        // check if last step
        if ((ONEP1 * work[2]) >= fabs(tEnd - t)) {
            work[2] = fabs(tEnd - t);
        }

        // calculate number of steps
        int m = 1 + (int)(sqrt(ONEP54 * work[2] * work[3] + ONE));

        if (m > m_max) {
            m = m_max;
            work[2] = (Real)(m * m - 1) / (ONEP54 * work[3]);
        }

        hmin = TEN * UROUND * fmax(fabs(t), fabs(t + work[2]));

        // perform tentative time step
        rkc_step (t, pr, work[2], y_n, F_n, m, y);

        // calculate F_np1 with tenative y_np1
        dydt (t + work[2], pr, y, temp_arr);

        // estimate error
        err = ZERO;
        for (int i = 0; i < NSP; ++i) {
            Real est = P8 * (y_n[i] - y[i]) + P4 * work[2] * (F_n[i] + temp_arr[i]);
            est /= (ATOL + RTOL * fmax(fabs(y[i]), fabs(y_n[i])));
            err += est * est;
        }
        err = sqrt(err / ((Real)NSP));

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

            Real fac = TEN;
            Real temp1, temp2;

            if (work[1] < UROUND) {
                temp2 = pow(err, ONE3RD);
                if (P8 < (fac * temp2)) {
                    fac = P8 / temp2;
                }
            } else {
                temp1 = P8 * work[2] * pow(work[0], ONE3RD);
                temp2 = work[1] * pow(err, TWO3RD);
                if (temp1 < (fac * temp2)) {
                    fac = temp1 / temp2;
                }
            }

            // set "old" values to those for current time step
            work[0] = err;
            work[1] = work[2];

            for (int i = 0; i < NSP; ++i) {
                y_n[i] = y[i];
                F_n[i] = temp_arr[i];
            }

            // store next time step
            work[2] *= fmax(P1, fac);
            work[2] = fmax(hmin, fmin(hmax, work[2]));

            /* currently not supported
            if (task == 0) {
                // only perform one step
                return;
            }*/
        }

    }

    return EC_success;

} // rkc_driver
