#include "rkc.cuh"
#include "dydt.cuh"

#ifdef GENERATE_DOCS
namespace rkc_cu {
#endif

//#define SHARED

/////////////////////////////////////////////////////////

__device__
Real rkc_spec_rad (const Real t, const Real pr, const Real hmax, const Real* y,
                   const Real* F, Real* v, Real* Fv,
                   mechanism_memory const * const __restrict__ mech) {
   /**
    * Function to estimate spectral radius.
    *
    * @param t    the time.
    * @param pr   A parameter used for pressure or density to pass to the derivative function.
    * @param hmax Max time step size.
    * @param y    Array of dependent variable.
    * @param F    Derivative evaluated at current state
    * @param v
    * @param Fv
    */

    const int itmax = 50;
    Real small = ONE / hmax;

    //for (int i = 0; i < NSP; ++i) {
    //    v[i] = F[i];
    //}

    Real nrm1 = ZERO;
    Real nrm2 = ZERO;
    for (int i = 0; i < NSP; ++i) {
        nrm1 += (y[INDEX(i)] * y[INDEX(i)]);
        nrm2 += (v[INDEX(i)] * v[INDEX(i)]);
    }
    nrm1 = sqrt(nrm1);
    nrm2 = sqrt(nrm2);

    Real dynrm;
    if ((nrm1 != ZERO) && (nrm2 != ZERO)) {
        dynrm = nrm1 * sqrt(UROUND);
        for (int i = 0; i < NSP; ++i) {
            v[INDEX(i)] = y[INDEX(i)] + v[INDEX(i)] * (dynrm / nrm2);
        }
    } else if (nrm1 != ZERO) {
        dynrm = nrm1 * sqrt(UROUND);
        for (int i = 0; i < NSP; ++i) {
            v[INDEX(i)] = y[INDEX(i)] * (ONE + sqrt(UROUND));
        }
    } else if (nrm2 != ZERO) {
        dynrm = UROUND;
        for (int i = 0; i < NSP; ++i) {
            v[INDEX(i)] *= (dynrm / nrm2);
        }
    } else {
        dynrm = UROUND;
        for (int i = 0; i < NSP; ++i) {
            v[INDEX(i)] = UROUND;
        }
    }

    // now iterate using nonlinear power method
    Real sigma = ZERO;
    for (int iter = 1; iter <= itmax; ++iter) {

        dydt (t, pr, v, Fv, mech);

        nrm1 = ZERO;
        for (int i = 0; i < NSP; ++i) {
            nrm1 += ((Fv[INDEX(i)] - F[INDEX(i)]) * (Fv[INDEX(i)] - F[INDEX(i)]));
        }
        nrm1 = sqrt(nrm1);
        nrm2 = sigma;
        sigma = nrm1 / dynrm;

        nrm2 = fabs(sigma - nrm2) / sigma;
        if ((iter >= 2) && (fabs(sigma - nrm2) <= (fmax(sigma, small) * P01))) {
            for (int i = 0; i < NSP; ++i) {
                v[INDEX(i)] -= y[INDEX(i)];
            }
            return (ONEP2 * sigma);
        }

        if (nrm1 != ZERO) {
            for (int i = 0; i < NSP; ++i) {
                v[INDEX(i)] = y[INDEX(i)] + ((Fv[INDEX(i)] - F[INDEX(i)]) * (dynrm / nrm1));
            }
        } else {
            int ind = (iter % NSP);
            v[INDEX(ind)] = y[INDEX(ind)] - (v[INDEX(ind)] - y[INDEX(ind)]);
        }
    }
    return (ONEP2 * sigma);
}

///////////////////////////////////////////////////////

__device__
void rkc_step (const Real t, const Real pr, const Real h, const Real* y_0,
               const Real* F_0, const int s, Real* y_j,
               Real* y_jm1, Real* y_jm2,
               mechanism_memory const * const __restrict__ mech) {
   /**
    * Function to take a single RKC integration step
    *
    * @param t    the starting time.
    * @param pr   A parameter used for pressure or density to pass to the derivative function.
    * @param h    Time-step size.
    * @param y_0  Initial conditions.
    * @param F_0  Derivative function at initial conditions.
    * @param s    number of steps.
    * @param y_j  Integrated variables.
    */

    const Real w0 = ONE + TWO / (13.0 * (Real)(s * s));
    Real temp1 = (w0 * w0) - ONE;
    Real temp2 = sqrt(temp1);
    const Real arg = (Real)(s) * log(w0 + temp2);
    const Real w1 = sinh(arg) * temp1 / (cosh(arg) * (Real)(s) * temp2 - w0 * sinh(arg));

    Real b_jm1 = ONE / (FOUR * (w0 * w0));
    Real b_jm2 = b_jm1;

    //Real y_jm1[NSP];
    //Real y_jm2[NSP];

      // calculate y_1
    Real mu_t = w1 * b_jm1;
    for (int i = 0; i < NSP; ++i) {
        y_jm2[INDEX(i)] = y_0[INDEX(i)];
        y_jm1[INDEX(i)] = y_0[INDEX(i)] + (mu_t * h * F_0[INDEX(i)]);
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
        dydt (t + (h * c_jm1), pr, y_jm1, y_j, mech);

        for (int i = 0; i < NSP; ++i) {
            y_j[INDEX(i)] = (ONE - mu - nu) * y_0[INDEX(i)] + (mu * y_jm1[INDEX(i)]) + (nu * y_jm2[INDEX(i)])
                 + h * mu_t * (y_j[INDEX(i)] - (gamma_t * F_0[INDEX(i)]));
        }
        Real c_j = (mu * c_jm1) + (nu * c_jm2) + mu_t * (ONE - gamma_t);

        if (j < s) {
            for (int i = 0; i < NSP; ++i) {
                y_jm2[INDEX(i)] = y_jm1[INDEX(i)];
                y_jm1[INDEX(i)] = y_j[INDEX(i)];
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

__device__ void integrate (const Real tstart,
                            const Real tEnd,
                            const Real pr,
                            Real *y,
                            mechanism_memory const * const __restrict__ mech,
                            solver_memory const * const __restrict__ solver) {
   /**
    * Driver function for RKC integrator.
    *
    * @param t      the starting time.
    * @param tEnd   the desired end time.
    * @param pr     A parameter used for pressure or density to pass to the derivative function.
    * @param y      Dependent variable array, integrated values replace initial conditions.
    * @param mech   The mechanism_memory struct that contains the pre-allocated memory for the RHS \ Jacobian evaluation
    * @param solver The solver_memory struct that contains the pre-allocated memory for the solver
    */

    Real t = tstart;
    int nstep = 0;
    int mMax = (int)(round(sqrt(RTOL / (10.0 * UROUND))));

    if (mMax < 2) {
        mMax = 2;
    }

    Real * const __restrict__ y_n = solver->y_n;
    //Real y_n[INDEX(NSP)];
    for (int i = 0; i < NSP; ++i) {
        y_n[INDEX(i)] = y[INDEX(i)];
    }

    // calculate F_n for initial y
    Real * const __restrict__ F_n = solver->F_n;
    //Real F_n[INDEX(NSP)];
    dydt (t, pr, y_n, F_n, mech);

    // load initial estimate for eigenvector
    // Real work [INDEX(NSP + 4)];
    Real * const __restrict__ work = solver->work;
    if (work[INDEX(2)] < UROUND) {
        for (int i = 0; i < NSP; ++i) {
            work[INDEX(4 + i)] = F_n[INDEX(i)];
        }
    }

    const Real stepSizeMax = fabs(tEnd - t);
    Real stepSizeMin = TEN * UROUND * fmax(fabs(t), stepSizeMax);

    //Real spec_rad;
    Real * const __restrict__ temp_arr = solver->temp_arr;
    //Real temp_arr[INDEX(NSP)];
    Real * const __restrict__ temp_arr2 = solver->temp_arr2;
    //Real temp_arr2[INDEX(NSP)];

    Real * const __restrict__ y_jm1 = solver->y_jm1;
    Real * const __restrict__ y_jm2 = solver->y_jm2;

    while (t < tEnd) {
        Real err;

        // estimate Jacobian spectral radius
        // only if 25 steps passed
        if ((nstep % 25) == 0) {
            //spec_rad = rkc_spec_rad (t, pr, y_n, F_n, temp_arr, temp_arr2);
            work[INDEX(3)] = rkc_spec_rad (t, pr, stepSizeMax, y_n, F_n, &work[4 * GRID_DIM], temp_arr2, mech);
        }
        //Real spec_rad = rkc_spec_rad (t, pr, y_n, F_n, temp_arr, temp_arr2);

        if (work[INDEX(2)] < UROUND) {
            // estimate first time step
            work[INDEX(2)] = stepSizeMax;
            if ((work[INDEX(3)] * work[INDEX(2)]) > ONE) {
                work[INDEX(2)] = ONE / work[INDEX(3)];
            }
            work[INDEX(2)] = fmax(work[INDEX(2)], stepSizeMin);

            for (int i = 0; i < NSP; ++i) {
                temp_arr[INDEX(i)] = y_n[INDEX(i)] + (work[INDEX(2)] * F_n[INDEX(i)]);
            }
            dydt (t + work[INDEX(2)], pr, temp_arr, temp_arr2, mech);

            err = ZERO;
            for (int i = 0; i < NSP; ++i) {
                Real est = (temp_arr2[INDEX(i)] - F_n[INDEX(i)]) / (ATOL + RTOL * fabs(y_n[INDEX(i)]));
                err += est * est;
            }
            err = work[INDEX(2)] * sqrt(err / NSP);

            if ((P1 * work[INDEX(2)]) < (stepSizeMax * sqrt(err))) {
                work[INDEX(2)] = fmax(P1 * work[INDEX(2)] / sqrt(err), stepSizeMin);
            } else {
                work[INDEX(2)] = stepSizeMax;
            }
        }

        // otherwise use value stored in work[INDEX(2)], calculated in previous step

        // check if last step
        if ((ONEP1 * work[INDEX(2)]) >= fabs(tEnd - t)) {
            work[INDEX(2)] = fabs(tEnd - t);
        }

        // calculate number of steps
        int m = 1 + (int)(sqrt(ONEP54 * work[INDEX(2)] * work[INDEX(3)] + ONE));

        if (m > mMax) {
            m = mMax;
            work[INDEX(2)] = ((Real)(m * m - 1)) / (ONEP54 * work[INDEX(3)]);
        }

        // perform tentative time step
        rkc_step (t, pr, work[INDEX(2)], y_n, F_n, m, y, y_jm1, y_jm2, mech);

        // calculate F_np1 with tenative y_np1
        dydt (t + work[INDEX(2)], pr, y, temp_arr, mech);

        // estimate error
        err = ZERO;
        for (int i = 0; i < NSP; ++i) {
            Real est = P8 * (y_n[INDEX(i)] - y[INDEX(i)]) + P4 * work[INDEX(2)] * (F_n[INDEX(i)] + temp_arr[INDEX(i)]);
            est /= (ATOL + RTOL * fmax(fabs(y[INDEX(i)]), fabs(y_n[INDEX(i)])));
            err += est * est;
        }
        err = sqrt(err / ((Real)NSP));

        if (err > ONE) {
            // error too large, step is rejected

            // select smaller step size
            work[INDEX(2)] = P8 * work[INDEX(2)] / (pow(err, ONE3RD));

            // reevaluate spectral radius
            //spec_rad = rkc_spec_rad (t, pr, y_n, F_n, temp_arr, temp_arr2);
            work[INDEX(3)] = rkc_spec_rad (t, pr, stepSizeMax, y_n, F_n, &work[GRID_DIM * 4], temp_arr2, mech);
        } else {
            // step accepted
            t += work[INDEX(2)];
            nstep++;

            Real fac = TEN;
            Real temp1, temp2;

            if (work[INDEX(1)] < UROUND) {
                temp2 = pow(err, ONE3RD);
                if (P8 < (fac * temp2)) {
                    fac = P8 / temp2;
                }
            } else {
                temp1 = P8 * work[INDEX(2)] * pow(work[INDEX(0)], ONE3RD);
                temp2 = work[INDEX(1)] * pow(err, TWO3RD);
                if (temp1 < (fac * temp2)) {
                    fac = temp1 / temp2;
                }
            }

            // set "old" values to those for current time step
            work[INDEX(0)] = err;
            work[INDEX(1)] = work[INDEX(2)];

            for (int i = 0; i < NSP; ++i) {
                y_n[INDEX(i)] = y[INDEX(i)];
                F_n[INDEX(i)] = temp_arr[INDEX(i)];
            }

            work[INDEX(2)] *= fmax(P1, fac);
            work[INDEX(2)] = fmax(stepSizeMin, fmin(stepSizeMax, work[INDEX(2)]));

            /* for the momentm this isn't supported
            if (task == 0) {
                // only perform one step
                return;
            }*/
        }

    }

    int * const __restrict__ result = solver->result;
    result[T_ID] = EC_success;

} // rkc_driver


#ifdef GENERATE_DOCS
}
#endif