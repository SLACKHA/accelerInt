#include "header.h"

#include "rkc.cuh"
#include "dydt.cuh"

//#define SHARED

const Real rel_tol = 1.0e-6;
const Real abs_tol = 1.0e-10;

/////////////////////////////////////////////////////////

__device__
Real rkc_spec_rad (const Real t, const Real pr, const Real hmax, const Real* y,
                   const Real* F, Real* v, Real* Fv) {
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

    //for (int i = 0; i < NN; ++i) {
    //    v[i] = F[i];
    //}

    Real nrm1 = ZERO;
    Real nrm2 = ZERO;
    for (int i = 0; i < NN; ++i) {
        nrm1 += (y[i] * y[i]);
        nrm2 += (v[i] * v[i]);
    }
    nrm1 = sqrt(nrm1);
    nrm2 = sqrt(nrm2);

    Real dynrm;
    if ((nrm1 != ZERO) && (nrm2 != ZERO)) {
        dynrm = nrm1 * sqrt(UROUND);
        for (int i = 0; i < NN; ++i) {
            v[i] = y[i] + v[i] * (dynrm / nrm2);
        }
    } else if (nrm1 != ZERO) {
        dynrm = nrm1 * sqrt(UROUND);
        for (int i = 0; i < NN; ++i) {
            v[i] = y[i] * (ONE + sqrt(UROUND));
        }
    } else if (nrm2 != ZERO) {
        dynrm = UROUND;
        for (int i = 0; i < NN; ++i) {
            v[i] *= (dynrm / nrm2);
        }
    } else {
        dynrm = UROUND;
        for (int i = 0; i < NN; ++i) {
            v[i] = UROUND;
        }
    }

    // now iterate using nonlinear power method
    Real sigma = ZERO;
    for (int iter = 1; iter <= itmax; ++iter) {

        dydt (t, pr, v, Fv);

        nrm1 = ZERO;
        for (int i = 0; i < NN; ++i) {
            nrm1 += ((Fv[i] - F[i]) * (Fv[i] - F[i]));
        }
        nrm1 = sqrt(nrm1);
        nrm2 = sigma;
        sigma = nrm1 / dynrm;

        nrm2 = fabs(sigma - nrm2) / sigma;
        if ((iter >= 2) && (fabs(sigma - nrm2) <= (fmax(sigma, small) * P01))) {
            for (int i = 0; i < NN; ++i) {
                v[i] -= y[i];
            }
            return (ONEP2 * sigma);
        }

        if (nrm1 != ZERO) {
            for (int i = 0; i < NN; ++i) {
                v[i] = y[i] + ((Fv[i] - F[i]) * (dynrm / nrm1));
            }
        } else {
            int ind = (iter % NN);
            v[ind] = y[ind] - (v[ind] - y[ind]);
        }
    }
    return (ONEP2 * sigma);
}

///////////////////////////////////////////////////////

__device__
void rkc_step (const Real t, const Real pr, const Real h, const Real* y_0,
               const Real* F_0, const int s, Real* y_j) {
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

    Real y_jm1[NN];
    Real y_jm2[NN];

      // calculate y_1
    Real mu_t = w1 * b_jm1;
    for (int i = 0; i < NN; ++i) {
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

        for (int i = 0; i < NN; ++i) {
            y_j[i] = (ONE - mu - nu) * y_0[i] + (mu * y_jm1[i]) + (nu * y_jm2[i])
                 + h * mu_t * (y_j[i] - (gamma_t * F_0[i]));
        }
        Real c_j = (mu * c_jm1) + (nu * c_jm2) + mu_t * (ONE - gamma_t);

        if (j < s) {
            for (int i = 0; i < NN; ++i) {
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

__device__
void rkc_driver (Real t, const Real tEnd, const Real pr, int task, Real* work, Real* y) {
   /**
    * Driver function for RKC integrator.
    *
    * @param t    the starting time.
    * @param tEnd the desired end time.
    * @param pr   A parameter used for pressure or density to pass to the derivative function.
    * @param task 0 to take a single integration step, 1 to integrate to tEnd.
    * @param work Real work array, size 3.
    * @param y    Dependent variable array, integrated values replace initial conditions.
    */

    int nstep = 0;
    int mMax = (int)(round(sqrt(rel_tol / (10.0 * UROUND))));

    if (mMax < 2) {
        mMax = 2;
    }

    Real y_n[NN];
    for (int i = 0; i < NN; ++i) {
        y_n[i] = y[i];
    }

    // calculate F_n for initial y
    Real F_n[NN];
    dydt (t, pr, y_n, F_n);

    // load initial estimate for eigenvector
    if (work[2] < UROUND) {
        for (int i = 0; i < NN; ++i) {
            work[4 + i] = F_n[i];
        }
    }

    const Real stepSizeMax = fabs(tEnd - t);
    Real stepSizeMin = TEN * UROUND * fmax(fabs(t), stepSizeMax);

    //Real spec_rad;

    while (t < tEnd) {

        Real temp_arr[NN];
        Real temp_arr2[NN];
        Real err;

        // estimate Jacobian spectral radius
        // only if 25 steps passed
        if ((nstep % 25) == 0) {
            //spec_rad = rkc_spec_rad (t, pr, y_n, F_n, temp_arr, temp_arr2);
            work[3] = rkc_spec_rad (t, pr, stepSizeMax, y_n, F_n, &work[4], temp_arr2);
        }
        //Real spec_rad = rkc_spec_rad (t, pr, y_n, F_n, temp_arr, temp_arr2);

        if (work[2] < UROUND) {
            // estimate first time step
            work[2] = stepSizeMax;
            if ((work[3] * work[2]) > ONE) {
                work[2] = ONE / work[3];
            }
            work[2] = fmax(work[2], stepSizeMin);

            for (int i = 0; i < NN; ++i) {
                temp_arr[i] = y_n[i] + (work[2] * F_n[i]);
            }
            dydt (t + work[2], pr, temp_arr, temp_arr2);

            err = ZERO;
            for (int i = 0; i < NN; ++i) {
                Real est = (temp_arr2[i] - F_n[i]) / (abs_tol + rel_tol * fabs(y_n[i]));
                err += est * est;
            }
            err = work[2] * sqrt(err / NN);

            if ((P1 * work[2]) < (stepSizeMax * sqrt(err))) {
                work[2] = fmax(P1 * work[2] / sqrt(err), stepSizeMin);
            } else {
                work[2] = stepSizeMax;
            }
        }

        // otherwise use value stored in work[2], calculated in previous step

        // check if last step
        if ((ONEP1 * work[2]) >= fabs(tEnd - t)) {
            work[2] = fabs(tEnd - t);
        }

        // calculate number of steps
        int m = 1 + (int)(sqrt(ONEP54 * work[2] * work[3] + ONE));

        if (m > mMax) {
            m = mMax;
            work[2] = ((Real)(m * m - 1)) / (ONEP54 * work[3]);
        }

        // perform tentative time step
        rkc_step (t, pr, work[2], y_n, F_n, m, y);

        // calculate F_np1 with tenative y_np1
        dydt (t + work[2], pr, y, temp_arr);

        // estimate error
        err = ZERO;
        for (int i = 0; i < NN; ++i) {
            Real est = P8 * (y_n[i] - y[i]) + P4 * work[2] * (F_n[i] + temp_arr[i]);
            est /= (abs_tol + rel_tol * fmax(fabs(y[i]), fabs(y_n[i])));
            err += est * est;
        }
        err = sqrt(err / ((Real)NN));

        if (err > ONE) {
            // error too large, step is rejected

            // select smaller step size
            work[2] = P8 * work[2] / (pow(err, ONE3RD));

            // reevaluate spectral radius
            //spec_rad = rkc_spec_rad (t, pr, y_n, F_n, temp_arr, temp_arr2);
            work[3] = rkc_spec_rad (t, pr, stepSizeMax, y_n, F_n, &work[4], temp_arr2);
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

            for (int i = 0; i < NN; ++i) {
                y_n[i] = y[i];
                F_n[i] = temp_arr[i];
            }

            work[2] *= fmax(P1, fac);
            work[2] = fmax(stepSizeMin, fmin(stepSizeMax, work[2]));

            if (task == 0) {
                // only perform one step
                return;
            }
        }

    }

} // rkc_driver
