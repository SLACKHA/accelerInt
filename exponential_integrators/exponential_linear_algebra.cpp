/*!
 * \file exponential_linear_algebra.cpp
 * \brief Implementation of various linear algebra functions needed in the exponential integrators
 *
 * \author Nicholas Curtis
 * \date 03/09/2015
 *
 * Modified 09/21 for C++ conversion
 *
 */

#include "exp_solver.hpp"

///////////////////////////////////////////////////////////////////////////////

/*!
 * \brief Matrix-vector multiplication of a matrix sized MxM and a vector Mx1
 * \param[in]       m       size of the matrix
 * \param[in]       A       matrix of size MxM
 * \param[in]       V       vector of size Mx1
 * \param[out]      Av      vector that is A * v
 */
void ExponentialIntegrator::matvec_m_by_m (const int m, const double * __restrict__ A,
                                           const double * __restrict__ V, double * __restrict__ Av) {
    //for each row

    for (int i = 0; i < m; ++i) {
        Av[i] = A[i] * V[0];

        //go across a row of A, multiplying by a column
        for (int j = 1; j < m; ++j) {
            Av[i] += A[j * STRIDE + i] * V[j];
        }
    }
}

///////////////////////////////////////////////////////////////////////////////

/*!
 *
 * \brief Matrix-vector plus equals for a matrix of size MxM and vector of size Mx1.
 *        That is, it returns (A + I) * v
 *
 * \param[in]       m       size of the matrix
 * \param[in]       A       matrix of size MxM
 * \param[in]       V       vector of size Mx1
 * \param[out]      Av      vector that is (A + I) * v
 */
void ExponentialIntegrator::matvec_m_by_m_plusequal (const int m, const double * __restrict__ A,
                                                     const double * __restrict__ V, double * __restrict__ Av)
{
    //for each row

    for (int i = 0; i < m; ++i) {
        Av[i] = A[i] * V[0];

        //go across a row of A, multiplying by a column of phiHm

        for (int j = 1; j < m; ++j) {
            Av[i] += A[j * STRIDE + i] * V[j];
        }

        Av[i] += V[i];
    }
}

/*!
 *
 *  \brief Matrix-vector multiplication of a matrix sized NSPxM and a vector of size Mx1 scaled by a specified factor
 *         That is, it returns A * v * scale
 *
 * \param[in]       m       size of the matrix
 * \param[in]       scale   a number to scale the multplication by
 * \param[in]       A       matrix
 * \param[in]       V       the vector
 * \param[out]      Av      vector that is A * V * scale
 */
void ExponentialIntegrator::matvec_n_by_m_scale (const int m, const double scale,
                                                 const double * __restrict__ A,
                                                 const double * __restrict__ V,
                                                 double * __restrict__ Av) {
    //for each row

    for (int i = 0; i < _neq; ++i) {
        Av[i] = A[i] * V[0];

        //go across a row of A, multiplying by a column of phiHm

        for (int j = 1; j < m; ++j) {
            Av[i] += A[j * _neq + i] * V[j];
        }

        Av[i] *= scale;
    }
}

/*!
 *  \brief Matrix-vector multiplication of a matrix sized NSPxM and a vector of size Mx1 scaled by a specified factor
 *
 *  Computes the following:
 *  \f$Av1 = A * V1 * scale[0]\f$,
 *  \f$Av2 = A * V2 * scale[1]\f$, and
 *  \f$Av3 = A * V3 * scale[2] + V4 + V5\f$
 *
 * \param[in]       m       size of the matrix
 * \param[in]       scale   a list of numbers to scale the multplication by
 * \param[in]       A       matrix
 * \param[in]       V       a list of 5 pointers corresponding to V1, V2, V3, V4, V5
 * \param[out]      Av      a list of 3 pointers corresponding to Av1, Av2, Av3
 */

void ExponentialIntegrator::matvec_n_by_m_scale_special (const int m, const double* __restrict__ scale,
                                                         const double * __restrict__ A, const double** __restrict__ V,
                                                         double** __restrict__ Av) {
    //for each row
    for (int i = 0; i < _neq; ++i) {
        Av[0][i] = A[i] * V[0][0];
        Av[1][i] = A[i] * V[1][0];
        Av[2][i] = A[i] * V[2][0];

        //go across a row of A, multiplying by a column of phiHm

        for (int j = 1; j < m; ++j) {
            Av[0][i] += A[j * _neq + i] * V[0][j];
            Av[1][i] += A[j * _neq + i] * V[1][j];
            Av[2][i] += A[j * _neq + i] * V[2][j];
        }

        Av[0][i] *= scale[0];
        Av[1][i] *= scale[1];
        Av[2][i] = Av[2][i] * scale[2] + V[3][i] + V[4][i];
    }
}

/*!
 *  \brief Matrix-vector multiplication of a matrix sized NSPxM and a vector of size Mx1 scaled by a specified factor
 *
 *  Computes the following:
 *      \f$Av1 = A * V1 * scale[0]\f$
 *  and:
 *      \f$Av2 = A * V2 * scale[1]\f$
 *
 * Performs inline matrix-vector multiplication (with unrolled loops)
 *
 * \param[in]       m       size of the matrix
 * \param[in]       scale   a list of numbers to scale the multplication by
 * \param[in]       A       matrix
 * \param[in]       V       a list of 2 pointers corresponding to V1, V2
 * \param[out]      Av      a list of 2 pointers corresponding to Av1, Av2
 */
void ExponentialIntegrator::matvec_n_by_m_scale_special2 (const int m, const double* __restrict__ scale,
                                                          const double* __restrict__ A, const double** __restrict__ V,
                                                          double** __restrict__ Av) {
    //for each row
    for (int i = 0; i < _neq; ++i) {
        Av[0][i] = A[i] * V[0][0];
        Av[1][i] = A[i] * V[1][0];

        //go across a row of A, multiplying by a column of phiHm

        for (int j = 1; j < m; ++j) {
            Av[0][i] += A[j * _neq + i] * V[0][j];
            Av[1][i] += A[j * _neq + i] * V[1][j];
        }

        Av[0][i] *= scale[0];
        Av[1][i] *= scale[1];
    }
}

///////////////////////////////////////////////////////////////////////////////

/*!
 * \brief Matrix-vector multiplication of a matrix sized NSPxM and a vector of size Mx1 scaled by a specified factor and added to another vector
 *
 * Computes \f$A * V * scale + add\f$
 *
 * \param[in]       m       size of the matrix
 * \param[in]       scale   a number to scale the multplication by
 * \param[in]       add     the vector to add to the result
 * \param[in]       A       matrix
 * \param[in]       V       the vector
 * \param[out]      Av      vector that is A * V * scale + add
 */
void ExponentialIntegrator::matvec_n_by_m_scale_add (const int m, const double scale, const double* __restrict__ A,
                                                     const double* __restrict__ V, double* __restrict__ Av,
                                                     const double* __restrict__ add) {
    //for each row
    for (int i = 0; i < _neq; ++i) {
        Av[i] = A[i] * V[0];

        //go across a row of A, multiplying by a column of phiHm

        for (int j = 1; j < m; ++j) {
            Av[i] += A[j * _neq + i] * V[j];
        }

        Av[i] = Av[i] * scale + add[i];
    }
}

///////////////////////////////////////////////////////////////////////////////

/*!
 *  \brief Matrix-vector multiplication of a matrix sized NSPxM and a vector of size Mx1 scaled by a specified factor and adds and subtracts the specified vectors
 *         note, the addition is twice the specified vector
 *
 *  Computes \f$scale * A * V + 2 * add - sub\f$
 *
 * \param[in]       m       size of the matrix
 * \param[in]       scale   a number to scale the multplication by
 * \param[in]       A       matrix
 * \param[in]       V       the vector
 * \param[out]      Av      vector that is scale * A * V + 2 * add - sub
 * \param[in]       add     the vector to add to the result
 * \param[in]       sub     the vector to subtract from the result
 */
void ExponentialIntegrator::matvec_n_by_m_scale_add_subtract (const int m, const double scale,
                                                              const double* __restrict__ A, const double* __restrict__ V,
                                                              double* __restrict__ Av, const double* __restrict__ add,
                                                              const double* __restrict__ sub) {
    //for each row

    for (int i = 0; i < _neq; ++i) {
        Av[i] = A[i] * V[0];

        //go across a row of A, multiplying by a column of phiHm

        for (int j = 1; j < m; ++j) {
            Av[i] += A[j * _neq + i] * V[j];
        }

        Av[i] = Av[i] * scale + 2.0 * add[i] - sub[i];
    }
}

///////////////////////////////////////////////////////////////////////////////

/*!
 *  \brief Get scaling for weighted norm
 *
 *  Computes \f$\frac{1.0}{ATOL + \max\left(\left|y0\right|, \left|y1\right|) * RTOL\right)}\f$
 *
 * \param[in]       y0      values at current timestep
 * \param[in]       y1      values at next timestep
 * \param[out]      sc  array of scaling values
 */
void ExponentialIntegrator::scale(const double* __restrict__ y0, const double* __restrict__ y1, double* __restrict__ sc) {

    for (int i = 0; i < _neq; ++i) {
        sc[i] = 1.0 / (ATOL + fmax(fabs(y0[i]), fabs(y1[i])) * RTOL);
    }
}

///////////////////////////////////////////////////////////////////////////////

/*!
 * \brief Get scaling for weighted norm for the initial timestep (used in krylov process)
 *
 * \param[in]       y0      values at current timestep
 * \param[out]  sc  array of scaling values
 */
void ExponentialIntegrator::scale_init (const double* __restrict__ y0, double* __restrict__ sc) {

    for (int i = 0; i < _neq; ++i) {
        sc[i] = 1.0 / (ATOL + fabs(y0[i]) * RTOL);
    }
}

///////////////////////////////////////////////////////////////////////////////

/*!
 *  \brief Perform weighted norm
 *
 *  Computes \f$\left| nums * sc\right|_2\f$
 *
 * \param[in]       nums    values to be normed
 * \param[in]       sc      scaling array for norm
 * \return          norm    weighted norm
 */
double ExponentialIntegrator::sc_norm (const double* __restrict__ nums, const double* __restrict__ sc) {
    double norm = 0.0;

    for (int i = 0; i < _neq; ++i) {
        norm += nums[i] * nums[i] * sc[i] * sc[i];
    }

    return sqrt(norm / _neq);
}

/*!
 * \brief Computes and returns the two norm of a vector
 *
 *  Computes \f$\sqrt{\sum{v^2}}\f$
 *
 *  \param[in]      v       the vector
 */
double ExponentialIntegrator::two_norm(const double* __restrict__ v)
{
    double norm = 0.0;

    for (int i = 0; i < _neq; ++i) {
        norm += v[i] * v[i];
    }
    return sqrt(norm);
}

/*!
 *  \brief Normalize the input vector using a 2-norm
 *
 *  \f$v_{out} = \frac{v}{\left| v \right|}_2\f$
 *
 * \param[in]       v       vector to be normalized
 * \param[out]      v_out   where to stick the normalized part of v (in a column)
 */
double ExponentialIntegrator::normalize (const double* __restrict__ v, double* __restrict__ v_out) {

    double norm = two_norm(v);

    if (norm == 0)
        norm = 1;

    double m_norm = 1.0 / norm;


    for (int i = 0; i < _neq; ++i) {
        v_out[i] = v[i] * m_norm;
    }
    return norm;
}

/*!
 *  \brief Performs the dot product of the w (_neq x 1) vector with the given subspace vector (_neq x 1)
 *
 *  returns \f$Vm \dot w\f$
 *
 * \param[in]       w       the vector with with to dot
 * \param[in]       Vm      the subspace vector
 * \returns         sum - the dot product of the specified vectors
 */
double ExponentialIntegrator::dotproduct(const double* __restrict__ w, const double* __restrict__ Vm)
{
    double sum = 0;

    for(int i = 0; i < _neq; i++)
    {
        sum += w[i] * Vm[i];
    }
    return sum;
}

/*!
 * \brief Subtracts Vm scaled by s from w
 *
 *  \f$ w -= Vm * s\f$
 *
 * \param[in]       s       the scale multiplier to use
 * \param[in]       Vm      the subspace matrix
 * \param[out]      w       the vector to subtract from
 */
void ExponentialIntegrator::scale_subtract(const double s, const double* __restrict__ Vm, double* __restrict__ w)
{

    for (int i = 0; i < _neq; i++)
    {
        w[i] -= s * Vm[i];
    }
}


/*!
 *  \brief Sets Vm to s * w
 *
 *  \f$Vm = s * w\f$
 *
 * \param[in]       s       the scale multiplier to use
 * \param[in]       w       the vector to use as a base
 * \param[out]      Vm      the subspace matrix to set
 */
void ExponentialIntegrator::scale_mult(const double s, const double* __restrict__ w, double* __restrict__ Vm)
{

    for (int i = 0; i < _neq; i++)
    {
        Vm[i] = w[i] * s;
    }
}
