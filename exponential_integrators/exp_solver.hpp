/**
 * \file
 * \brief Definition of the RKC CPU solver
 * \author Nicholas Curtis, Kyle Niemeyer
 * \date 09/19/2019
 */

#ifndef EXP_SOLVER_HPP
#define EXP_SOLVER_HPP

#include <complex>
#include <vector>

#include "solver.hpp"
#include "rational_approximant.hpp"

#define HAVE_SPARSE_MULTIPLIER
#ifdef HAVE_SPARSE_MULTIPLIER
extern "C"
{
    void sparse_multiplier(const double * A, const double * Vm, double* w);
}
#else
#warning ("This isn't correct.")
#define sparse_multiplier(A, v, o) (matvec_m_by_m(_neq, (A), (v), (o)))
#endif

namespace c_solvers
{

    class EXPSolverOptions : public SolverOptions
    {
    public:
        EXPSolverOptions(double atol=1e-10, double rtol=1e-6, bool logging=false,
                         int N_RA=10, int M_MAX=-1):
                SolverOptions(atol, rtol, logging),
                N_RA(N_RA),
                M_MAX(M_MAX)
            {

            }

        //! The number of rational approximants to utilize for the exponential approximation [default 10]
        inline int num_rational_approximants() const
        {
            return N_RA;
        }

        //! the allowed maximum krylov subspace size [defaults to #_neq]
        inline int krylov_subspace_size(int neq) const
        {
            if (M_MAX < 0)
            {
                return neq;
            }
            return M_MAX;
        }

    protected:
        //! The number of rational approximants to utilize for the exponential approximation [default 10]
        const int N_RA;
        //! the maximum krylov subspace size
        const int M_MAX;
    };

    class ExponentialIntegrator : public Integrator
    {

    private:
        //! The required memory size of this integrator in bytes.
        //! This is cummulative with any base classes
        std::size_t _ourMemSize;

    protected:
        /*
         * \brief a shared base-class for the exponential integrators
         */
        ExponentialIntegrator(int neq, int numThreads, int order,
                              const EXPSolverOptions& options) :
            Integrator(neq, numThreads, options),
            poles(options.num_rational_approximants(), 0),
            res(options.num_rational_approximants(), 0),
            STRIDE(options.krylov_subspace_size(neq) + order)
        {
            _ourMemSize = this->setOffsets();
            find_poles_and_residuals(N_RA(), poles, res);
        }

        virtual std::size_t setOffsets()
        {
            std::size_t working = Integrator::requiredSolverMemorySize();
            // invA from hessenberg inversions
            _invA = working;
            working += STRIDE * STRIDE * sizeof(std::complex<double>);
            // ipiv
            _ipiv = working;
            working += STRIDE * sizeof(int);
            // w
            _w = working;
            working += _neq * sizeof(double);
            return working;
        }

        /*
         * \brief Return the required memory size (per-thread) in bytes
         */
        virtual std::size_t requiredSolverMemorySize()
        {
            return _ourMemSize;
        }

        //! log format t, h, err, m, m1, m2
        std::vector<std::tuple<double, double, double, int, int, int>> subspaceLog;

        //! poles for matrix exponentiation via CF
        std::vector<std::complex<double>> poles;
        //! residuals for matrix exponentiation via CF
        std::vector<std::complex<double>> res;

        //! Krylov matrix stride
        const int STRIDE;
        //! \brief Return the number of rational approximants to utilize for the exponential approximation
        const int N_RA()
        {
            return static_cast<const EXPSolverOptions&>(_options).num_rational_approximants();
        }

        //! offsets

        //! invA
        std::size_t _invA;
        //! ipiv
        std::size_t _ipiv;
        //! working vector for arnoldi & getComplexInverseHessenbergLU
        std::size_t _w;

        /** \brief Compute the correct order Phi (exponential) matrix function.
         *         This is dependent on the exponential solver type, and hence must be
         *         overridden in the subclasses.
         *
         *  \param[in]      m       The Hessenberg matrix size (mxm)
         *  \param[in]      A       The input Hessenberg matrix
         *  \param[in]      c       The scaling factor
         *  \param[out]     phiA    The resulting exponential matrix
         */
        virtual int exponential(const int m, const double* A, const double c, double* phiA) = 0;



        /*!
         * \brief Matrix-vector multiplication of a matrix sized MxM and a vector Mx1
         * \param[in]       m       size of the matrix
         * \param[in]       A       matrix of size MxM
         * \param[in]       V       vector of size Mx1
         * \param[out]      Av      vector that is A * v
         */
        void matvec_m_by_m (const int m, const double * __restrict__ A,
                            const double * __restrict__ V, double * __restrict__ Av);
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
        void matvec_m_by_m_plusequal (const int m, const double * __restrict__ A,
                                      const double * __restrict__ V, double * __restrict__ Av);

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
        void matvec_n_by_m_scale (const int m, const double scale,
                                  const double * __restrict__ A, const double * __restrict__ V,
                                  double * __restrict__ Av);

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

        void matvec_n_by_m_scale_special (const int m, const double* __restrict__ scale,
                                          const double * __restrict__ A, const double** __restrict__ V,
                                          double** __restrict__ Av);

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
        void matvec_n_by_m_scale_special2 (const int m, const double* __restrict__ scale,
                                           const double* __restrict__ A, const double** __restrict__ V,
                                           double** __restrict__ Av);
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
        void matvec_n_by_m_scale_add (const int m, const double scale, const double* __restrict__ A,
                                      const double* __restrict__ V, double* __restrict__ Av,
                                      const double* __restrict__ add);

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
        void matvec_n_by_m_scale_add_subtract (const int m, const double scale,
                                               const double* __restrict__ A, const double* __restrict__ V,
                                               double* __restrict__ Av, const double* __restrict__ add,
                                               const double* __restrict__ sub);

        /*!
         *  \brief Get scaling for weighted norm
         *
         *  Computes \f$\frac{1.0}{ATOL + \max\left(\left|y0\right|, \left|y1\right|) * RTOL\right)}\f$
         *
         * \param[in]       y0      values at current timestep
         * \param[in]       y1      values at next timestep
         * \param[out]      sc  array of scaling values
         */
        void scale (const double* __restrict__ y0, const double* __restrict__ y1, double* __restrict__ sc);

        /*!
         * \brief Get scaling for weighted norm for the initial timestep (used in krylov process)
         *
         * \param[in]       y0      values at current timestep
         * \param[out]      sc      array of scaling values
         */
        void scale_init (const double* __restrict__ y0, double* __restrict__ sc);

        /*!
         *  \brief Perform weighted norm
         *
         *  Computes \f$\left| nums * sc\right|_2\f$
         *
         * \param[in]       nums    values to be normed
         * \param[in]       sc      scaling array for norm
         * \return          norm    weighted norm
         */
        double sc_norm (const double* __restrict__ nums, const double* __restrict__ sc);

        /*!
         * \brief Computes and returns the two norm of a vector
         *
         *  Computes \f$\sqrt{\sum{v^2}}\f$
         *
         *  \param[in]      v       the vector
         */
        double two_norm(const double* __restrict__ v);

        /*!
         *  \brief Normalize the input vector using a 2-norm
         *
         *  \f$v_{out} = \frac{v}{\left| v \right|}_2\f$
         *
         * \param[in]       v       vector to be normalized
         * \param[out]      v_out   where to stick the normalized part of v (in a column)
         */
        double normalize (const double* __restrict__ v, double* __restrict__ v_out);

        /*!
         *  \brief Performs the dot product of the w (NSP x 1) vector with the given subspace vector (NSP x 1)
         *
         *  returns \f$Vm \dot w\f$
         *
         * \param[in]       w       the vector with with to dot
         * \param[in]       Vm      the subspace vector
         * \returns         sum - the dot product of the specified vectors
         */
        double dotproduct(const double* __restrict__ w, const double* __restrict__ Vm);

        /*!
         * \brief Subtracts Vm scaled by s from w
         *
         *  \f$ w -= Vm * s\f$
         *
         * \param[in]       s       the scale multiplier to use
         * \param[in]       Vm      the subspace matrix
         * \param[out]      w       the vector to subtract from
         */
        void scale_subtract(const double s, const double* __restrict__ Vm, double* __restrict__ w);

        /*!
         *  \brief Sets Vm to s * w
         *
         *  \f$Vm = s * w\f$
         *
         * \param[in]       s       the scale multiplier to use
         * \param[in]       w       the vector to use as a base
         * \param[out]      Vm      the subspace matrix to set
         */
        void scale_mult(const double s, const double* __restrict__ w, double* __restrict__ Vm);

        /*
         * \fn int arnoldi(const double scale, const int p, const double h, const double* A, const double* v,
         *                 const double* sc, double* beta, double* Vm, double* Hm, double* phiHm)
         * \brief Runs the arnoldi iteration to calculate the Krylov projection
         * \returns             m       the ending size of the matrix
         * \param[in]           scale   the value to scale the timestep by
         * \param[in]           p       the order of the maximum phi function needed
         * \param[in]           h       the timestep
         * \param[in]           A       the jacobian matrix
         * \param[in]           v       the vector to use for the krylov subspace
         * \param[in]           sc      the error scaling vector
         * \param[out]          beta    the norm of the v vector
         * \param[out]          Vm      the arnoldi basis matrix
         * \param[out]          Hm      the constructed Hessenberg matrix, used in actual exponentials
         * \param[out]          phiHm   the exponential matrix computed from h * scale * Hm
         */
        int arnoldi(const double scale, const int p, const double h, const double* __restrict__ A,
                    const double* __restrict__ v, const double* __restrict__ sc,
                    double* __restrict__ beta, double* __restrict__ Vm, double* __restrict__ Hm,
                    double* __restrict__ phiHm);

        /** \brief Compute the 2nd order Phi (exponential) matrix function
         *
         *  Computes \f$\phi_2(c*A)\f$
         *
         *  \param[in]      m       The Hessenberg matrix size (mxm)
         *  \param[in]      A       The input Hessenberg matrix
         *  \param[in]      c       The scaling factor
         *  \param[out]     phiA    The resulting exponential matrix
         */
        int phi2Ac_variable(const int m, const double* A, const double c, double* phiA);

        /** \brief Compute the first order Phi (exponential) matrix function
         *
         *  Computes \f$\phi_1(c*A)\f$
         *
         *  \param[in]      m       The Hessenberg matrix size (mxm)
         *  \param[in]      A       The input Hessenberg matrix
         *  \param[in]      c       The scaling factor
         *  \param[out]     phiA    The resulting exponential matrix
         */
        int phiAc_variable(const int m, const double* A, const double c, double* phiA);

        /** \brief Compute the zeroth order Phi (exponential) matrix function.
         *         This is the regular matrix exponential
         *
         *  Computes \f$\phi_0(c*A)\f$
         *
         *  \param[in]      m       The Hessenberg matrix size (mxm)
         *  \param[in]      A       The input Hessenberg matrix
         *  \param[in]      c       The scaling factor
         *  \param[out]     phiA    The resulting exponential matrix
         */
        int expAc_variable(const int m, const double* A, const double c, double* phiA);

    };
}

#endif
