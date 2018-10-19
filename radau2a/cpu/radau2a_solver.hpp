/**
 * \file
 * \brief Definition of the RadauIIA CPU solver
 * \author Nicholas Curtis
 * \date 09/19/2019
 */

#ifndef RADAU2A_SOLVER_H
#define RADAU2A_SOLVER_H

#include "solver.hpp"
#include <complex>

namespace c_solvers
{
    //! Maximum number of allowed internal timesteps before error
    #define Max_no_steps (200000)
    //! Maximum number of allowed Newton iteration steps before error
    #define NewtonMaxit (8)
    //! Use quadratic interpolation from previous step if possible
    #define StartNewton (true)
    //! Use gustafsson time stepping control
    #define Gustafsson
    //! Smallist representable double precision number
    #define Roundoff (EPS)
    //! Controls maximum decrease in timestep size
    #define FacMin (0.2)
    //! Controls maximum increase in timestep size
    #define FacMax (8)
    //! Safety factor for Gustafsson time stepping control
    #define FacSafe (0.9)
    //! Time step factor on rejected step
    #define FacRej (0.1)
    //! Minimum Newton convergence rate
    #define ThetaMin (0.001)
    //! Newton convergence tolerance
    #define NewtonTol (0.03)
    //! Min Timestep ratio to skip LU decomposition
    #define Qmin (1.0)
    //! Max Timestep ratio to skip LU decomposition
    #define Qmax (1.2)
    //#define UNROLL (8)
    //! Error allowed on this many consecutive internal timesteps before exit
    #define Max_consecutive_errs (5)
    //#define SDIRK_ERROR

    static constexpr double rkA[3][3] = { {
         1.968154772236604258683861429918299e-1,
        -6.55354258501983881085227825696087e-2,
         2.377097434822015242040823210718965e-2
        }, {
         3.944243147390872769974116714584975e-1,
         2.920734116652284630205027458970589e-1,
        -4.154875212599793019818600988496743e-2
        }, {
         3.764030627004672750500754423692808e-1,
         5.124858261884216138388134465196080e-1,
         1.111111111111111111111111111111111e-1
        }
    };

    static constexpr double rkB[3] = {
    3.764030627004672750500754423692808e-1,
    5.124858261884216138388134465196080e-1,
    1.111111111111111111111111111111111e-1
    };

    static constexpr double rkC[3] = {
    1.550510257216821901802715925294109e-1,
    6.449489742783178098197284074705891e-1,
    1.0
    };

    #ifdef SDIRK_ERROR
        // Classical error estimator:
        // H* Sum (B_j-Bhat_j)*f(Z_j) = H*E(0)*f(0) + Sum E_j*Z_j
        static constexpr double rkE[4] = {
        0.02,
        -10.04880939982741556246032950764708*0.02,
        1.382142733160748895793662840980412*0.02,
        -0.3333333333333333333333333333333333*0.02
        };
        // H* Sum Bgam_j*f(Z_j) = H*Bgam(0)*f(0) + Sum Theta_j*Z_j
        static constexpr double rkTheta[3] = {
        -1.520677486405081647234271944611547 - 10.04880939982741556246032950764708*0.02,
        2.070455145596436382729929151810376 + 1.382142733160748895793662840980413*0.02,
        -0.3333333333333333333333333333333333*0.02 - 0.3744441479783868387391430179970741
        };
        // ! Sdirk error estimator
        static constexpr double rkBgam[5] = {
        0.02,
        0.3764030627004672750500754423692807-1.558078204724922382431975370686279*0.02,
        0.8914115380582557157653087040196118*0.02+0.5124858261884216138388134465196077,
        -0.1637777184845662566367174924883037-0.3333333333333333333333333333333333*0.02,
        0.2748888295956773677478286035994148
        };
    #else
        // Classical error estimator:
        // H* Sum (B_j-Bhat_j)*f(Z_j) = H*E(0)*f(0) + Sum E_j*Z_j
        static constexpr double rkE[4] = {
        0.05,
        -10.04880939982741556246032950764708*0.05,
        1.382142733160748895793662840980412*0.05,
        -0.3333333333333333333333333333333333*0.05
        };
        // H* Sum Bgam_j*f(Z_j) = H*Bgam(0)*f(0) + Sum Theta_j*Z_j
        static constexpr double rkTheta[3] = {
        -1.520677486405081647234271944611547 - 10.04880939982741556246032950764708*0.05,
        2.070455145596436382729929151810376 + 1.382142733160748895793662840980413*0.05,
        -0.3333333333333333333333333333333333*0.05 - 0.3744441479783868387391430179970741
        };
    #endif

    //Local order of error estimator
    /*
    !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    !~~~> Diagonalize the RK matrix:
    ! rkTinv * inv(rkA) * rkT =
    !           |  rkGamma      0           0     |
    !           |      0      rkAlpha   -rkBeta   |
    !           |      0      rkBeta     rkAlpha  |
    !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    */

    static constexpr double rkGamma = 3.637834252744495732208418513577775;
    static constexpr double rkAlpha = 2.681082873627752133895790743211112;
    static constexpr double rkBeta  = 3.050430199247410569426377624787569;

    static constexpr double rkT[3][3] = {
    {9.443876248897524148749007950641664e-2,
    -1.412552950209542084279903838077973e-1,
    -3.00291941051474244918611170890539e-2},
    {2.502131229653333113765090675125018e-1,
    2.041293522937999319959908102983381e-1,
    3.829421127572619377954382335998733e-1},
    {1.0,
    1.0,
    0.0e0}
    };

    static constexpr double rkTinv[3][3] =
    {{4.178718591551904727346462658512057,
    3.27682820761062387082533272429617e-1,
    5.233764454994495480399309159089876e-1},
    {-4.178718591551904727346462658512057,
    -3.27682820761062387082533272429617e-1,
    4.766235545005504519600690840910124e-1},
    {-5.02872634945786875951247343139544e-1,
    2.571926949855605429186785353601676e0,
    -5.960392048282249249688219110993024e-1}
    };

    static constexpr double rkTinvAinv[3][3] = {
    {1.520148562492775501049204957366528e+1,
    1.192055789400527921212348994770778,
    1.903956760517560343018332287285119},
    {-9.669512977505946748632625374449567,
    -8.724028436822336183071773193986487,
    3.096043239482439656981667712714881},
    {-1.409513259499574544876303981551774e+1,
    5.895975725255405108079130152868952,
    -1.441236197545344702389881889085515e-1}
    };

    static constexpr double rkAinvT[3][3] = {
    {0.3435525649691961614912493915818282,
    -0.4703191128473198422370558694426832,
    0.3503786597113668965366406634269080},
    {0.9102338692094599309122768354288852,
    1.715425895757991796035292755937326,
    0.4040171993145015239277111187301784},
    {3.637834252744495732208418513577775,
    2.681082873627752133895790743211112,
    -3.050430199247410569426377624787569}
    };

    static constexpr double rkELO = 4;
    //! Lapack - non-transpose
    static constexpr char TRANS = 'N';
    //! Lapack - 1 RHS solve
    static constexpr int NRHS = 1;

    class RadauIntegrator : public Integrator
    {

    public:

        RadauIntegrator(int neq, int numThreads, const IVP& ivp, const SolverOptions& options) :
            Integrator(neq, numThreads, ivp, options),
            ARRSIZE(neq),
            STRIDE(neq)
        {
            _ourMemSize = this->setOffsets();
            this->reinitialize(numThreads);
        }

        /*!
           \fn char* solverName()
           \brief Returns a descriptive solver name
        */
        const char* solverName() const
        {
            const char* name = "radau2a-int";
            return name;
        }

        void initSolverLog()
        {
            // pass
        }

        void solverLog()
        {
            // pass
        }

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
        ErrorCode integrate (
            const double t_start, const double t_end, const double pr, double* y);

    protected:

        //! offsets
        std::size_t _sc;
        std::size_t _A;
        std::size_t _E1;
        std::size_t _E2;
        std::size_t _ipiv1;
        std::size_t _ipiv2;
        std::size_t _Z1;
        std::size_t _Z2;
        std::size_t _Z3;
        #ifdef SDIRK_ERROR
        std::size_t _Z4;
        std::size_t _DZ4;
        std::size_t _G;
        std::size_t _TMP_SDIRK;
        #endif
        std::size_t _DZ1;
        std::size_t _DZ2;
        std::size_t _DZ3;
        std::size_t _CONT;
        std::size_t _y0;
        std::size_t _F0;
        std::size_t _F1;
        std::size_t _F2;
        std::size_t _TMP;

    private:
        //! the total amount of memory (in bytes) required for this solver
        std::size_t _ourMemSize;

    protected:

        virtual std::size_t setOffsets()
        {
            std::size_t working = Integrator::requiredSolverMemorySize();
            // sc
            _sc = working;
            working += _neq * sizeof(double);
            // A
            _A = working;
            working += _neq * _neq * sizeof(double);
            // E1
            _E1 = working;
            working += _neq * _neq * sizeof(double);
            // E2
            _E2 = working;
            working += _neq * _neq * sizeof(std::complex<double>);
            // ipiv1
            _ipiv1 = working;
            working += _neq * sizeof(int);
            // ipiv2
            _ipiv2 = working;
            working += _neq * sizeof(int);
            // Z1
            _Z1 = working;
            working += _neq * sizeof(double);
            // Z2
            _Z2 = working;
            working += _neq * sizeof(double);
            // Z3
            _Z3 = working;
            working += _neq * sizeof(double);
            #ifdef SDIRK_ERROR
            // Z4
            _Z4 = working;
            working += _neq * sizeof(double);
            // DZ4
            _DZ4 = working;
            working += _neq * sizeof(double);
            // G
            _G = working;
            working += _neq * sizeof(double);
            // TMP
            _TMP_SDIRK = working;
            working += _neq * sizeof(double);
            #endif
            // DZ1
            _DZ1 = working;
            working += _neq * sizeof(double);
            // DZ2
            _DZ2 = working;
            working += _neq * sizeof(double);
            // DZ3
            _DZ3 = working;
            working += _neq * sizeof(double);
            // CONT
            _CONT = working;
            working += 3 * _neq * sizeof(double);
            // y0
            _y0 = working;
            working += _neq * sizeof(double);
            // F0
            _F0 = working;
            working += _neq * sizeof(double);
            // F1
            _F1 = working;
            working += _neq * sizeof(double);
            // F2
            _F2 = working;
            working += _neq * sizeof(double);
            // TMP
            _TMP = working;
            working += _neq * sizeof(double);
            // and return required work size
            return working;
        }

        /*
         * \brief Return the required memory size (per-thread) in bytes
         */
        virtual std::size_t requiredSolverMemorySize()
        {
            return _ourMemSize;
        }


        //! Lapack - Array size
        const int ARRSIZE;
        //! the matrix dimensions
        const int STRIDE;

        /**
         * \brief Computes error weight scaling from initial and current state
         * \param[in]       y0          the initial state vector to use
         * \param[in]       y           the current state vector
         * \param[out]      sc          the populated error weight scalings
         */
        inline void scale (const double * __restrict__ y0,
                           const double* __restrict__ y,
                           double * __restrict__ sc);

        /**
         * \brief Computes error weight scaling from initial state
         * \param[in]       y0          the initial state vector to use
         * \param[out]      sc          the populated error weight scalings
         */
        inline void scale_init (const double * __restrict__ y0,
                                double * __restrict__ sc);

        /**
         * \brief Compute E1 & E2 matricies and their LU Decomposition
         * \param[in]            H               The timestep size
         * \param[in,out]        E1              The non-complex matrix system
         * \param[in,out]        E2              The complex matrix system
         * \param[in]            Jac             The Jacobian matrix
         * \param[out]           ipiv1           The pivot indicies for E1
         * \param[out]           ipiv2           The pivot indicies for E2
         * \param[out]           info            An indicator variable determining if an error occured.
         */
        inline void RK_Decomp(const double H, double* __restrict__ E1,
                              std::complex<double>* __restrict__ E2, const double* __restrict__ Jac,
                              int* __restrict__ ipiv1, int* __restrict__ ipiv2, int* __restrict__ info);

        /**
         * \brief Compute Quadaratic interpolate
         */
        inline void RK_Make_Interpolate(const double* __restrict__ Z1, const double* __restrict__ Z2,
                                        const double* __restrict__ Z3, double* __restrict__ CONT);

        /**
         * \brief Apply quadaratic interpolate to get initial values
         */
        inline void RK_Interpolate(const double H, const double Hold, double* __restrict__ Z1,
                                   double* __restrict__ Z2, double* __restrict__ Z3, const double* __restrict__ CONT);

        /**
         * \brief Performs \f$Z:= X + Y\f$ with unrolled (or at least bounds known at compile time) loops
         */
        inline void WADD(const double* __restrict__ X, const double* __restrict__ Y,
                         double* __restrict__ Z);

        /**
         * \brief Specialization of DAXPY with unrolled (or at least bounds known at compile time) loops
         *
         *
         * Performs:
         *    *  \f$DY1:= DA1 * DX\f$
         *    *  \f$DY2:= DA2 * DX\f$
         *    *  \f$DY3:= DA3 * DX\f$
         */
        inline void DAXPY3(const double DA1, const double DA2, const double DA3,
                                  const double* __restrict__ DX, double* __restrict__ DY1,
                                  double* __restrict__ DY2, double* __restrict__ DY3);

        /**
         *  \brief Prepare the right-hand side for Newton iterations:
         *     \f$R = Z - hA * F\f$
         */
        inline void RK_PrepareRHS(const double t, const  double pr, const  double H,
                                  const double* __restrict__ Y, const double* __restrict__ Z1,
                                  const double* __restrict__ Z2, const double* __restrict__ Z3,
                                  double* __restrict__ R1, double* __restrict__ R2, double* __restrict__ R3,
                                  double* __restrict__ TMP, double* __restrict__ F1);

        /**
         * \brief Solves for the RHS values in the Newton iteration
         */
        void RK_Solve(const double H, double* __restrict__ E1,
                      std::complex<double>* __restrict__ E2, double* __restrict__ R1,
                      double* __restrict__ R2, double* __restrict__ R3, int* __restrict__ ipiv1,
                      int* __restrict__ ipiv2);

        /**
         * \brief Computes the scaled error norm from the given `scale` and `DY` vectors
         */
        inline double RK_ErrorNorm(const double* __restrict__ scale, double* __restrict__ DY);

        /**
         * \brief Computes and returns the error estimate for this step
         */
        double RK_ErrorEstimate(const double H, const double t, const double pr,
                                const double* __restrict__ Y, const double* __restrict__ F0,
                                double* __restrict__ F1, double* __restrict__ F2, double* __restrict__ TMP,
                                const double* __restrict__ Z1, const double* __restrict__ Z2,
                                const double* __restrict__ Z3, const double* __restrict__ scale,
                                double* __restrict__ E1, int* __restrict__ ipiv1,
                                const bool FirstStep, const bool Reject);

    };
}

#endif
