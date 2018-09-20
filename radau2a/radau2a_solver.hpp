/**
 * \file
 * \brief Definition of the RadauIIA CPU solver
 * \author Nicholas Curtis
 * \date 09/19/2019
 */

#ifndef RADAU2A_SOLVER_H
#define RADAU2A_SOLVER_H

#include "header.h"
#include <stdio.h>
#include "solver.hpp"
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

    // and the union of the two
    class Radau : public Solver
    {

    public:
        //! the matrix dimensions
        static constexpr int STRIDE = NSP;

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

        /**
         * @}
         */

        //! Lapack - non-transpose
        static constexpr char TRANS = 'N';
        //! Lapack - 1 RHS solve
        static constexpr int NRHS = 1;
        //! Lapack - Array size
        static constexpr int ARRSIZE = NSP;
    };

    class RadauIntegrator : public Integrator
    {
        public:

            RadauIntegrator(int numThreads) : Integrator(numThreads)
            {
            }


            /*! checkError
                \brief Checks the return code of the given thread (IVP) for an error, and exits if found
                \param tid The thread (IVP) index
                \param code The return code of the thread
                @see ErrorCodes
            */
            void checkError(int tid, ErrorCode code) const
            {
                switch(code)
                {
                    case ErrorCode::MAX_CONSECUTIVE_ERRORS_EXCEEDED:
                        printf("During integration of ODE# %d, an error occured on too many consecutive integration steps,"
                                "exiting...\n", tid);
                        exit(code);
                    case ErrorCode::MAX_STEPS_EXCEEDED:
                        printf("During integration of ODE# %d, the allowed number of integration steps was exceeded,"
                                "exiting...\n", tid);
                        exit(code);
                    case ErrorCode::H_PLUS_T_EQUALS_H:
                        printf("During integration of ODE# %d, the stepsize 'h' was decreased such that h = t + h,"
                                "exiting...\n", tid);
                        exit(code);
                    case ErrorCode::MAX_NEWTON_ITER_EXCEEDED:
                        printf("During integration of ODE# %d, the allowed number of newton iteration steps was exceeded,"
                                "exiting...\n", tid);
                        exit(code);
                }
            }

            /*!
               \fn char* solverName()
               \brief Returns a descriptive solver name
            */
            const char* solverName() const {
                const char* name = "radau2a-int";
                return name;
            }

            void clean() {
                // pass
            }
            void reinitialize(int numThreads) {
                // pass
            }

            void initSolverLog() {
                // pass
            }

            void solverLog() {
                // pass
            }

            std::size_t requiredSolverMemorySize() const
            {
                // C-solvers don't require pre-allocation
                return 0;
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
                const double t_start, const double t_end, const double pr, double* y) const;

        };
}

#endif
