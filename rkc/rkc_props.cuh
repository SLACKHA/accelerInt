/**
 * \file
 * \brief Various macros controlling behaviour of RKC algorithm
 * \author Nicholas Curtis
 * \date 08/12/2017
 */

#ifndef RKC_PROPS_CUH
#define RKC_PROPS_CUH

#include "header.cuh"
#include <stdio.h>

#ifdef GENERATE_DOCS
namespace rkc_cu {
#endif

/** Set double precision */
#define DOUBLE

#ifdef DOUBLE
    #define Real double

    #define ZERO 0.0
    #define ONE 1.0
    #define TWO 2.0
    #define THREE 3.0
    #define FOUR 4.0

    #define TEN 10.0
    #define ONEP1 1.1
    #define ONEP2 1.2
    #define ONEP54 1.54
    #define P8 0.8
    #define P4 0.4
    #define P1 0.1
    #define P01 0.01
    #define ONE3RD (1.0 / 3.0)
    #define TWO3RD (2.0 / 3.0)
    #define UROUND (2.22e-16)
#else
    #define Real float

    #define ZERO 0.0f
    #define ONE 1.0f
    #define TWO 2.0f
    #define THREE 3.0f
    #define FOUR 4.0f

    #define TEN 10.0f
    #define ONEP1 1.1f
    #define ONEP2 1.2f
    #define ONEP54 1.54f
    #define P8 0.8f
    #define P4 0.4f
    #define P1 0.1f
    #define P01 0.01f
    #define ONE3RD (1.0f / 3.0f)
    #define TWO3RD (2.0f / 3.0f)
    #define UROUND (2.22e-16)
#endif

//! Memory required for Radau-IIa GPU solver
struct solver_memory
{
    //! Initial state vectors
    Real* y_n;
    //! The derivative vectors
    Real* F_n;
    //! The a work vector
    Real* work;
    //! The a work vector
    Real* temp_arr;
    //! The a work vector
    Real* temp_arr2;
    //! The a work vector
    Real* y_jm1;
    //! The a work vector
    Real* y_jm2;
    //! array of return codes @see RKCCU_ErrCodes
    int* result;
};

/**
 * \addtogroup RKCErrorCodes Return codes of Integrators
 * @{
 */
/**
 * \defgroup RKCCU_ErrCodes Return codes of GPU RKC Integrator
 * @{
 */

//! Successful time step
#define EC_success (0)
/**
 * @}
 */
/**
 * @}
 */

#ifdef GENERATE_DOCS
}
#endif

#endif