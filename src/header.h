/**
 * @mainpage Exponential integration of ROBER stiff problem.
 *
 * @author <a href="mailto:niemeyer@case.edu">Kyle E. Niemeyer</a> and Jerry C. Lee
 *
 * Change stiffness using stiffness factor eps, in header.h.
 */

#ifndef HEAD
#define HEAD

/** Header file for CSP model problem project.
 * \file header.h
 *
 * \author Kyle E. Niemeyer
 * \date 07/17/2012
 *
 * Contains libraries, definitions, and constants.
 */

#include <stdlib.h>
#include <math.h>
#include <float.h>

/** Constant pressure or volume. */
#define CONP
//#define CONV

/** Include mechanism header to get NSP and NN **/
#ifdef __cplusplus
  #include "mechanism.cuh"
#else
  #include "mechanism.h"
#endif

/** Absolute tolerance */
#define ATOL (1.0E-15)

/** Relative tolerance */
#define RTOL (1.0E-8)

/** type of rational approximant (n, n) */
#define N_RA 10

/** Sets precision as double or float. */
#define DOUBLE

#ifdef DOUBLE
  /** Define Real as double. */
  #define Real double
  
  /** Double precision ZERO. */
  #define ZERO 0.0
  /** Double precision ONE. */
  #define ONE 1.0
  /** Double precision TWO. */
  #define TWO 2.0
  /** Double precision THREE. */
  #define THREE 3.0
  /** Double precision FOUR. */
  #define FOUR 4.0
  
  /** Machine precision constant. */
  #define EPS DBL_EPSILON

  #define SMALL DBL_MIN
#else
  /** Define Real as float. */
  #define Real float
  
  /** Single precision ZERO. */
  #define ZERO 0.0f
  /** Single precision ONE. */
  #define ONE 1.0f
  /** Single precision (float) TWO. */
  #define TWO 2.0f
  /** Single precision THREE. */
  #define THREE 3.0f
  /** Single precision FOUR. */
  #define FOUR 4.0f
  
  /** Machine precision constant. */
  #define EPS FLT_EPSILON

  #define SMALL FLT_MIN

#endif

/** Unsigned int typedef. */
typedef unsigned int uint;
/** Unsigned short int typedef. */
typedef unsigned short int usint;

// OpenMP
#ifdef _OPENMP
	#include <omp.h>
#else
	#define omp_get_max_threads() 1
 	#define omp_get_num_threads() 1
#endif

/* CVodes Parameters */
//#define CV_MAX_ORD 5 //maximum order for method, default for BDF is 5
#define CV_MAX_STEPS 20000 // maximum steps the solver will take in one timestep
//#define CV_HMAX 0  //upper bound on step size (integrator step, not global timestep)
//#define CV_HMIN 0 //lower bound on step size (integrator step, not global timestep)
#define CV_MAX_HNIL 0 //maximum number of t + h = t warnings
#define CV_MAX_ERRTEST_FAILS 5 //maximum number of error test fails before an error is thrown

#define COMPILE_TESTING_METHODS //comment out to remove unit testing stubs

//turn on to log the krylov space and step sizes to log.txt
#ifdef DEBUG
  #if defined(RB43) || defined(EXP4)
    #define LOG_KRYLOV_AND_STEPSIZES
  #endif
#endif

/* These are now controlled by the makefile 
// load same initial conditions for all threads
#define SAME_IC

// shuffle initial conditions randomly
#define SHUFFLE

//print the output to screen
#define PRINT

// output ignition time
#define IGN

//log output to file
#define LOG_OUTPUT
*/

#endif