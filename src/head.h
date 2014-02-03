/**
 * @mainpage Exponential integration of ROBER stiff problem.
 *
 * @author <a href="mailto:niemeyer@case.edu">Kyle E. Niemeyer</a> and Jerry C. Lee
 *
 * Change stiffness using stiffness factor eps, in head.h.
 */

#ifndef HEAD
#define HEAD

/** Header file for CSP model problem project.
 * \file head.h
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

/** Number of species.
 * 0 H
 * 1 H2
 * 2 O
 * 3 OH
 * 4 H2O
 * 5 O2
 * 6 HO2
 * 7 H2O2
 * 8 N2
 * 9 AR
 * 10 HE
 * 11 CO
 * 12 CO2
 */
#define NSP 13
/** Number of variables. NN = NSP + 1 (temperature). */
#define NN 14


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

/** Struct holding information for exp4 integrator. */
typedef struct {
	double h_old; /** Previous timestep */
	double err_old[NN]; /** Error from previous step */
} exp4_info;

#define A(I,J)	A[(I) + (J) * (NN)]
#define invA(I,J)		invA[(I) + (J) * (NN)]

/** Max macro (type safe, from GNU) */
#define MAX(a,b) ({ __typeof__ (a) _a = (a); __typeof__ (b) _b = (b); _a > _b ? _a : _b; })

/** Min macro (type safe) */
#define MIN(a,b) ({ __typeof__ (a) _a = (a); __typeof__ (b) _b = (b); _a < _b ? _a : _b; })

// OpenMP
#ifdef _OPENMP
	#include <omp.h>
#else
	#define omp_get_max_threads() 1
 	#define omp_get_num_threads() 1
#endif

#endif