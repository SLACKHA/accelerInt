#ifndef HEADER
#define HEADER

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdbool.h>

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

// OpenMP
#ifdef _OPENMP
	#include <omp.h>
#else
	#define omp_get_max_threads() 1
 	#define omp_get_num_threads() 1
#endif

#endif
