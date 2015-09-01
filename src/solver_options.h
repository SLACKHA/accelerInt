/*
solver_options.h

A file that in conjunction with the makefile can specify the various options
to the solvers

*/

#ifdef LOW_TOL
    /** Absolute tolerance */
    #define ATOL (1.0E-10)

    /** Relative tolerance */
    #define RTOL (1.0E-6)

#else
    /** Absolute tolerance */
    #define ATOL (1.0E-15)

    /** Relative tolerance */
    #define RTOL (1.0E-8)
#endif

#ifdef LARGE_STEP
    #define t_step (1e-4)
#else
    #define t_step (1e-6)
#endif

/** Machine precision constant. */
#define EPS DBL_EPSILON

#define SMALL DBL_MIN

/** type of rational approximant (n, n) */
#define N_RA 10

/** Unsigned int typedef. */
typedef unsigned int uint;
/** Unsigned short int typedef. */
typedef unsigned short int usint;

/* CVodes Parameters */
//#define CV_MAX_ORD 5 //maximum order for method, default for BDF is 5
#define CV_MAX_STEPS 20000 // maximum steps the solver will take in one timestep
//#define CV_HMAX 0  //upper bound on step size (integrator step, not global timestep)
//#define CV_HMIN 0 //lower bound on step size (integrator step, not global timestep)
#define CV_MAX_HNIL 1 //maximum number of t + h = t warnings
#define CV_MAX_ERRTEST_FAILS 5 //maximum number of error test fails before an error is thrown

//#define COMPILE_TESTING_METHODS //comment out to remove unit testing stubs

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