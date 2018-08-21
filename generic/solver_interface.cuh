/**
 * \file
 * \brief Interface implementation for GPU solvers to be called as a library
 *
 * \author Nicholas Curtis
 * \date 03/09/2015
 *
 * Contains initialization, integration and cleanup header definitions
 */

#ifndef SOLVER_INTERFACE_CUH
#define SOLVER_INTERFACE_CUH

#include "solver.cuh"
#include "solver_init.cuh"
#include "launch_bounds.cuh"
#include "gpu_macros.cuh"
#include "gpu_memory.cuh"
#include "header.cuh"
#include "solver_props.cuh"
#include <stdio.h>
#include <float.h>

#define EPS DBL_EPSILON

#ifdef GENERATE_DOCS
namespace genericcu {
#endif

/**
 * \brief Initializes the solver
 * \param[in]       NUM         The number of ODEs to integrate
 * \param[in]       device      The CUDA device number, if < 0 set to the first available GPU
 */
void accelerInt_initialize(int NUM, int device);


/**
 * \brief integrate NUM odes from time `t_start` to time `t_end`, using stepsizes of `t_step`
 *
 * \param[in]           NUM             The number of ODEs to integrate.  This should be the size of the leading dimension of `y_host` and `var_host`.  @see accelerint_indx
 * \param[in]           t_start         The system time
 * \param[in]           t_end           The end time
 * \param[in]           stepsize        The integration step size.  If `stepsize` < 0, the step size will be set to `t_end - t`
 * \param[in,out]       y_host          The state vectors to integrate.
 * \param[in]           var_host        The parameters to use in dydt() and eval_jacob()
 *
 */
void accelerInt_integrate(const int NUM, const double t_start, const double t_end, const double stepsize,
                          double * __restrict__ y_host, const double * __restrict__ var_host);

/**
 * \brief Cleans up the solver
 */
void accelerInt_cleanup();




#ifdef GENERATE_DOCS
}
#endif


#endif