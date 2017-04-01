/*!
 * \file
 * \brief An example header file that defines system size, memory functions and other required methods
 *        for integration of the van der Pol's equation with the CUDA solvers
 */

#ifndef HEADER_GUARD_CUH
#define HEADER_GUARD_CUH

#include <stdlib.h>
#include "gpu_macros.cuh"

#ifdef GENERATE_DOCS
//put this in the van der Pol namespace for documentation
namespace van_der_pol_cu {
#endif

//! The IVP system size
#define NSP (2)
//! Input vector size (in read_initial_conditions)
#define NN (NSP)

/*!
 *
 * \brief Set same ICs for all problems
 * \param NUM       The number of initial value problems
 * \param y_host    The state vectors to initialize
 * \param var_host  The vector of \f$mu\f$ parameters for the van der Pol equation
 *
 */
void set_same_initial_conditions(int NUM, double** y_host, double** var_host);

//dummy definitions that are used for pyJac

/**
 * \brief Not needed for van der Pol
 *
 *  In pyJac, these are used to transform the input/output vectors to deal with moving the
 *         last species mass fraction
 */
void apply_mask(double* y_host);
/**
 * \brief Not needed for van der Pol
 *
 *  In pyJac, these are used to transform the input/output vectors to deal with moving the
 *         last species mass fraction
 */
void apply_reverse_mask(double* y_host);

/**
 * \brief This struct is used to store memory for the CUDA RHS and Jacobian evaluation.
 *        Along with the solver_memory struct, this must be initialized and passed to the solvers
 *        the run.  @see solver_memory, initialize_gpu_memory, required_mechanism_size
 *
 * \param           y               The global state vector arrays
 * \param           dy              The global dydt vector arrays
 * \param           var             The global param array [Used for \f$\mu\f$]
 * \param           jac             The global Jacobian arrays
 *
 * This is also heavily used when using with @pyJac to hold additional arrays
 * for evaluating chemical kinetic reactions.  For an example see \ref pyJac_ex
 */
struct mechanism_memory {
  double * y;
  double * dy;
  double * var;
  double * jac;
};

#ifdef GENERATE_DOCS
}
#endif

#endif
