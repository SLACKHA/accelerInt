/*!
 * \file
 * \brief Headers for GPU memory initialization
 */

<<<<<<< HEAD
=======
#ifndef GPU_MEM_CUH
#define GPU_MEM_CUH

>>>>>>> master
#include "header.cuh"

#ifdef GENERATE_DOCS
//put this in the van der Pol namespace for documentation
namespace van_der_pol_cu {
#endif

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
 * for evaluating chemical kinetic reactions.
 */
struct mechanism_memory {
  double * y;
  double * dy;
  double * var;
  double * jac;
};

/**
 * \brief Calculates and returns the total memory size (in bytes) required by an individual thread for the
 *        mechanism_memory struct.
 */
size_t required_mechanism_size();
/**
 * \brief Initializes the host and device mechanism_memory structs.  This is required in order to enable
 *        passing the struct to CUDA
 * \param[in]               padded              The padded number of threads to be used by the CUDA solver
 * \param[in,out]           h_mem               The host version of the mechanism_memory struct to initialize.
 * \param[in,out]           d_mem               The device version of the mechanism_memory struct to copy the resulting host mechanism_memory struct to.
 */
void initialize_gpu_memory(int padded, mechanism_memory** h_mem, mechanism_memory** d_mem);
/**
 * \brief Frees the host and device mechanism_memory structs
 * \param[in,out]           h_mem               The host version of the mechanism_memory struct.
 * \param[in,out]           d_mem               The device version of the mechanism_memory struct.
 */
void free_gpu_memory(mechanism_memory** h_mem, mechanism_memory** d_mem);

#ifdef GENERATE_DOCS
}
#endif
<<<<<<< HEAD
=======

#endif
>>>>>>> master
