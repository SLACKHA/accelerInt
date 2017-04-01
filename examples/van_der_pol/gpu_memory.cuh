/*!
 * \file
 * \brief Headers for GPU memory initialization
 */

#include "header.cuh"

#ifdef GENERATE_DOCS
//put this in the van der Pol namespace for documentation
namespace van_der_pol_cu {
#endif

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
