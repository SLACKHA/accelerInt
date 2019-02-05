/*!
 * \file
 * \brief Initializes and calculates required GPU memory
 */

#include "gpu_memory.cuh"

#ifdef GENERATE_DOCS
//put this in the Oregonator namespace for documentation
namespace oregonator_cu {
#endif

/**
 * \brief Calculates and returns the total memory size (in bytes) required by an individual thread for the
 *        mechanism_memory struct.
 */
size_t required_mechanism_size() {
  //returns the total required size for the mechanism per thread
  size_t mech_size = 0;
  //state vector y
  mech_size += NSP;
  //dydt vector
  mech_size += NSP;
  //Jacobian
  mech_size += NSP * NSP;
  //and mu parameter
  mech_size += 1;
  return mech_size * sizeof(double);
}
/**
 * \brief Initializes the host and device mechanism_memory structs.  This is required in order to enable
 *        passing the struct to CUDA
 * \param[in]               padded              The padded number of threads to be used by the CUDA solver
 * \param[in,out]           h_mem               The host version of the mechanism_memory struct to initialize.
 * \param[in,out]           d_mem               The device version of the mechanism_memory struct to copy the resulting host mechanism_memory struct to.
 */
void initialize_gpu_memory(int padded, mechanism_memory** h_mem, mechanism_memory** d_mem)
{
  // Allocate storage for the device struct
  cudaErrorCheck( cudaMalloc(d_mem, sizeof(mechanism_memory)) );
  //allocate the device arrays on the host pointer
  cudaErrorCheck( cudaMalloc(&((*h_mem)->y), NSP * padded * sizeof(double)) );
  cudaErrorCheck( cudaMalloc(&((*h_mem)->dy), NSP * padded * sizeof(double)) );
  cudaErrorCheck( cudaMalloc(&((*h_mem)->var), 1 * padded * sizeof(double)) );
  cudaErrorCheck( cudaMalloc(&((*h_mem)->jac), NSP * NSP * padded * sizeof(double)) );
  // set non-initialized values to zero
  cudaErrorCheck( cudaMemset((*h_mem)->dy, 0, NSP * padded * sizeof(double)) );
  cudaErrorCheck( cudaMemset((*h_mem)->jac, 0, NSP * NSP * padded * sizeof(double)) );
  // and copy to device pointer
  cudaErrorCheck( cudaMemcpy(*d_mem, *h_mem, sizeof(mechanism_memory), cudaMemcpyHostToDevice) );
}
/**
 * \brief Frees the host and device mechanism_memory structs
 * \param[in,out]           h_mem               The host version of the mechanism_memory struct.
 * \param[in,out]           d_mem               The device version of the mechanism_memory struct.
 */
void free_gpu_memory(mechanism_memory** h_mem, mechanism_memory** d_mem)
{
  cudaErrorCheck(cudaFree((*h_mem)->y));
  cudaErrorCheck(cudaFree((*h_mem)->dy));
  cudaErrorCheck(cudaFree((*h_mem)->var));
  cudaErrorCheck(cudaFree((*h_mem)->jac));
  cudaErrorCheck(cudaFree(*d_mem));
}

#ifdef GENERATE_DOCS
}
#endif
