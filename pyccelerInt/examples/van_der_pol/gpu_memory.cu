/*!
 * \file
 * \brief Initializes and calculates required GPU memory
 */

#include "gpu_memory.cuh"

#ifdef GENERATE_DOCS
//put this in the van der Pol namespace for documentation
namespace van_der_pol_cu {
#endif

/**
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
