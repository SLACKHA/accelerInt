/**
 * \file
 * \brief Implementation of the necessary initialization for the RKC GPU solver
 *
 * \author Nicholas Curtis
 * \date 08/12/2017
 *
 */

 #include "solver_init.cuh"

#ifdef GENERATE_DOCS
namespace rkc_cu {
#endif

 void init_solver_log() {

 }

 void solver_log() {

 }

/*!
   \fn char* solver_name()
   \brief Returns a descriptive solver name
*/
 const char* solver_name() {
    const char* name = "rkc-gpu";
    return name;
 }

 /*!
   \brief Returns the total size (in bytes) required for memory storage for a single GPU thread
   Used in calculation of the maximum number of possible GPU threads to launch, this method
   returns the size of the solver_memory structure (per-GPU thread)
   @see solver_memory
 */
 size_t required_solver_size() {
    //return the size (in bytes), needed per cuda thread
    size_t num_bytes = 0;
    // state vector
    num_bytes += NSP * sizeof(double);
    // derivatives
    num_bytes += NSP * sizeof(double);
    // work array with 4 extra spots
    num_bytes += (4 + NSP) * sizeof(double);
    // four regular work arrays of size NSP
    num_bytes += 4 * NSP * sizeof(double);
    // result array
    num_bytes += 1 * sizeof(int);

    return num_bytes;
 }

/*!
 * \brief Convience method to Cuda Malloc and memset a pointer to zero
 * \param ptr The address of the pointer to malloc
 * \param size The total size (in bytes) of the pointer to malloc
 */
void createAndZero(void** ptr, size_t size)
{
  cudaErrorCheck(cudaMalloc(ptr, size));
  cudaErrorCheck(cudaMemset(*ptr, 0, size));
}

/*!
   \brief Initializes solver_memory
   \param padded The total (padded) number of GPU threads (IVPs) to solve
   \param h_mem The host solver_memory structure (to be copied to the GPU)
   \param d_mem The device solver_memory structure (to be operated on by the GPU)
*/
void initialize_solver(const int padded, solver_memory** h_mem, solver_memory** d_mem) {
  // Allocate storage for the device struct
  cudaErrorCheck( cudaMalloc(d_mem, sizeof(solver_memory)) );
  //allocate the device arrays on the host pointer
  createAndZero((void**)&((*h_mem)->y_n), NSP * padded * sizeof(double));
  createAndZero((void**)&((*h_mem)->F_n), NSP * padded * sizeof(double));
  createAndZero((void**)&((*h_mem)->work), (NSP + 4) * padded * sizeof(double));
  createAndZero((void**)&((*h_mem)->temp_arr), NSP * padded * sizeof(double));
  createAndZero((void**)&((*h_mem)->temp_arr2), NSP * padded * sizeof(double));
  createAndZero((void**)&((*h_mem)->y_jm1), NSP * padded * sizeof(double));
  createAndZero((void**)&((*h_mem)->y_jm2), NSP * padded * sizeof(double));
  createAndZero((void**)&((*h_mem)->result), padded * sizeof(int));

  //copy host struct to device
  cudaErrorCheck( cudaMemcpy(*d_mem, *h_mem, sizeof(solver_memory), cudaMemcpyHostToDevice) );
}

/*!
   \fn void cleanup_solver(solver_memory** h_mem, solver_memory** d_mem)
   \brief Cleans up solver memory
   @see solver_memory
   @see solver_options.cuh
*/
 void cleanup_solver(solver_memory** h_mem, solver_memory** d_mem) {
  cudaErrorCheck(cudaFree((*h_mem)->y_n));
  cudaErrorCheck(cudaFree((*h_mem)->F_n));
  cudaErrorCheck(cudaFree((*h_mem)->work));
  cudaErrorCheck(cudaFree((*h_mem)->temp_arr));
  cudaErrorCheck(cudaFree((*h_mem)->temp_arr2));
  cudaErrorCheck(cudaFree((*h_mem)->y_jm1));
  cudaErrorCheck(cudaFree((*h_mem)->y_jm2));
  cudaErrorCheck(cudaFree((*h_mem)->result));
  cudaErrorCheck(cudaFree(*d_mem));
}