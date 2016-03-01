/* radau2a_init.cu
*  Implementation of the necessary initialization for the Radau2A solver
 * \file radau2a_init.cu
 *
 * \author Nicholas Curtis
 * \date 03/09/2015
 *
 */

 #include "header.cuh"

 void init_solver_log() {
 	
 }

 void solver_log() {
 	
 }

void initialize_solver(int padded, solver_memory** h_mem, solver_memory** d_mem) {
  // Allocate storage for the device struct
  cudaErrorCheck( cudaMalloc(d_mem, sizeof(solver_memory)) );
  //allocate the device arrays on the host pointer
  cudaErrorCheck( cudaMalloc(&((*h_mem)->E1), NSP * NSP * padded * sizeof(double)) );
  cudaErrorCheck( cudaMalloc(&((*h_mem)->E2), NSP * NSP * padded * sizeof(cuDoubleComplex)) );
  cudaErrorCheck( cudaMalloc(&((*h_mem)->ipiv1), NSP * padded * sizeof(int)) );
  cudaErrorCheck( cudaMalloc(&((*h_mem)->ipiv2), NSP * padded * sizeof(int)) );
  cudaErrorCheck( cudaMalloc(&((*h_mem)->sc), NSP * padded * sizeof(double)) );
  cudaErrorCheck( cudaMalloc(&((*h_mem)->Z1), NSP * padded * sizeof(double)) );
  cudaErrorCheck( cudaMalloc(&((*h_mem)->Z2), NSP * padded * sizeof(double)) );
  cudaErrorCheck( cudaMalloc(&((*h_mem)->Z3), NSP * padded * sizeof(double)) );
  cudaErrorCheck( cudaMalloc(&((*h_mem)->DZ1), NSP * padded * sizeof(double)) );
  cudaErrorCheck( cudaMalloc(&((*h_mem)->DZ2), NSP * padded * sizeof(double)) );
  cudaErrorCheck( cudaMalloc(&((*h_mem)->DZ3), NSP * padded * sizeof(double)) );
  cudaErrorCheck( cudaMalloc(&((*h_mem)->CONT), 3 * NSP * padded * sizeof(double)) );
  cudaErrorCheck( cudaMalloc(&((*h_mem)->y0), NSP * padded * sizeof(double)) );
  cudaErrorCheck( cudaMalloc(&((*h_mem)->F0), NSP * padded * sizeof(double)) );
  cudaErrorCheck( cudaMalloc(&((*h_mem)->work1), NSP * padded * sizeof(double)) );
  cudaErrorCheck( cudaMalloc(&((*h_mem)->work2), NSP * padded * sizeof(double)) );
  cudaErrorCheck( cudaMalloc(&((*h_mem)->work3), NSP * padded * sizeof(double)) );
  cudaErrorCheck( cudaMalloc(&((*h_mem)->work4), NSP * padded * sizeof(cuDoubleComplex)) );
  cudaErrorCheck( cudaMalloc(&((*h_mem)->result), padded * sizeof(double)) );

  //copy host struct to device
  cudaErrorCheck( cudaMemcpy(*d_mem, *h_mem, sizeof(solver_memory), cudaMemcpyHostToDevice) );
}

 const char* solver_name() {
 	const char* name = "radau2a-int-gpu";
 	return name;
 }

 void calculate_required_size() {
 	//return the size (in bytes), needed per cuda thread
 	size_t num_bytes = 0;
 	//Jacobian (doubles as factorized)
 	num_bytes += NSP * NSP;
 	//two pivot index arrays
 	num_bytes += 2 * NSP;
 	//an error scale array
 	num_bytes += NSP;
 	//6 RHS and interpolant arrays
 	num_bytes += 6 * NSP;
 	//continuation array of size 3 * NSP
 	num_bytes += 3 * NSP;
 	//y0
 	num_bytes += NSP;
 	//F0
 	num_bytes += NSP;
 	//3 work arrays
 	num_bytes += 3 * NSP;
  //result flag
  num_bytes += 1;
 	//convert to bytes
 	num_bytes *= sizeof(double);
 	//and add complex jacobian factorization
 	num_bytes += NSP * NSP * sizeof(cuDoubleComplex);
 }

 void cleanup_solver() {
 	//nothing to do
 }

void free_gpu_memory(solver_memory** h_mem, solver_memory** d_mem, double** d_y, double** d_var)
{
  cudaErrorCheck(cudaFree((*h_mem)->spec_rates));
  cudaErrorCheck(cudaFree((*h_mem)->rev_rates));
  cudaErrorCheck(cudaFree((*h_mem)->conc));
  cudaErrorCheck(cudaFree((*h_mem)->dy));
  cudaErrorCheck(cudaFree((*h_mem)->dBdT));
  cudaErrorCheck(cudaFree((*h_mem)->cp));
  cudaErrorCheck(cudaFree((*h_mem)->dot_prod));
  cudaErrorCheck(cudaFree((*h_mem)->h));
  cudaErrorCheck(cudaFree((*h_mem)->fwd_rates));
  cudaErrorCheck(cudaFree((*h_mem)->pres_mod));
  cudaErrorCheck(cudaFree((*h_mem)->y));
  cudaErrorCheck(cudaFree((*h_mem)->jac));
  cudaErrorCheck(cudaFree(*d_mem));
  cudaErrorCheck(cudaFree(*d_y));
  cudaErrorCheck(cudaFree(*d_var));
}