/* rb43_init.cu
*  Implementation of the necessary initialization for the 4th order (3rd order embedded) Rosenbrock Solver
 * \file rb43_init.cu
 *
 * \author Nicholas Curtis
 * \date 03/09/2015
 *
 */

#include "rational_approximant.cuh"
#include "solver_options.h"
#include "solver_props.cuh"
#include "gpu_macros.cuh"

 void initialize_solver() {
    find_poles_and_residuals();
 }

 const char* solver_name() {
    const char* name = "rb43-int-gpu";
    return name;
 }

#ifdef LOG_OUTPUT
    //make logging array definitions
    __device__ double err_log[MAX_STEPS];
    __device__ int m_log[MAX_STEPS];
    __device__ int m1_log[MAX_STEPS];
    __device__ int m2_log[MAX_STEPS];
    __device__ double t_log[MAX_STEPS];
    __device__ double h_log[MAX_STEPS];
    __device__ bool reject_log[MAX_STEPS];
    __device__ int num_integrator_steps;
    double err_log_host[MAX_STEPS];
    int m_log_host[MAX_STEPS];
    int m1_log_host[MAX_STEPS];
    int m2_log_host[MAX_STEPS];
    double t_log_host[MAX_STEPS];
    double h_log_host[MAX_STEPS];
    bool reject_log_host[MAX_STEPS];
    int num_integrator_steps_host;
    FILE* logFile = 0;
    FILE* rFile = 0;
#endif

 void solver_log() {
 #ifdef LOG_OUTPUT
 	//first copy back num steps to make sure we're inbounds
    cudaErrorCheck( cudaMemcpyFromSymbol(&num_integrator_steps_host, num_integrator_steps, sizeof(int)) );
    if (num_integrator_steps_host == -1)
        exit(-1);
    //otherwise copy back
    cudaErrorCheck( cudaMemcpyFromSymbol(err_log_host, err_log, num_integrator_steps_host * sizeof(double)) );
    cudaErrorCheck( cudaMemcpyFromSymbol(m_log_host, m_log, num_integrator_steps_host * sizeof(int)) );
    cudaErrorCheck( cudaMemcpyFromSymbol(m1_log_host, m1_log, num_integrator_steps_host * sizeof(int)) );
    cudaErrorCheck( cudaMemcpyFromSymbol(m2_log_host, m2_log, num_integrator_steps_host * sizeof(int)) );
    cudaErrorCheck( cudaMemcpyFromSymbol(t_log_host, t_log, num_integrator_steps_host * sizeof(double)) );
    cudaErrorCheck( cudaMemcpyFromSymbol(h_log_host, h_log, num_integrator_steps_host * sizeof(double)) );
    cudaErrorCheck( cudaMemcpyFromSymbol(reject_log_host, reject_log, num_integrator_steps_host * sizeof(bool)) );
    //and print
    for (int i = 0; i < num_integrator_steps_host; ++i)
    {
        if (reject_log_host[i])
        {
            fprintf(rFile, "%.15le\t%.15le\t%.15le\t%d\t%d\t%d\n", t_log_host[i], h_log_host[i], err_log_host[i], m_log_host[i], m1_log_host[i], m2_log_host[i]);
        }
        else
        {
            fprintf(logFile, "%.15le\t%.15le\t%.15le\t%d\t%d\t%d\n", t_log_host[i], h_log_host[i], err_log_host[i], m_log_host[i], m1_log_host[i], m2_log_host[i]);
        }
    }
 #endif
 }

 void init_solver_log() {
 #ifdef LOG_OUTPUT
	//file for krylov logging
	//open and clear
	const char* f_name = solver_name();
	int len = strlen(f_name);
	char out_name[len + 17];
	sprintf(out_name, "log/%s-kry-log.txt", f_name);
	logFile = fopen(out_name, "w");

	char out_reject_name[len + 23];
	sprintf(out_reject_name, "log/%s-kry-reject.txt", f_name);    
	//file for krylov logging
	//open and clear
	rFile = fopen(out_reject_name, "w");
 #endif
 }

 size_t required_solver_size() {
    //return the size (in bytes), needed per cuda thread
    size_t num_bytes = 0;
    //scale array
    num_bytes += NSP * sizeof(double);
    //three work arrays
    num_bytes += 3 * STRIDE * sizeof(double);
    //gy
    num_bytes += NSP * sizeof(double);
    //Hm, phiHm
    num_bytes += 2 * STRIDE * STRIDE * sizeof(double);
    //Vm
    num_bytes += NSP * STRIDE * sizeof(double);
    //saved actions
    num_bytes += 5 * NSP * sizeof(double);
    //one pivot array
    num_bytes += STRIDE * sizeof(int);
    //complex inverse
    num_bytes += STRIDE * STRIDE * sizeof(cuDoubleComplex);
    //complex work array
    num_bytes += STRIDE * sizeof(cuDoubleComplex);
    //result flag
    num_bytes += 1 * sizeof(int);
    
    return num_bytes;
 }

 void createAndZero(void** ptr, size_t size)
{
  cudaErrorCheck(cudaMalloc(ptr, size));
  cudaErrorCheck(cudaMemset(*ptr, 0, size));
}

void initialize_solver(int padded, solver_memory** h_mem, solver_memory** d_mem) {
  find_poles_and_residuals();
  // Allocate storage for the device struct
  cudaErrorCheck( cudaMalloc(d_mem, sizeof(solver_memory)) );
  //allocate the device arrays on the host pointer
  createAndZero((void**)&((*h_mem)->sc), NSP * padded * sizeof(double));
  createAndZero((void**)&((*h_mem)->work1), NSP * padded * sizeof(double));
  createAndZero((void**)&((*h_mem)->work2), NSP * padded * sizeof(double));
  createAndZero((void**)&((*h_mem)->work3), NSP * padded * sizeof(double));
  createAndZero((void**)&((*h_mem)->gy), NSP * padded * sizeof(double));
  createAndZero((void**)&((*h_mem)->Hm), STRIDE * STRIDE * padded * sizeof(double));
  createAndZero((void**)&((*h_mem)->phiHm), STRIDE * STRIDE * padded * sizeof(double));
  createAndZero((void**)&((*h_mem)->Vm), NSP * STRIDE * padded * sizeof(double));
  createAndZero((void**)&((*h_mem)->savedActions), 5 * NSP * padded * sizeof(double));
  createAndZero((void**)&((*h_mem)->ipiv), NSP * sizeof(int));
  createAndZero((void**)&((*h_mem)->invA), STRIDE * STRIDE * padded * sizeof(cuDoubleComplex));
  createAndZero((void**)&((*h_mem)->work4), NSP * padded * sizeof(double));
  createAndZero((void**)&((*h_mem)->result), padded * sizeof(int));

  //copy host struct to device
  cudaErrorCheck( cudaMemcpy(*d_mem, *h_mem, sizeof(solver_memory), cudaMemcpyHostToDevice) );
}

 void cleanup_solver(solver_memory** h_mem, solver_memory** d_mem) {
 #ifdef LOG_OUTPUT
    //close files
    fclose(rFile);
    fclose(logFile);
 #endif
    cudaErrorCheck( cudaFree((*h_mem)->sc) );
    cudaErrorCheck( cudaFree((*h_mem)->work1) );
    cudaErrorCheck( cudaFree((*h_mem)->work2) );
    cudaErrorCheck( cudaFree((*h_mem)->work3) );
    cudaErrorCheck( cudaFree((*h_mem)->gy) );
    cudaErrorCheck( cudaFree((*h_mem)->Hm) );
    cudaErrorCheck( cudaFree((*h_mem)->phiHm) );
    cudaErrorCheck( cudaFree((*h_mem)->Vm) );
    cudaErrorCheck( cudaFree((*h_mem)->savedActions) );
    cudaErrorCheck( cudaFree((*h_mem)->ipiv) );
    cudaErrorCheck( cudaFree((*h_mem)->invA) );
    cudaErrorCheck( cudaFree((*h_mem)->work4) );
    cudaErrorCheck( cudaFree((*h_mem)->result) );
    cudaErrorCheck( cudaFree(*d_mem) );
 }