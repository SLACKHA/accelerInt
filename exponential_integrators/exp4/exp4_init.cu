/**
 * \file
 * \brief Implementation of the necessary initialization for the EXP4 method
 *
 * \author Nicholas Curtis
 * \date 03/09/2015
 *
 */

#include "rational_approximant.cuh"
#include "solver_options.cuh"
#include "solver_props.cuh"
#include "gpu_macros.cuh"

#ifdef GENERATE_DOCS
namespace exp4cu {
#endif

/*!
 * \fn void createAndZero(void** ptr, size_t size)
 * \brief Convienvience method to Cuda Malloc and memset a pointer to zero
 * \param ptr The address of the pointer to malloc
 * \param size The total size (in bytes) of the pointer to malloc
 */
void createAndZero(void** ptr, size_t size)
{
  cudaErrorCheck(cudaMalloc(ptr, size));
  cudaErrorCheck(cudaMemset(*ptr, 0, size));
}

/*! \fn void initialize_solver(int padded, solver_memory** h_mem, solver_memory** d_mem)
   \brief Initializes the GPU solver
   \param padded The total (padded) number of GPU threads (IVPs) to solve
   \param h_mem The host solver_memory structure (to be copied to the GPU)
   \param d_mem The device solver_memory structure (to be operated on by the GPU)

   Solves for the poles and residuals used for the Rational Approximants in the Krylov subspace methods and initializes solver_memory
*/
void initialize_solver(int padded, solver_memory** h_mem, solver_memory** d_mem) {
    find_poles_and_residuals();
    // Allocate storage for the device struct
    cudaErrorCheck( cudaMalloc(d_mem, sizeof(solver_memory)) );
    //allocate the device arrays on the host pointer
    createAndZero((void**)&((*h_mem)->sc), NSP * padded * sizeof(double));
    createAndZero((void**)&((*h_mem)->work1), STRIDE * padded * sizeof(double));
    createAndZero((void**)&((*h_mem)->work2), STRIDE * padded * sizeof(double));
    createAndZero((void**)&((*h_mem)->work3), STRIDE * padded * sizeof(double));
    createAndZero((void**)&((*h_mem)->work4), STRIDE * padded * sizeof(cuDoubleComplex));
    createAndZero((void**)&((*h_mem)->Hm), STRIDE * STRIDE * padded * sizeof(double));
    createAndZero((void**)&((*h_mem)->phiHm), STRIDE * STRIDE * padded * sizeof(double));
    createAndZero((void**)&((*h_mem)->Vm), NSP * STRIDE * padded * sizeof(double));
    createAndZero((void**)&((*h_mem)->ipiv), NSP * padded * sizeof(int));
    createAndZero((void**)&((*h_mem)->invA), STRIDE * STRIDE * padded * sizeof(cuDoubleComplex));
    createAndZero((void**)&((*h_mem)->result), padded * sizeof(int));
    createAndZero((void**)&((*h_mem)->k1), NSP * padded * sizeof(double));
    createAndZero((void**)&((*h_mem)->k2), NSP * padded * sizeof(double));
    createAndZero((void**)&((*h_mem)->k3), NSP * padded * sizeof(double));
    createAndZero((void**)&((*h_mem)->k4), NSP * padded * sizeof(double));
    createAndZero((void**)&((*h_mem)->k5), NSP * padded * sizeof(double));
    createAndZero((void**)&((*h_mem)->k6), NSP * padded * sizeof(double));
    createAndZero((void**)&((*h_mem)->k7), NSP * padded * sizeof(double));

    //copy host struct to device
    cudaErrorCheck( cudaMemcpy(*d_mem, *h_mem, sizeof(solver_memory), cudaMemcpyHostToDevice) );
 }

/*!
   \fn char* solver_name()
   \brief Returns a descriptive solver name
*/
 const char* solver_name() {
    const char* name = "exp4-int-gpu";
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


/*!
   \fn solver_log()
   \brief Executes solver specific logging tasks

   Logs errors, step-sizes, and krylov subspace size (if #LOG_OUTPUT is defined)
   @see solver_options.cuh
*/
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

 /*!
   \fn init_solver_log()
   \brief Initializes solver specific items for logging

   Initializes the Krylov subspace logging files (if #LOG_OUTPUT is defined)
   @see solver_options.cuh
*/
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

/*!
   \fn size_t required_solver_size()
   \brief Returns the total size (in bytes) required for memory storage for a single GPU thread
   Used in calculation of the maximum number of possible GPU threads to launch, this method
   returns the size of the solver_memory structure (per-GPU thread)
   @see solver_memory
*/
  size_t required_solver_size() {
    //return the size (in bytes), needed per cuda thread
    size_t num_bytes = 0;
    //three work arrays
    num_bytes += 3 * STRIDE;
    //Hm, phiHm
    num_bytes += 2 * STRIDE * STRIDE;
    //Vm
    num_bytes += NSP * STRIDE;
    //7 k arrays
    num_bytes += 7 * NSP;
    //add all doubles
    num_bytes *= sizeof(double);
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

/*!
   \fn void cleanup_solver(solver_memory** h_mem, solver_memory** d_mem)
   \brief Cleans up solver memory
   @see solver_memory
   @see solver_options.cuh

   Additionally closes Krylov subspace logfiles (if #LOG_OUTPUT is defined)
*/
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
    cudaErrorCheck( cudaFree((*h_mem)->work4) );
    cudaErrorCheck( cudaFree((*h_mem)->Hm) );
    cudaErrorCheck( cudaFree((*h_mem)->phiHm) );
    cudaErrorCheck( cudaFree((*h_mem)->Vm) );
    cudaErrorCheck( cudaFree((*h_mem)->ipiv) );
    cudaErrorCheck( cudaFree((*h_mem)->invA) );
    cudaErrorCheck( cudaFree((*h_mem)->result) );
    cudaErrorCheck( cudaFree((*h_mem)->k1) );
    cudaErrorCheck( cudaFree((*h_mem)->k2) );
    cudaErrorCheck( cudaFree((*h_mem)->k3) );
    cudaErrorCheck( cudaFree((*h_mem)->k4) );
    cudaErrorCheck( cudaFree((*h_mem)->k5) );
    cudaErrorCheck( cudaFree((*h_mem)->k6) );
    cudaErrorCheck( cudaFree((*h_mem)->k7) );
    cudaErrorCheck( cudaFree(*d_mem) );
 }

#ifdef GENERATE_DOCS
}
#endif