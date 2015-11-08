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

 void cleanup_solver() {
 #ifdef LOG_OUTPUT
    //close files
    fclose(rFile);
    fclose(logFile);
 #endif
 }