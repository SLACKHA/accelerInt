/**
 * \file
 * \brief Defines some simple macros to simplify GPU indexing
 *
 */

#ifndef GPU_MACROS_CUH
#define GPU_MACROS_CUH
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>

#ifdef GENERATE_DOCS
//put this in the van der Pol namespace for documentation
namespace van_der_pol_cu {
#endif

//! The total number of threads in the Grid, provides an offset between vector entries
#define GRID_DIM (blockDim.x * gridDim.x)
//! The global CUDA thread index
#define T_ID (threadIdx.x + blockIdx.x * blockDim.x)
//! Convenience macro to get the value of a vector at index i, calculated as i * #GRID_DIM + #T_ID
#define INDEX(i) (T_ID + (i) * GRID_DIM)

#define cudaErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}


#ifdef GENERATE_DOCS
}
#endif

#endif
