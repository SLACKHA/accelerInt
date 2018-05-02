/**
 *  \file
 *  \brief A number of definitions that control CUDA kernel launches
 */

#ifndef LAUNCH_BOUNDS_CUH
#define LAUNCH_BOUNDS_CUH

//! The target number of threads per block
#define TARGET_BLOCK_SIZE (64)
//! Shared memory per thread
#define SHARED_SIZE (0)
//! Large L1 cache active
#define PREFERL1

#endif