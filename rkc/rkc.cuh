#ifndef RKC_HEAD
#define RKC_HEAD

#include "header.cuh"
#include "rkc_props.cuh"
#include "solver_options.cuh"

//__device__ Real rkc_spec_rad (const Real, const Real, const Real*, const Real*, Real*, Real*);
__device__ Real rkc_spec_rad (const Real, const Real, const Real, const Real*, const Real*, Real*, Real*);
__device__ void rkc_step (const Real, const Real, const Real, const Real*, const Real*, const int, Real*);
__device__ void rkc_driver (Real, const Real, const Real, int, Real*, Real*);

#endif