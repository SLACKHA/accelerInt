#ifndef SOLVER_CLH
#define SOLVER_CLH

/* OpenCL compatibility Macros */

#ifndef __blockSize
  #define __blockSize (1)
#endif

#ifndef __arrayStride
  #define __arrayStride (__blockSize)
#endif

#define __STRINGIFY(__x) #__x
#define STRINGIFY(__x) __STRINGIFY(__x)

#define __inline inline

#define VERBOSE
#ifdef VERBOSE
  #pragma message "__blockSize   = " STRINGIFY(__blockSize)
  #pragma message "__getIndex    = " STRINGIFY(__getIndex(1))
  #pragma message "__arrayStride = " STRINGIFY(__arrayStride)
  #ifdef __Alignment
    #pragma message "__Alignment   = " STRINGIFY(__Alignment)
  #endif
  #pragma message "__EnableQueue = " STRINGIFY(__EnableQueue)
#endif


#define __PASTE(a,b) a ## b
#define PASTE(a,b) __PASTE(a,b)

#ifndef __ValueSize
  #define __ValueSize 1
#endif

#if (__ValueSize > 1)// && defined(__OPENCL_VERSION__)
  #define __ValueType PASTE(double, __ValueSize)
  #define __IntType PASTE(int, __ValueSize)
  #define __MaskType PASTE(long, __ValueSize)
#else
  #define __ValueType double
  #define __IntType int
  #define __MaskType int
#endif

#ifndef __order
#define __order 'C'
#endif

#ifdef VERBOSE
  #pragma message "__ValueSize  = " STRINGIFY(__ValueSize)
  #pragma message "__ValueType  = " STRINGIFY(__ValueType)
  #pragma message "__MaskType   = " STRINGIFY(__MaskType)
  #pragma message "__order      = " STRINGIFY(__order)
  //#pragma message "FUNC_TYPE(func)   = " FUNC_TYPE("func")
#endif

//! \brief Macro to determine offsets for pointer unpacking
#define __getOffset1D(dim0) ((dim0) * get_global_size(0))

#if (__order == 'C')
  /*!
    Use row-major ordering -- this results in (1-D, neq-sized) arrays that are shaped as:

      > (get_num_groups(0), neq, __arrayStride) - for GPU (implicit-SIMD)
      with indexing strides:
      > (neq * __arrayStride, __arrayStride, 1)

      or

      > (get_num_groups(0), neq, __ValueSize) - for CPU / accelerator (explicit-SIMD)
      with indexing strides:
      > (neq, 1)

  */

  #if (__ValueSize == 1)
    //! \brief Indexer macro for C-ordered implict-SIMD
    #define __getIndex1D(dim0, idx) (get_group_id(0) * (dim0) * __arrayStride + (idx) * __arrayStride + get_local_id(0))
  #else
    // \brief Indexer macro for C-ordered explicit-SIMD
    // using explicit SIMD -> don't need to refer to local index (identically 0)
    #define __getIndex1D(dim0, idx) (get_group_id(0) * (dim0) + (idx))
  #endif

  /*! \brief Row-major indexing macro for 2D working buffer arrays

      The array is shaped as (get_num_groups(0), dim0, dim1, vec_size), where vec_size is either
      __arrayStride or __valueSize as described above.

    */
  #define __getIndex2D(dim0, dim1, row, col) (__getIndex1D((dim0), (row) * (dim1)  + (col)))
  /*! \brief Row-major indexing macro for global arrays

      The array is shaped as (nprob, dim0), while the index arguements correspond to:
      \param[in]    pid     the global problem index
      \param[in]    idx     the index inside the array for problem index pid
   */
  #define __globalIndex1D(nprob, dim0, pid, idx) ((dim0) * (pid) + (idx))

#elif (__order == 'F')
  /*
    Use column-major ordering -- this results in (1-D, neq-sized) arrays that are shaped as:

      > (__arrayStride, get_num_groups(0), neq) - for GPU
      with indexing strides:
      > (1, __arrayStride, get_num_groups(0) * __arrayStride)

      or

      > (__ValueSize, get_num_groups(0), neq) - for CPU / accelerator (explicit-SIMD)
      with indexing strides:
      > (1, get_num_groups(0))

  Hence, a 1-D accessor looks like:
  */

  #if (__ValueSize == 1)
    //! \brief Indexer macro for F-ordered implict-SIMD
    #define __getIndex1D(dim0, idx) (get_local_id(0) + __arrayStride * get_group_id(0) + (idx) * get_num_groups(0) * __arrayStride)
  #else
    // using explicit SIMD -> don't need to refer to local index (identically 0)
    #define __getIndex1D(dim0, idx) (get_group_id(0) + (idx) * get_num_groups(0))
  #endif

  /*! \brief Column-major indexing macro for 2D working buffer arrays

      The array is shaped as (get_num_groups(0), dim0, dim1, vec_size), where vec_size is either
      __arrayStride or __valueSize as described above.

    */
  #define __getIndex2D(dim0, dim1, row, col) (__getIndex1D((dim1), (col) * (dim0)  + (row)))

  /*! \brief Column-major indexing macro for global arrays

      The array is shaped as (nprob, dim0), while the index arguements correspond to:
      \param[in]    pid     the global problem index
      \param[in]    idx     the index inside the array for problem index pid
   */
  #define __globalIndex1D(nprob, dim0, pid, idx) ((nprob) * (idx) + (pid))

#else
#pragma error "Order " STRINGIFY(__order) " not recognized"
#endif

//! \brief unused parameter macro
#define UNUSED(x) (void)(x)

// ensure we have the number of equations and

/* shared utility functions */

inline __ValueType __fast_powu(__ValueType p, unsigned q)
{
   if      (q == 0) return 1.0;
   else if (q == 1) return p;
   else if (q == 2) return p*p;
   else if (q == 3) return p*p*p;
   else if (q == 4) return p*p*p*p;
   else
   {
      // q^p -> (q^(p/2))^2 ... recursively takes log(q) ops
      __ValueType r = 1;
      while (q)
      {
         if (q % 2) //is_odd(q)) // odd power ...
         {
            r *= p;
            --q;
         }
         else
         {
            p *= p; // square the base ...
            q /= 2; // drop the power by two ...
         }
      }
      return r;
   }
}
// p^q where q is an integral
inline __ValueType __fast_powi(__ValueType p, int q)
{
#if (__ValueSize == 1)
   if (p == 0.0)
   {
      if (q == 0)
         return 1.0;
      //else if (q < 0)
      //   return std::numeric_limits<double>::infinity();
      else
         return 0.0;
   }
#endif
   if      (q > 0) return __fast_powu(p,q);
   else if (q < 0) return __fast_powu(1.0/p,(unsigned int)(-q));
   else            return 1.0;
}

//inline double pow(const double &a, const double &b) { return std::pow(a,b); }
inline __ValueType __powi(const __ValueType a, const int b) { return __fast_powi(a,b); }
inline __ValueType __powu(const __ValueType a, const unsigned int b) { return __fast_powu(a,b); }

inline __ValueType __sqr(const __ValueType p) { return (p*p); }

#ifndef __any
  #if (__ValueSize == 1)
    #define __any(__val) (__val)
  #else
    #define __any(__val) (any(__val))
  #endif
#endif
#ifndef __all
  #if (__ValueSize == 1)
    #define __all(__val) (__val)
  #else
    #define __all(__val) (all(__val))
  #endif
#endif
#ifndef __select
  #if (__ValueSize == 1)
    #define __select(__is_false, __is_true, __cmp) ( (__cmp) ? (__is_true) : (__is_false) )
  #else
    #define __select(__is_false, __is_true, __cmp) (select((__is_false), (__is_true), (__cmp)))
  #endif
#endif
#ifndef __not
  #define __not(__val) ( !(__val) )
#endif

// Scalar helper functions for the pivot operation -- need a Vector version here.
#if (__ValueSize == 1)
  #define __read_from(__src, __lane, __dest) { (__dest) = (__src); }
  #define __write_to(__src, __lane, __dest) { (__dest) = (__src); }
#elif (__ValueSize == 2)
  #define __read_from(__src, __lane, __dest) \
  { \
     if      ((__lane) ==  0) (__dest) = (__src).s0; \
     else if ((__lane) ==  1) (__dest) = (__src).s1; \
  }
  #define __write_to(__src, __lane, __dest) \
  { \
     if      ((__lane) ==  0) (__dest).s0 = (__src); \
     else if ((__lane) ==  1) (__dest).s1 = (__src); \
  }
#elif (__ValueSize == 3)
  #define __read_from(__src, __lane, __dest) \
  { \
     if      ((__lane) ==  0) (__dest) = (__src).s0; \
     else if ((__lane) ==  1) (__dest) = (__src).s1; \
     else if ((__lane) ==  2) (__dest) = (__src).s2; \
  }
  #define __write_to(__src, __lane, __dest) \
  { \
     if      ((__lane) ==  0) (__dest).s0 = (__src); \
     else if ((__lane) ==  1) (__dest).s1 = (__src); \
     else if ((__lane) ==  2) (__dest).s2 = (__src); \
  }
#elif (__ValueSize == 4)
  #define __read_from(__src, __lane, __dest) \
  { \
     if      ((__lane) ==  0) (__dest) = (__src).s0; \
     else if ((__lane) ==  1) (__dest) = (__src).s1; \
     else if ((__lane) ==  2) (__dest) = (__src).s2; \
     else if ((__lane) ==  3) (__dest) = (__src).s3; \
  }
  #define __write_to(__src, __lane, __dest) \
  { \
     if      ((__lane) ==  0) (__dest).s0 = (__src); \
     else if ((__lane) ==  1) (__dest).s1 = (__src); \
     else if ((__lane) ==  2) (__dest).s2 = (__src); \
     else if ((__lane) ==  3) (__dest).s3 = (__src); \
  }
#elif (__ValueSize == 8)
  #define __read_from(__src, __lane, __dest) \
  { \
     if      ((__lane) ==  0) (__dest) = (__src).s0; \
     else if ((__lane) ==  1) (__dest) = (__src).s1; \
     else if ((__lane) ==  2) (__dest) = (__src).s2; \
     else if ((__lane) ==  3) (__dest) = (__src).s3; \
     else if ((__lane) ==  4) (__dest) = (__src).s4; \
     else if ((__lane) ==  5) (__dest) = (__src).s5; \
     else if ((__lane) ==  6) (__dest) = (__src).s6; \
     else if ((__lane) ==  7) (__dest) = (__src).s7; \
  }
  #define __write_to(__src, __lane, __dest) \
  { \
     if      ((__lane) ==  0) (__dest).s0 = (__src); \
     else if ((__lane) ==  1) (__dest).s1 = (__src); \
     else if ((__lane) ==  2) (__dest).s2 = (__src); \
     else if ((__lane) ==  3) (__dest).s3 = (__src); \
     else if ((__lane) ==  4) (__dest).s4 = (__src); \
     else if ((__lane) ==  5) (__dest).s5 = (__src); \
     else if ((__lane) ==  6) (__dest).s6 = (__src); \
     else if ((__lane) ==  7) (__dest).s7 = (__src); \
  }
#elif (__ValueSize == 16)
  #define __read_from(__src, __lane, __dest) \
  { \
     if      ((__lane) ==  0) (__dest) = (__src).s0; \
     else if ((__lane) ==  1) (__dest) = (__src).s1; \
     else if ((__lane) ==  2) (__dest) = (__src).s2; \
     else if ((__lane) ==  3) (__dest) = (__src).s3; \
     else if ((__lane) ==  4) (__dest) = (__src).s4; \
     else if ((__lane) ==  5) (__dest) = (__src).s5; \
     else if ((__lane) ==  6) (__dest) = (__src).s6; \
     else if ((__lane) ==  7) (__dest) = (__src).s7; \
     else if ((__lane) ==  8) (__dest) = (__src).s8; \
     else if ((__lane) ==  9) (__dest) = (__src).s9; \
     else if ((__lane) == 10) (__dest) = (__src).sA; \
     else if ((__lane) == 11) (__dest) = (__src).sB; \
     else if ((__lane) == 12) (__dest) = (__src).sC; \
     else if ((__lane) == 13) (__dest) = (__src).sD; \
     else if ((__lane) == 14) (__dest) = (__src).sE; \
     else if ((__lane) == 15) (__dest) = (__src).sF; \
  }
  #define __write_to(__src, __lane, __dest) \
  { \
     if      ((__lane) ==  0) (__dest).s0 = (__src); \
     else if ((__lane) ==  1) (__dest).s1 = (__src); \
     else if ((__lane) ==  2) (__dest).s2 = (__src); \
     else if ((__lane) ==  3) (__dest).s3 = (__src); \
     else if ((__lane) ==  4) (__dest).s4 = (__src); \
     else if ((__lane) ==  5) (__dest).s5 = (__src); \
     else if ((__lane) ==  6) (__dest).s6 = (__src); \
     else if ((__lane) ==  7) (__dest).s7 = (__src); \
     else if ((__lane) ==  8) (__dest).s8 = (__src); \
     else if ((__lane) ==  9) (__dest).s9 = (__src); \
     else if ((__lane) == 10) (__dest).sA = (__src); \
     else if ((__lane) == 11) (__dest).sB = (__src); \
     else if ((__lane) == 12) (__dest).sC = (__src); \
     else if ((__lane) == 13) (__dest).sD = (__src); \
     else if ((__lane) == 14) (__dest).sE = (__src); \
     else if ((__lane) == 15) (__dest).sF = (__src); \
  }
#endif

#if (__ValueSize == 1)
  #define __vload(__offset, __ptr) ( *(__ptr) )
#else
  #define __vload(__offset, __ptr) ( PASTE(vload,__ValueSize)((__offset), (__ptr)) )
#endif

/**
 * \brief The expected signature of Jacobian function of the IVP
 *
 * \param[in]           t               The current system time
 * \param[in]           param           The van der Pol parameter
 * \param[in]           y               The state vector at time t
 * \param[out]          jac             The jacobian to populate
 * \param[in]           rwk             The working buffer for Jacobian evaluation
 */
void eval_jacob(__global const __ValueType* __restrict__ t, __global const __ValueType* __restrict__ param,
                __global const __ValueType* __restrict__ y, __global __ValueType* __restrict__ jac,
                __global __ValueType* __restrict__ rwk);

#endif
