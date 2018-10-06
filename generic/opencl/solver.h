#ifndef SOLVER_CLH
#define SOLVER_CLH

    #include "cl_macros.h"

    /* OpenCL compatibility Macros */

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

    #if (__ValueSize > 1)
      #define FUNC_SIZE(__a) PASTE( __a, PASTE(__, __ValueSize) )
      #define FUNC_TYPE(__a) PASTE( __a, PASTE(__, __ValueType) )
    #else
      #define FUNC_SIZE(__a) (__a)
      #define FUNC_TYPE(__a) (__a)
    #endif

    #ifdef VERBOSE
      #pragma message "__ValueSize  = " STRINGIFY(__ValueSize)
      #pragma message "__ValueType  = " STRINGIFY(__ValueType)
      #pragma message "__MaskType   = " STRINGIFY(__MaskType)
      //#pragma message "FUNC_TYPE(func)   = " FUNC_TYPE("func")
    #endif

    //! \brief Macro to determine offsets for pointer unpacking
    #define __getOffset1D(dim0) (dim0 * get_global_size(0))

    #ifndef __order
    #define __order 'C'
    #endif
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

      Hence, a 1-D accessor looks like:
      */

      #ifndef __ValueSize
        //! \brief Indexer macro for C-ordered implict-SIMD
        #define __getIndex1D(dim0, idx) (get_group_id(0) * dim0 * __arrayStride + idx * __arrayStride + get_local_id(0))
      #else
        // \brief Indexer macro for C-ordered explicit-SIMD
        // using explicit SIMD -> don't need to refer to local index (identically 0)
        #define __getIndex1D(dim0, idx) (get_group_id(0) * dim0 + idx)
      #endif
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

      #ifndef __ValueSize
        //! \brief Indexer macro for F-ordered implict-SIMD
        #define __getIndex1D(dim0, idx) (get_local_id(0) + __arrayStride * get_group_id(0) +
                                         idx * get_num_groups(0) * __arrayStride)
      #else
        // using explicit SIMD -> don't need to refer to local index (identically 0)
        #define __getIndex1D(dim0, idx) (get_group_id(0) + idx * get_num_groups(0))
      #endif

    #else
    #pragma error "Order " STRINGIFY(__order) " not recognized"
    #endif

    // ensure we have the number of equations and

    /* shared utility functions */

    inline __ValueType FUNC_TYPE(__fast_powu) (__ValueType p, unsigned q)
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
    inline __ValueType FUNC_TYPE(__fast_powi) (__ValueType p, int q)
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
       if      (q > 0) return FUNC_TYPE(__fast_powu)(p,q);
       else if (q < 0) return FUNC_TYPE(__fast_powu)(1.0/p,(unsigned int)(-q));
       else            return 1.0;
    }

    //inline double pow(const double &a, const double &b) { return std::pow(a,b); }
    inline __ValueType FUNC_TYPE(__powi)(const __ValueType a, const int b) { return FUNC_TYPE(__fast_powi)(a,b); }
    inline __ValueType FUNC_TYPE(__powu)(const __ValueType a, const unsigned int b) { return FUNC_TYPE(__fast_powu)(a,b); }

    inline __ValueType FUNC_TYPE(__sqr) (const __ValueType p) { return (p*p); }

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


    /* forward declare of RHS and Jacobian functions */

    /**
     * \brief The expected signature of the source term function of the IVP
     * \param[in]        t         The current system time
     * \param[in]        param     The system parameter
     * \param[in]        y         The state vector
     * \param[out]       dy        The output RHS (dydt) vector
     * \param[in]        rwk       The working buffer for source rate evaluation
     */
    void dydt (__global const __ValueType t, __global const __ValueType* param,
               __global const __ValueType* __restrict__ y, __global __ValueType* __restrict__ dy,
               __global __ValueType* rwk);

    /**
     * \brief The expected signature of Jacobian function of the IVP
     *
     * \param[in]           t               The current system time
     * \param[in]           param           The van der Pol parameter
     * \param[in]           y               The state vector at time t
     * \param[out]          jac             The jacobian to populate
     * \param[in]           rwk             The working buffer for Jacobian evaluation
     */
    void eval_jacob(__global const __ValueType t, __global const __ValueType* param,
                    __global const __ValueType* __restrict__ y, __global __ValueType* __restrict__ jac,
                    __global __ValueType* rwk);

#endif
