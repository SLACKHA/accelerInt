#ifndef __cl_macros_h
#define __cl_macros_h

#ifdef __OPENCL_VERSION__
  #define __ckdata_attr __global
#else
  /* Define these as empty for non-OpenCL builds ... */
  #define __global
  #define __constant
  #define __local
  #define __kernel
  #define __private
  #define __ckdata_attr
#endif

#ifndef __blockSize
  //#warning 'setting __blockSize = 1'
  #define __blockSize (1)
#endif

#ifndef __arrayStride
  #define __arrayStride (__blockSize)
#endif

#if (__arrayStride == 1)
  #define __getIndex(__idx) (__idx)
#else
  #define __getIndex(__idx) ((__idx) * __arrayStride)
#endif

#define __inline inline

#if 1
  #define __tmpstr2(__x__) #__x__
  #define __tmpstr1(__x__) __tmpstr2(__x__)
  #pragma message "__blockSize   = " __tmpstr1(__blockSize)
  #pragma message "__getIndex    = " __tmpstr1(__getIndex(1))
  #pragma message "__arrayStride = " __tmpstr1(__arrayStride)
  #pragma message "__ckdata_attr = " __tmpstr1(__ckdata_attr)
  //#pragma message "__ValueType   = " __tmpstr1(__ValueType)
  //#pragma message "__MaskType    = " __tmpstr1(__MaskType)
  #ifdef __Alignment
    #pragma message "__Alignment   = " __tmpstr1(__Alignment)
  #endif

  #undef __tmpstr1
  #undef __tmpstr2
#endif

#endif
