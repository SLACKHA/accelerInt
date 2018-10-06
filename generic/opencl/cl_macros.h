#ifndef __cl_macros_h
#define __cl_macros_h

#ifndef __OPENCL_VERSION__
  /* Define these as empty for non-OpenCL builds ... */
  #define __global
  #define __constant
  #define __local
  #define __kernel
  #define __private
#endif

#ifndef __blockSize
  //#warning 'setting __blockSize = 1'
  #define __blockSize (1)
#endif

#ifndef __arrayStride
  #define __arrayStride (__blockSize)
#endif

#define __STRINGIFY(__x) #__x
#define STRINGIFY(__x) __STRINGIFY(__x)

#define __inline inline

#define VERBOSE
#if VERBOSE
  #pragma message "__blockSize   = " STRINGIFY(__blockSize)
  #pragma message "__getIndex    = " STRINGIFY(__getIndex(1))
  #pragma message "__arrayStride = " STRINGIFY(__arrayStride)
  #ifdef __Alignment
    #pragma message "__Alignment   = " STRINGIFY(__Alignment)
  #endif
#endif

#endif
