#ifndef ERROR_CODES_H
#define ERROR_CODES_H

#ifdef (__OPENCL_VERSION__)
// todo... this isn't guarenteed.
#define CL_SUCCESS          (0)
#endif
#define CL_TOO_MUCH_WORK    (-1)
#define CL_TDIST_TOO_SMALL  (-2)

#endif
