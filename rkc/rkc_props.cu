/**
 * \file
 * \brief Error checking for the GPU RKC solver
 */

#include "rkc_props.cuh"

/*! \fn void check_error(int tid, int code)
    \brief Checks the return code of the given thread (IVP) for an error, and exits if found
    \param tid The thread (IVP) index
    \param code The return code of the thread
    @see ErrorCodes
 */
__host__
void check_error(int num_cond, int* codes)
{
    for (int tid = 0; tid < num_cond; ++tid)
    {
        int code = codes[tid];
        if (code != EC_success)
        {
            printf("During integration of ODE# %d, an unknown error occurred,"
                            "exiting...\n", tid);
                    exit(code);
        }
    }
}
