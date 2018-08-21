/**
 * \file
 * \brief Error checking for the CPU RKC solver
 */
#include "rkc_props.h"

/*! \fn void check_error(int tid, int code)
    \brief Checks the return code of the given thread (IVP) for an error, and exits if found
    \param tid The thread (IVP) index
    \param code The return code of the thread
    @see ErrorCodes
 */
void check_error(int tid, int code)
{
    if (code != EC_success)
    {
        printf("During integration of ODE# %d, an unknown error occurred,"
                        "exiting...\n", tid);
                exit(code);
    }
}