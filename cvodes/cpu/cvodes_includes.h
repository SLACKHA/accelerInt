#ifndef CVODES_INCLUDES_HEAD
#define CVODES_INCLUDES_HEAD

/* CVODES INCLUDES */
#include "sundials/sundials_config.h"
#if SUNDIALS_VERSION_MAJOR >= 3
    #include "sunlinsol/sunlinsol_lapackdense.h"
    #include "cvodes/cvodes_direct.h"
#else
    #include "cvodes/cvodes_lapack.h"
#endif
#include "sundials/sundials_types.h"
#include "sundials/sundials_math.h"
#include "sundials/sundials_nvector.h"
#include "nvector/nvector_serial.h"
#include "cvodes/cvodes.h"
#include "cv_user_data.h"

#endif
