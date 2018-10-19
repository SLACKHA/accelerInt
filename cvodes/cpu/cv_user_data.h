#ifndef CV_USER_DATA_H
#define CV_USER_DATA_H

/*!  \brief a C-definition of a user-data struct that can be packaged to encapsulate
            both user specified parameters, as well as working memory for CVODEs
 */

#ifdef __cplusplus
extern "C" {
#endif


struct CVUserData
{
    double param;
    double* __restrict__ rwk;
};

#ifdef __cplusplus
}
#endif
#endif
