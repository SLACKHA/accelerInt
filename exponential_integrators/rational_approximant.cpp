/**
* \file rational_approximant.c
* \brief The generic initialization file for poles/hosts for RA based evaulation of the matrix exponential
*
* \author Nicholas Curtis
* \date 03/09/2015
*
* Contains initialization and declaration of RA
*/

//cf
extern "C" {
    #include "cf.h"
}
#include "rational_approximant.hpp"

/**
* \brief get poles and residues for rational approximant to matrix exponential
*/
void find_poles_and_residuals(const int N_RA, std::vector<std::complex<double>>& poles, std::vector<std::complex<double>>& res)
{
	// get poles and residues for rational approximant to matrix exponential
    double *poles_r = (double *) calloc (N_RA, sizeof(double));
    double *poles_i = (double *) calloc (N_RA, sizeof(double));
    double *res_r = (double *) calloc (N_RA, sizeof(double));
    double *res_i = (double *) calloc (N_RA, sizeof(double));

    cf (N_RA, poles_r, poles_i, res_r, res_i);

    for (int i = 0; i < N_RA; ++i)
    {
        poles.emplace_back(std::complex<double>(poles_r[i], poles_i[i]));
        res.emplace_back(std::complex<double>(res_r[i], res_i[i]));
    }

    // free memory
    free (poles_r);
    free (poles_i);
    free (res_r);
    free (res_i);
}
