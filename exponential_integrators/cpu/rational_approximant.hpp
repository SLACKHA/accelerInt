/**
* \file rational_approximant.hpp
* \brief The generic initialization file for poles/hosts for RA based evaulation of the matrix exponential
*
* \author Nicholas Curtis
* \date 03/09/2015
*
* Contains declaration of RA
*/

#ifndef RATIONAL_APPROXIMANT_HPP
#define RATIONAL_APPROXIMANT_HPP

#include <vector>
#include <complex>
void find_poles_and_residuals(const int N_RA, const double,
                              std::vector<std::complex<double>>& poles,
                              std::vector<std::complex<double>>& res);

#endif
