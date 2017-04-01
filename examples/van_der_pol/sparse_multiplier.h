/**
 * \file
 * \brief Header definition for Jacobian vector multiplier, used in exponential integrators
 *
 */

#ifndef SPARSE_HEAD
#define SPARSE_HEAD

#include "header.h"


#ifdef GENERATE_DOCS
//put this in the van der Pol namespace for documentation
namespace van_der_pol {
#endif

void sparse_multiplier (const double *, const double *, double*);


#ifdef GENERATE_DOCS
}
#endif

#endif
