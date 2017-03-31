/**
 * \file
 * \brief Sets same Initial Conditions (ICs) for all problems
 *
 * This provides simple definitions of the set_same_initial_conditions function
 * used when SAME_ICS is defined
 */

#include "header.h"

#ifdef GENERATE_DOCS
//put this in the van der Pol namespace for documentation
namespace van_der_pol {
#endif


/*!
 *
 * \brief Set same ICs for all problems
 * \param NUM       The number of initial value problems
 * \param y_host    The state vectors to initialize
 * \param var_host  The vector of \f$mu\f$ parameters for the van der Pol equation
 *
 */
void set_same_initial_conditions(int NUM, double** y_host, double** var_host)
{
    //init vectors
    (*y_host) = (double*)malloc(NUM * NSP * sizeof(double));
    (*var_host) = (double*)malloc(NUM * sizeof(double));
    //now set the values
    for (int i = 0; i < NUM; ++i){
        //set mu
        (*var_host)[i] = 500;
        //set y1, y2
        (*y_host)[i] = 2;
        (*y_host)[i + NUM] = 0;
    }
}

#ifdef GENERATE_DOCS
}
#endif