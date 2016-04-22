/*
 * \file rk78_typedefs.cpp
 *
 * \author Nicholas J. Curtis
 * \date 04/29/2016
 *
 * Defines an interface for boost's runge_kutta_fehlberg78 solver
 * 
*/

#ifndef RK78_TYPEDEFS_HPP
#define RK78_TYPEDEFS_HPP

#include <boost/numeric/odeint.hpp>
using namespace boost::numeric::odeint;

//our code
extern "C" {
	#include "dydt.h"
	#include "solver_options.h"
}

//state vector
typedef std::vector< double > state_type;

//solver type
typedef runge_kutta_fehlberg78< state_type , double > stepper;

//controller type
typedef controlled_runge_kutta< stepper > controller;

//stiffness measure for project
//do a binary search to find the maximum available stepsize
#ifdef LOG_OUTPUT
//#define STIFFNESS_MEASURE
#endif
#ifdef STIFFNESS_MEASURE
#include <boost/numeric/odeint/stepper/controlled_step_result.hpp>
#endif

/* A wrapper class to evaluate the rhs function y' = f(y) 
   stores the state variable, and provides to dydt
*/
class rhs_eval {
	double m_statevar;
public:
	rhs_eval() {
		this->m_statevar = -1;
	}

	void set_state_var(const double state_var)
	{
		this->m_statevar = state_var;
	}

	//wrapper for the pyJac RHS fn
	void operator() (const state_type &y , state_type &fy , const double t) const
	{
		dydt(t, this->m_statevar, &y[0], &fy[0]);
	}
};

#endif