/** A file designed to initialize the initial conditions for the integrators
 * \file mechanism.c
 *
 * \author Nicholas Curtis
 * \date 08/25/14
 *
 * Contains get_same_initial_conditions, customized for the mechanism at hand
 */

 //need to include this here to avoid circular dependencies
 #include "header.h"
 #include "mass_mole.h"

 void set_same_initial_conditions(int NUM, double* y_host, double* pres_host, double* rho_host)
 {
 	// load same ICs for all threads


 	//////////////////////////////////////////////////
	// species indices:
	// 
	// 13 - CH4
	//  3 -  O2
	// 47 -  N2
	//
	/////////////////////////////////////////////////

	// initial mole fractions
	Real Xi[NSP];
	for (int j = 0; j < NSP; ++ j) {
		Xi[j] = 0.0;
	}

	//
	// set initial mole fractions here
	//
	
	// CH4
	Xi[13] = 1.0;
	// o2
	Xi[3] = 2.0;
	// n2
	Xi[47] = 7.52;
	
	// normalize mole fractions to sum to 1
	Real Xsum = 0.0;
	for (int j = 0; j < NSP; ++ j) {
		Xsum += Xi[j];
	}
	for (int j = 0; j < NSP; ++ j) {
		Xi[j] /= Xsum;
	}

	// initial mass fractions
	Real Yi[NSP];
	mole2mass ( Xi, Yi );

	// set initial pressure, units [dyn/cm^2]
	// 1 atm = 1.01325e6 dyn/cm^2
	Real pres = 1.01325e6;

	// set initial temperature, units [K]
	Real T0 = 1600.0;

	// load temperature and mass fractions for all threads (cells)
	for (int i = 0; i < NUM; ++i) {
		y_host[i] = T0;

		// loop through species
		for (int j = 1; j < NN; ++j) {
			y_host[i + NUM * j] = Yi[j - 1];
		}
	}

	#ifdef CONV
	// if constant volume, calculate density
	Real rho;
  	rho = getDensity (T0, pres, Xi);
	#endif

	for (int i = 0; i < NUM; ++i) {
		#ifdef CONV
		// density
		rho_host[i] = rho;
		#else
		// pressure
		pres_host[i] = pres;
		#endif
	}
 }