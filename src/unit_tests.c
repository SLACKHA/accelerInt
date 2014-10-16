#ifndef UNITTESTS_H
#define UNITTESTS_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <complex.h>
#include "complexInverse.h"
#include "complexInverseHessenberg.h"
#include <string.h>
#include <stdbool.h>
#include "header.h"
#include "timer.h"
#include "phiA.h"
#include "phiAHessenberg.h"
#include "cf.h"
#include "krylov.h"

bool LUTests()
{
	bool passed = true;
	//test hessian inverse
	double complex* testMatrix = (double complex*)calloc(9, sizeof(double complex));
	double complex* testMatrix2 = (double complex*)calloc(9, sizeof(double complex));
	int * ipiv = (int*)calloc(3, sizeof(int));
	int * ipiv2 = (int*)calloc(3, sizeof(int));

	testMatrix[0] = 1;
	testMatrix[1] = 4;
	testMatrix[2] = 0;
	testMatrix[3] = 2;
	testMatrix[4] = 5;
	testMatrix[5] = 8;
	testMatrix[6] = 3;
	testMatrix[7] = 6;
	testMatrix[8] = 9;
	getHessenbergLU_test (3, 3, testMatrix, ipiv);

	passed &= testMatrix[0] == 4;
	passed &= testMatrix[1] == 0;
	passed &= testMatrix[2] == 1.0/4.0;
	passed &= testMatrix[3] == 5;
	passed &= testMatrix[4] == 8;
	passed &= testMatrix[5] == 3.0/32.0;
	passed &= testMatrix[6] == 6;
	passed &= testMatrix[7] == 9;
	passed &= testMatrix[8] == 21.0/32.0;

	testMatrix[0] = 1 + 1e3 * I;
	testMatrix[1] = 4 - 0.001 * I;
	testMatrix[2] = 0;
	testMatrix[3] = 16e5;
	testMatrix[4] = 0.0001 * I;
	testMatrix[5] = 80000 + 5 * I;
	testMatrix[6] = 3 - 5 * I;
	testMatrix[7] = 9 * I;
	testMatrix[8] = 9 * I;

	testMatrix2[0] = 1 + 1e3 * I;
	testMatrix2[1] = 4 - 0.001 * I;
	testMatrix2[2] = 0;
	testMatrix2[3] = 16e5;
	testMatrix2[4] = 0.0001 * I;
	testMatrix2[5] = 80000 + 5 * I;
	testMatrix2[6] = 3 - 5 * I;
	testMatrix2[7] = 9 * I;
	testMatrix2[8] = 9 * I;

	getHessenbergLU_test (3, 3, testMatrix, ipiv);
	getComplexLU_test (3, testMatrix2, ipiv2);
	for (int i = 0; i < 9; i ++)
	{
		passed &= cabs(testMatrix[i] - testMatrix2[i]) < ATOL;
	}


	free(testMatrix);

	testMatrix = (double complex*)calloc(16, sizeof(double complex));
	testMatrix[0] = 1 + 1e3 * I;
	testMatrix[1] = 4 - 0.001 * I;
	testMatrix[2] = 0;
	testMatrix[4] = 16e5;
	testMatrix[5] = 0.0001 * I;
	testMatrix[6] = 80000 + 5 * I;
	testMatrix[8] = 3 - 5 * I;
	testMatrix[9] = 9 * I;
	testMatrix[10] = 9 * I;
	getHessenbergLU_test (3, 4, testMatrix, ipiv);
	int count = 0;
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			passed &= cabs(testMatrix[i * 4 + j] - testMatrix2[count++]) < ATOL;
		}
	}

	free(testMatrix2);
	free(ipiv);
	free(ipiv2);

	return passed;
}

bool InverseTests()
{
	bool passed = true;
	//test hessian inverse
	double complex* testMatrix = (double complex*)calloc(9, sizeof(double complex));
	double complex* testMatrix2 = (double complex*)calloc(9, sizeof(double complex));
	
	testMatrix[0] = 1 + 1e3 * I;
	testMatrix[1] = 4 - 0.001 * I;
	testMatrix[2] = 0;
	testMatrix[3] = 16e5;
	testMatrix[4] = 0.0001 * I;
	testMatrix[5] = 80000 + 5 * I;
	testMatrix[6] = 3 - 5 * I;
	testMatrix[7] = 9 * I;
	testMatrix[8] = 9 * I;
	memcpy(testMatrix2, testMatrix, 9 * sizeof(complex double));
	getComplexInverseHessenberg (3, 3, testMatrix);
	getComplexInverse (3, testMatrix2);
	//test inverse
	for (int i = 0; i < 9; i ++)
	{
		passed &= cabs(testMatrix[i] - testMatrix2[i]) < ATOL;
	}

	free(testMatrix);
	
	testMatrix = (double complex*)calloc(16, sizeof(double complex));
	testMatrix[0] = 1 + 1e3 * I;
	testMatrix[1] = 4 - 0.001 * I;
	testMatrix[2] = 0;
	testMatrix[4] = 16e5;
	testMatrix[5] = 0.0001 * I;
	testMatrix[6] = 80000 + 5 * I;
	testMatrix[8] = 3 - 5 * I;
	testMatrix[9] = 9 * I;
	testMatrix[10] = 9 * I;
	getComplexInverseHessenberg (3, 4, testMatrix);
	//test inverse
	int count = 0;
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			passed &= cabs(testMatrix[i * 4 + j] - testMatrix2[count++]) < ATOL;
		}
	}
	
	free(testMatrix);
	free(testMatrix2);

	return passed;
}

static inline double getSign()
{
	return ((double)rand()/(double)RAND_MAX) > 0.5 ? 1.0 : -1.0;
}

static inline double getRand()
{
	return ((double)rand()/(double)RAND_MAX) * getSign();
}

bool speedTest()
{
	bool passed = true;

	double magnitude = 1e6;

	double r_max = 0;
	for (int size = 1; size <= 11; size++)
	{
		//create matricies
		int dim = pow(2, size);
		int actual_size = dim * dim;
		double complex* mtx = (double complex*)calloc(actual_size, sizeof(double complex));
		double complex* mtx2 = (double complex*)calloc(actual_size, sizeof(double complex));

		for (int col = 0; col < dim; col++)
		{
			for (int row = 0; row <= col + 1; row ++)
			{
				if (row + col * dim == actual_size)
					break;
				mtx[row + col * dim] = magnitude * getRand() + magnitude * getRand() * I;
			}
		}
		memcpy(mtx2, mtx, actual_size * sizeof(double complex));

		//time
		StartTimer();
		getComplexInverseHessenberg(dim, dim, mtx);
		double t_h = GetTimer();

		StartTimer();
		getComplexInverse(dim, mtx2);
		double t_r = GetTimer();

		printf ("%d\t%f\t%f\n", dim, t_h, t_r);

		for (int i = 0; i < actual_size; i ++)
		{
			double temp = 0;
			if (cabs(mtx2[i]) > 0)
			{
				temp = cabs(mtx[i] - mtx2[i]) / cabs(mtx2[i]);
			}
			else
			{
				temp = cabs(mtx[i] - mtx2[i]);
			}
			passed &= temp < 1e-08;
			if (!passed)
			{
				if (temp * 100.0 > r_max)
				{
					r_max = temp * 100.0;
					printf("failed\t%e%%\n", temp);
				}
			}
		}

		free(mtx);
		free(mtx2);
	}
	return passed;
}

Real complex poles[N_RA];
Real complex res[N_RA];

bool PhiTests()
{
	// get poles and residues for rational approximant to matrix exponential
	double *poles_r = (double*) calloc (N_RA, sizeof(double));
	double *poles_i = (double*) calloc (N_RA, sizeof(double));
	double *res_r = (double*) calloc (N_RA, sizeof(double));
	double *res_i = (double*) calloc (N_RA, sizeof(double));
	
	cf ( N_RA, poles_r, poles_i, res_r, res_i );
	
	for (int i = 0; i < N_RA; ++i) {
		poles[i] = poles_r[i] + poles_i[i] * _Complex_I;
		res[i] = res_r[i] + res_i[i] * _Complex_I;
	}
	
	// free memory
	free (poles_r);
	free (poles_i);
	free (res_r);
	free (res_i);


	bool passed = true;

	double magnitude = 1e6;

	//create matricies
	int dim = NN;
	int actual_size = dim * dim;
	double* mtx = (double*)calloc(actual_size, sizeof(double));
	double* mtx2 = (double*)calloc(actual_size, sizeof(double));
	double* mtx3 = (double*)calloc(actual_size, sizeof(double));

	for (int col = 0; col < dim; col++)
	{
		for (int row = 0; row <= col + 1; row ++)
		{
			if (row + col * dim == actual_size)
				break;
			mtx[row + col * dim] = magnitude * getRand();
		}
	}
	memcpy(mtx2, mtx, actual_size * sizeof(double));

	//time
	StartTimer();
	phiAc_variable(dim, dim, mtx, 1e-8 / 3.0, mtx2);
	double t_h = GetTimer();

	StartTimer();
	phiAc(mtx, 1e-8 / 3.0, mtx3);
	double t_r = GetTimer();

	printf ("%d\t%f\t%f\n", dim, t_h, t_r);

	for (int i = 0; i < actual_size; i ++)
	{
		passed &= (cabs(mtx2[i] - mtx3[i]) / cabs(mtx3[i])) < 1e-10;
	}

	int new_dim = dim * 2;
	double* mtx4 = (double*)calloc(new_dim * new_dim, sizeof(double));
	//copy over
	for (int i = 0; i < dim; i++)
	{
		for (int j = 0; j < dim; j++)
		{
			mtx4[i * new_dim + j] = mtx[i * dim + j];
		}
	}
	free(mtx2);
	mtx2 = (double*)calloc(new_dim * new_dim, sizeof(double));
	phiAc_variable(dim, new_dim, mtx4, 1e-8 / 3.0, mtx2);
	for (int i = 0; i < dim; i++)
	{
		for (int j = 0; j < dim; j++)
		{
			passed &= (cabs(mtx2[i * new_dim + j] - mtx3[i * dim + j]) / cabs(mtx3[i * dim + j])) < 1e-10;
		}
	}

	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			if (j <= i + 1)
			{
				mtx[i * 3 + j] = j * 3 + i + 1;
			}
			else
				mtx[i * 3 + j] = 0;
		}
	}
	expAc_variable(3, 3, mtx, 1.0, mtx2);

	/* 1.365853949602854e+05     4.744010693488505e+05     5.525249852197236e+05
	 3.124956634834923e+05     1.085387533657987e+06     1.264126589243000e+06
	 4.242043168094727e+05     1.473399960585930e+06     1.716036434748584e+06 */

	passed &= abs(mtx2[0] - 1.365853949602854e+05) < ATOL;
	passed &= abs(mtx2[1] - 3.124956634834923e+05) < ATOL;
	passed &= abs(mtx2[2] - 4.242043168094727e+05) < ATOL;
	passed &= abs(mtx2[3] - 4.744010693488505e+05) < ATOL;
	passed &= abs(mtx2[4] - 1.085387533657987e+06) < ATOL;
	passed &= abs(mtx2[5] - 1.473399960585930e+06) < ATOL;
	passed &= abs(mtx2[6] - 5.525249852197236e+05) < ATOL;
	passed &= abs(mtx2[7] - 1.264126589243000e+06) < ATOL;
	passed &= abs(mtx2[8] - 1.716036434748584e+06) < ATOL;

	free(mtx);
	free(mtx2);
	free(mtx3);
	free(mtx4);
	return passed;
}
#define STRIDE_MIRROR (NN + 2)
bool LinearAlgebraTests()
{
	#ifdef COMPILE_TESTING_METHODS
	//create and populate some matricies
	Real A[STRIDE_MIRROR * STRIDE_MIRROR] = {ZERO};
	Real v[STRIDE_MIRROR] = {ZERO};
	Real out[STRIDE_MIRROR] = {ZERO};
	int m_size = STRIDE_MIRROR / 2;
	for (int col = 0; col < m_size; ++col)
	{
		for (int row = 0; row < NN; ++row)
		{
			A[row + col * STRIDE_MIRROR] = col + row * m_size;
		}
	}
	for (int row = 0; row < m_size; ++row)
	{
		v[row] = row;
	}

	//MxM matrix * Mx1 vector
	matvec_m_by_m_test(m_size, A, v, out);
	for (int row = 0; row < m_size; row++)
	{
		int val = (m_size - 1) * m_size * ((3 * row + 2) * m_size - 1) / 6;
		if (out[row] != val)
			return false;
	}

	//NNxM matrix * Mx1 vector
	matvec_n_by_m_test(m_size, A, v, out);
	for (int row = 0; row < NN; row++)
	{
		int val = (m_size - 1) * m_size * ((3 * row + 2) * m_size - 1) / 6;
		if (out[row] != val)
			return false;
	}

	#else
	printf ("Please compile with COMPILE_TESTING_METHODS defined (check header.h)\n");
	return false;
	#endif
	return true;
}

int main()
{
	srand((unsigned) time(NULL));
	bool passed = true;
	passed &= LinearAlgebraTests();
	passed &= LUTests();
	passed &= InverseTests();
	passed &= PhiTests();
	passed &= speedTest();

	printf("%s\n", passed ? "true" : "false");

	return 0;
}

#endif