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
	srand((unsigned) time(NULL));

	double magnitude = 1e6;

	double r_max = 0;
	for (int size = 1; size <= 10; size++)
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
			passed &= (cabs(mtx[i] - mtx2[i]) / cabs(mtx[i])) < 1e-10;
			if (!passed)
			{
				double temp = cabs(mtx[i] - mtx2[i]) / cabs(mtx[i]);
				if (temp > r_max)
				{
					r_max = temp * 100.0;
					printf("failed\t%e\%\n",r_max);
				}
			}
		}

		free(mtx);
		free(mtx2);
	}
	return passed;
}

int main()
{
	//LU TESTS
	bool passed = true;
	passed &= LUTests();
	passed &= InverseTests();
	passed &= speedTest();

	printf("%s\n", passed ? "true" : "false");

	return 0;
}

#endif