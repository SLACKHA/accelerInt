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

bool speedTest()
{
	bool passed = true;
	return passed;
}

int main()
{
	//LU TESTS
	bool passed = true;
	passed &= LUTests(true);
	passed &= InverseTests(true);

	printf("%s\n", passed ? "true" : "false");

	return 0;
}

#endif