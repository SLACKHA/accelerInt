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

//NOTE: to get tests to work correctly you have to uncomment the

bool LUTests()
{
	bool passed = true;
	//test hessian inverse
	double complex* testMatrix = (double complex*)calloc(9, sizeof(double complex));
	testMatrix[0] = 1;
	testMatrix[1] = 4;
	testMatrix[2] = 0;
	testMatrix[3] = 2;
	testMatrix[4] = 5;
	testMatrix[5] = 8;
	testMatrix[6] = 3;
	testMatrix[7] = 6;
	testMatrix[8] = 9;
	int * ipiv = (int*)calloc(3, sizeof(int));
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

	getHessenbergLU_test (3, 3, testMatrix, ipiv);
	for (int i = 0; i < 9; i++)
	{
		if (i && i % 3 == 0)
			printf("\n");
		printf("%f + %f * I\t\t", creal(testMatrix[i]), cimag(testMatrix[i]));
	}
	printf("\n\n");

	testMatrix[0] = 1 + 1e3 * I;
	testMatrix[1] = 4 - 0.001 * I;
	testMatrix[2] = 0;
	testMatrix[3] = 16e5;
	testMatrix[4] = 0.0001 * I;
	testMatrix[5] = 80000 + 5 * I;
	testMatrix[6] = 3 - 5 * I;
	testMatrix[7] = 9 * I;
	testMatrix[8] = 9 * I;

	getComplexLU_test (3, testMatrix, ipiv);
	for (int i = 0; i < 9; i++)
	{
		if (i && i % 3 == 0)
			printf("\n");
		printf("%f + %f * I\t\t", creal(testMatrix[i]), cimag(testMatrix[i]));
	}
	printf("\n\n");

	free(testMatrix);
	free(ipiv);

	return passed;
}

bool InverseTests()
{
	bool passed = true;
	//test hessian inverse
	double complex* testMatrix = (double complex*)calloc(9, sizeof(double complex));
	double complex* testMatrix2 = (double complex*)calloc(9, sizeof(double complex));
	double complex* testMatrix3 = (double complex*)calloc(9, sizeof(double complex));
	testMatrix[0] = 1;
	testMatrix[1] = 4;
	testMatrix[2] = 0;
	testMatrix[3] = 2;
	testMatrix[4] = 5;
	testMatrix[5] = 8;
	testMatrix[6] = 3;
	testMatrix[7] = 6;
	testMatrix[8] = 9;
	for (int i = 0; i < 9; i++)
	{
		testMatrix2[i] = testMatrix[i];
	}
	getComplexInverseHessenberg (3, 3, testMatrix);
	//test inverse
	for (int row = 0; row < 3; row++)
	{
		for (int col = 0; col < 3; col++)
		{
			testMatrix3[col * 3 + row] = 0;
			for (int k = 0; k < 3; k++)
			{
				testMatrix3[col * 3 + row] += testMatrix2[k * 3 + row] * testMatrix[col * 3 + k];
			}
		}
	}
	passed &= testMatrix3[0] == 1;
	passed &= testMatrix3[1] == 0;
	passed &= testMatrix3[2] == 0;
	passed &= testMatrix3[3] == 0;
	passed &= testMatrix3[4] == 1;
	passed &= testMatrix3[5] == 0;
	passed &= testMatrix3[6] == 0;
	passed &= testMatrix3[7] == 0;
	passed &= testMatrix3[8] == 1;

	testMatrix[0] = 1 + 1e3 * I;
	testMatrix[1] = 4 - 0.001 * I;
	testMatrix[2] = 0;
	testMatrix[3] = 16e5;
	testMatrix[4] = 0.0001 * I;
	testMatrix[5] = 80000 + 5 * I;
	testMatrix[6] = 3 - 5 * I;
	testMatrix[7] = 9 * I;
	testMatrix[8] = 9 * I;
	for (int i = 0; i < 9; i++)
	{
		testMatrix2[i] = testMatrix[i];
	}
	getComplexInverseHessenberg (3, 3, testMatrix);
	//test inverse
	for (int row = 0; row < 3; row++)
	{
		for (int col = 0; col < 3; col++)
		{
			testMatrix3[col * 3 + row] = 0;
			for (int k = 0; k < 3; k++)
			{
				testMatrix3[col * 3 + row] += testMatrix2[k * 3 + row] * testMatrix[col * 3 + k];
			}
		}
	}
	passed &= testMatrix3[0] == 1;
	passed &= testMatrix3[1] == 0;
	passed &= testMatrix3[2] == 0;
	passed &= testMatrix3[3] == 0;
	passed &= testMatrix3[4] == 1;
	passed &= testMatrix3[5] == 0;
	passed &= testMatrix3[6] == 0;
	passed &= testMatrix3[7] == 0;
	passed &= testMatrix3[8] == 1;

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