#ifndef COMPLEX_INVERSE_H
#define COMPLEX_INVERSE_H

void getComplexInverse (int, double complex*);
void getComplexInverseHessenberg (const int, double complex*);

//#ifdef COMPILE_TESTING_METHODS
	int getHessenbergLU_test(const int, double complex*, int*);
	int getComplexLU_test (const int, double complex*, int*);
//#endif

#endif