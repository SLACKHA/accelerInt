#include "header.cuh"
#include "solver_props.cuh"
#include <cuComplex.h>
///////////////////////////////////////////////////////////

__device__
int getComplexMax (const int n, const cuDoubleComplex *complexArr) {
	
	int maxInd = 0;
	if (n == 1)
		return maxInd;
	
	double maxVal = cuCabs(complexArr[0]);
	for (int i = 1; i < n; ++i) {
		if (cuCabs(complexArr[i]) > maxVal) {
			maxInd = i;
			maxVal = cuCabs(complexArr[i]);
		}
	}
	
	return maxInd;
}

///////////////////////////////////////////////////////////

__device__
void scaleComplex (const int n, const cuDoubleComplex val, cuDoubleComplex* arrX) {
	
	for (int i = 0; i < n; ++i) {
		arrX[i] = cuCmul(arrX[i], val);
	}
	
}

///////////////////////////////////////////////////////////

__device__
void swapComplex (const int n, cuDoubleComplex* arrX, const int incX, cuDoubleComplex* arrY, const int incY) {
	
	int ix = 0;
	int iy = 0;
	
	for (int i = 0; i < n; ++i) {
		cuDoubleComplex temp = arrX[ix];
		arrX[ix] = arrY[iy];
		arrY[iy] = temp;
		ix += incX;
		iy += incY;
	}
	
}

///////////////////////////////////////////////////////////

__device__
void complexGERU (const int n, const cuDoubleComplex alpha, const cuDoubleComplex* arrX,
									const cuDoubleComplex* arrY, const int incY, cuDoubleComplex* A, const int lda) {
	
	for (int j = 0; j < n; ++j) {
    	if (cuCabs(arrY[j * incY]) > 0.0) {
      
			cuDoubleComplex temp = cuCmul(alpha, arrY[j * incY]);
      
			for (int i = 0; i < n; ++i) {
				A[i + (lda * j)] = cuCfma(arrX[i], temp, A[i + (lda * j)]);
			}
      
		}    
	}
	
}

///////////////////////////////////////////////////////////

__device__
void getComplexLU (const int n, cuDoubleComplex* A, int* indPivot) {
	
	//int info = 0;
	for (int j = 0; j < n; ++j) {
		
		// find pivot and test for singularity
		
		int jp = j + getComplexMax (n - j, &A[j + (STRIDE * j)]);
		indPivot[j] = jp;

    	if (cuCabs(A[jp + (STRIDE * j)]) > 0.0) {
			
			// apply interchange to columns 1:n-1
			if (jp != j)
				swapComplex (n, &A[j], STRIDE, &A[jp], STRIDE);
			
			// compute elements j+1:m-1 of the jth column
			
			if (j < n - 1)
				scaleComplex (n - j - 1, cuCdiv(make_cuDoubleComplex(1.0, 0.0), A[j + (STRIDE * j)]), &A[j + 1 + (STRIDE * j)]);
			
		} //else if (info == 0) {
			//info = j + 1;
		//}
		
		// update trailing submatrix
		if (j < n - 1)
			complexGERU (n - j - 1, make_cuDoubleComplex(-1.0, 0.0), &A[j + 1 + (STRIDE * j)], &A[j + STRIDE * (j + 1)], STRIDE, &A[j + 1 + STRIDE * (j + 1)], STRIDE);
		
	}
	
	//return info;
}

///////////////////////////////////////////////////////////

__device__
void multiplyComplexUpperMV (const int n, cuDoubleComplex* x, const int lda, const cuDoubleComplex* A) {
	
	for (int j = 0; j < n; ++j) {
		//if (x[j] != 0.0) {
    if (cuCabs(x[j]) > 0.0) {
      
			cuDoubleComplex temp = x[j];
			for (int i = 0; i < j; ++i) {
				//x[i] += temp * A[i + (lda * j)];
       			x[i] = cuCfma(temp, A[i + (lda * j)], x[i]);
			}
			//x[j] *= A[j + (lda * j)];
      		x[j] = cuCmul(x[j], A[j + (lda * j)]);
		}
	}
	
}

///////////////////////////////////////////////////////////

__device__
void complexGEMV (const int m, const int n, const cuDoubleComplex alpha, const cuDoubleComplex* A, 
									const cuDoubleComplex* arrX, cuDoubleComplex* arrY) {
	
	// first: y = beta*y
	// beta = 1, so nothing
	
	// second: y = alpha*A*x + y
	
	for (int j = 0; j < n - 1; ++j) {

    if (cuCabs(arrX[j]) > 0.0) {
			cuDoubleComplex temp = cuCmul(alpha, arrX[j]);
      
			for (int i = 0; i < m; ++i) {
				//arrY[i] += temp * A[i + (m * j)];
        		 arrY[i] = cuCfma(temp, A[i + (STRIDE * j)], arrY[i]);
			}
		}
	}
	
}

///////////////////////////////////////////////////////////

__device__
void getComplexInverseLU (const int n, cuDoubleComplex* A, const int* indPivot, cuDoubleComplex* work) {
	
	//int info = 0;
	
	// form inv(U)
	for (int j = 0; j < n; ++j) {
		A[j + (STRIDE * j)] = cuCdiv(make_cuDoubleComplex(1.0, 0.0), A[j + (STRIDE * j)]);
		cuDoubleComplex Ajj = cuCmul(make_cuDoubleComplex(-1.0, 0.0), A[j + (STRIDE * j)]);
		
		// compute elements 0:j-1 of jth column
		multiplyComplexUpperMV (j, &A[STRIDE * j], STRIDE, A);
		
		// scale
		scaleComplex (j, Ajj, &A[STRIDE * j]);
	}
	
	// solve equation inv(A)*L = inv(U) for inv(A)
	
	for (int j = n - 1; j >= 0; --j) {
		
		// copy current column of L to work and replace with 0.0s
		for (int i = j + 1; i < n; ++i) {
			work[i] = A[i + (STRIDE * j)];
			A[i + (STRIDE * j)] = make_cuDoubleComplex(0.0, 0.0);
		}
		
		// compute current column of inv(A)
		if (j < n - 1)
			complexGEMV (n, n - j, make_cuDoubleComplex(-1.0, 0.0), &A[STRIDE * (j + 1)], &work[j + 1], &A[STRIDE * j]);
		
	}
	
	// apply column interchanges
	
	for (int j = n - 2; j >= 0; --j) {
    
		if (indPivot[j] != j)
			swapComplex (n, &A[STRIDE * j], 1, &A[STRIDE * indPivot[j]], 1);
	}
	
	//return info;
}

///////////////////////////////////////////////////////////

__device__
void getComplexInverse (int n, cuDoubleComplex* A) {
	
	// pivot indices
	//int* ipiv = (int*) calloc (n, sizeof(int));
  	int ipiv[STRIDE];
	
	// output flag
	//int info = 0;
	
	// first get LU factorization
	getComplexLU (n, A, ipiv);
	
	// check for successful exit
  /*
	if (info != 0) {
		printf ("getComplexLU failure, info = %d.\n", info);
		exit (1);
	}
  */
	
	// work array
	//cuDoubleComplex* work = (double complex*) calloc (n, sizeof(double complex));
  	cuDoubleComplex work[STRIDE];
	
	// now get inverse
	getComplexInverseLU (n, A, ipiv, work);
	
	//free (work);
	//free (ipiv);
	
	// check for successful exit
  /*
	if (info != 0) {
		printf ("getComplexInverseLU failure, info = %d.\n", info);
		exit (1);
	}
  */
	
}

//Matrix Algorithms: Volume 1: Basic Decompositions
//By G. W. Stewart
__device__
void getHessenbergLU(const int n, cuDoubleComplex* A, int* indPivot)
{
	int last_free = 0;
	for (int i = 0; i < n - 1; i ++)
	{
		if (cuCabs(A[i * STRIDE + i]) < cuCabs(A[i * STRIDE + i + 1]))
		{
			//swap rows
			swapComplex(n - last_free, &A[last_free * STRIDE + i], STRIDE, &A[last_free * STRIDE + i + 1], STRIDE);
			indPivot[i] = i + 1;
		}
		else
		{
			indPivot[i] = i;
			last_free = i;
		}
		if (cuCabs(A[i * STRIDE + i]) > 0.0)
		{
			cuDoubleComplex tau = cuCdiv(A[i * STRIDE + i + 1], A[i * STRIDE + i]);
			for (int j = i + 1; j < n; j++)
			{
				A[j * STRIDE + i + 1] = cuCsub(A[j * STRIDE + i + 1], cuCmul(tau, A[j * STRIDE + i]));
			}
			A[i * STRIDE + i + 1] = tau;
		}
	}
	//last index is not pivoted
	indPivot[n - 1] = n - 1;
}

__device__
void getComplexInverseHessenberg (const int n, cuDoubleComplex* A)
{
	// pivot indices
	int ipiv[STRIDE];
	
	// first get LU factorization
	getHessenbergLU (n, A, ipiv);

	// work array
	cuDoubleComplex work[STRIDE];
	
	// now get inverse
	getComplexInverseLU (n, A, ipiv, work);
}