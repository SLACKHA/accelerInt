/* read_initial_conditions.c
 * the generic initial condition reader
 * \file read_initial_conditions
 *
 * \author Nicholas Curtis
 * \date 03/10/2015
 *
 */

#include "header.h"
#include "gpu_memory.cuh"
#include "gpu_macros.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

 int read_initial_conditions(int NUM, int block_size, int grid_size, double** y_host, double** y_device, double** variable_host, double** variable_device) {
    int padded = initialize_gpu_memory(NUM, block_size, grid_size, y_device, variable_device);
    (*y_host) = (double*)malloc(padded * NN * sizeof(double));
    (*variable_host) = (double*)malloc(padded * sizeof(double));
#ifndef SHUFFLE
    FILE *fp = fopen ("ign_data.txt", "r");
#else
    FILE *fp = fopen("shuffled_data.txt", "r");
#endif
    int buff_size = 1024;
    int retries = 0;
    //all lines should be the same size, so make sure the buffer is large enough
    for (retries = 0; retries < 5; retries++)
    {
        char buffer [buff_size];
        if (fgets (buffer, buff_size, fp) != NULL)
        {
            break;
        }
        rewind (fp);
        buff_size *= 2;
    }
    if (retries == 5)
    {
        printf("Could not parse ign_data.txt line with maximum buffer size of %d", buff_size);
        exit(-1);
    }

    //rewind and read
    rewind (fp);

    char buffer [buff_size];
    char *ptr, *eptr;
    double res[NN + 1];
    // load temperature and mass fractions for all threads (cells)
    for (int i = 0; i < NUM; ++i)
    {
        // read line from data file
        if (fgets (buffer, buff_size, fp) == NULL)
        {
            printf("Error reading ign_data.txt, exiting...");
            exit(-1);
        }
        //read doubles from buffer
        ptr = buffer;
        for (int j = 0 ; j <= NN; j++)
        {
            res[j] = strtod(ptr, &eptr);
            ptr = eptr;
        }
        //put into y_host
        (*y_host)[i] = res[0];
#ifdef CONP
        (*variable_host)[i] = res[1];
#elif CONV
        double pres = res[1];
#endif
        for (int j = 2; j <= NN; j++)
            (*y_host)[i + (j - 1) * padded] = res[j];

        // if constant volume, calculate density
#ifdef CONV
        double Yi[NSP];
        double Xi[NSP];

        for (int j = 1; j < NN; ++j)
        {
            Yi[j - 1] = (*y_host)[i + j * padded];
        }

        mass2mole (Yi, Xi);
        (*variable_host)[i] = getDensity ((*y_host)[i], pres, Xi);
#endif
    }
    fclose (fp);

/*
#ifdef SHUFFLE
    // now need to shuffle order
    struct timeval tv;
    gettimeofday(&tv, NULL);
    int usec = tv.tv_usec;
    srand48(usec);

    for (size_t i = NUM - 1; i > 0; i--)
    {
        size_t j = (unsigned int) (drand48() * (i + 1));

        for (size_t ind = 0; ind < NN; ++ind)
        {
            double t = (*y_host)[j + padded * ind];
            (*y_host)[j + padded * ind] = (*y_host)[i + padded * ind];
            (*y_host)[i + padded * ind] = t;

#ifdef CONP
            t = (*variable_host)[j];
            (*variable_host)[j] = (*variable_host)[i];
            (*variable_host)[i] = t;
#else
            t = (*variable_host)[j];
            (*variable_host)[j] = (*variable_host)[i];
            rho_host[i] = t;
#endif
        }
    }
#endif*/
    //finally copy to GPU memory
    cudaErrorCheck(cudaMemcpy(*y_device, *y_host, padded * NN * sizeof(double), cudaMemcpyHostToDevice));
    cudaErrorCheck(cudaMemcpy(*variable_device, *variable_host, padded * sizeof(double), cudaMemcpyHostToDevice));
    return padded;
}