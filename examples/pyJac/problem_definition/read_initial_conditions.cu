/**
 * \file
 * \brief the generic CUDA initial condition reader
 *
 * \author Nicholas Curtis
 * \date 03/10/2015
 *
 */

#include "header.cuh"
#include "gpu_memory.cuh"
#include "gpu_macros.cuh"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/time.h>

/**
 * \brief Reads initial conditions for IVPs from binary file
 *
 * \param[in]           filename            the file to read from
 * \param[in]           NUM                 the number of IVP initial conditions to read
 * \param[in,out]       y_host              Address of the host state vector pointer.
                                            This is initialized and populated by this method
 * \param[in, out]      variable_host       Address of the host pressure/density pointer.
                                            This is initialized and populated by this method
 *
 * Note: the data file is expected to be in the following format:\n
        time, Temperature, Pressure, mass fractions         (State #1) \n
        time, Temperature, Pressure, mass fractions         (State #2) ...
 */
 void read_initial_conditions(const char* filename, int NUM, double** y_host, double** variable_host)
 {
    (*y_host) = (double*)malloc(NUM * NN * sizeof(double));
    (*variable_host) = (double*)malloc(NUM * sizeof(double));
    FILE *fp = fopen (filename, "rb");
    if (fp == NULL)
    {
        fprintf(stderr, "Could not open file: %s\n", filename);
        exit(1);
    }
    double buffer[NN + 2];

    // load temperature and mass fractions for all threads (cells)
    for (int i = 0; i < NUM; ++i)
    {
        // read line from data file
        int count = fread(buffer, sizeof(double), NN + 2, fp);
        if (count != (NN + 2))
        {
            fprintf(stderr, "File (%s) is incorrectly formatted, %d doubles were expected but only %d were read.\\n", filename, NN + 1, count);
            exit(-1);
        }
        //apply mask if necessary
        apply_mask(&buffer[3]);
        //put into y_host
        (*y_host)[i] = buffer[1];
#ifdef CONP
        (*variable_host)[i] = buffer[2];
#elif CONV
        double pres = buffer[2];
#endif
        for (int j = 0; j < NSP; j++)
            (*y_host)[i + (j + 1) * NUM] = buffer[j + 3];

        // if constant volume, calculate density
#ifdef CONV
        double Yi[NSP];
        double Xi[NSP];

        for (int j = 1; j < NN; ++j)
        {
            Yi[j - 1] = (*y_host)[i + j * NUM];
        }

        mass2mole (Yi, Xi);
        (*variable_host)[i] = getDensity ((*y_host)[i], pres, Xi);
#endif
    }
    fclose (fp);
}