SHELL = /bin/sh

.SUFFIXES:
.SUFFIXES: .c .o .cu .cu.o .omp.o

CUDA_PATH = /usr/local/cuda
SDK_PATH = /usr/local/cuda/samples/common/inc/

# Compilers
#CC    = gcc-4.8
CC = gcc
NVCC = $(CUDA_PATH)/bin/nvcc
LINK   = $(CC) -fPIC -Xlinker -rpath $(CUDA_PATH)/lib64

# Directories
ODIR = ./obj
SDIR = ./src

#FLAGS, L=0 for testing, L=4 for optimization
ifndef L
  L = 4
endif

# Paths
INCLUDES    = -I.
ALLLIBS        = -llapack -lm -lfftw3

_DEPS = head.h
DEPS = $(patsubst %,$(SDIR)/%,$(_DEPS))

_OBJ = main.o phiA.o cf.o exp4.o linear-algebra.o complexInverse.o \
       dydt.o jacob.o chem_utils.o mass_mole.o rxn_rates.o spec_rates.o \
       rxn_rates_pres_mod.o
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))

_OBJ_GPU = main.cu.o phiA.cu.o cf.o exp4.cu.o complexInverse.cu.o \
           dydt.cu.o jacob.cu.o chem_utils.cu.o mass_mole.o rxn_rates.cu.o \
					 spec_rates.cu.o rxn_rates_pres_mod.cu.o
OBJ_GPU = $(patsubst %,$(ODIR)/%,$(_OBJ_GPU))

# Paths
INCLUDES = -I. -I$(CUDA_PATH)/include/ -I$(SDK_PATH)
LIBS = -lm $(ALLLIBS) -L$(CUDA_PATH)/lib64 -lcuda -lcudart

#flags
#ifeq ("$(CC)", "gcc")
  
ifeq ("$(L)", "0")
  FLAGS = -O0 -g3 -fbounds-check -Wunused-variable -Wunused-parameter \
	        -Wall -ftree-vrp -std=c99 \
else ifeq ("$(L)", "4")
  FLAGS = -O3 -std=c99 -fopenmp
endif

NVCCFLAGS = -O3 -arch=sm_20 -m64

$(ODIR)/%.cu.o : $(SDIR)/%.c $(DEPS)
	$(CC) -c -o $@ $< $(FLAGS) $(INCLUDES)

$(ODIR)/%.o : $(SDIR)/%.cu $(DEPS)
	$(NVCC) -dc -o $@ $< $(NVCCFLAGS) $(INCLUDES)

default: $(ODIR) all

$(ODIR):
	mkdir $(ODIR)

all: exp-int exp-int-gpu

exp-int : $(OBJ)
	$(LINK) -o $@ $(OBJ) $(LIBS) $(FLAGS)

exp-int-gpu : $(OBJ_GPU)
	$(NVCC) $(OBJ_GPU) $(LIBS) $(NVCCFLAGS) -dlink -o dlink.o
	$(LINK) -o $@ $(OBJ_GPU) dlink.o $(LIBS) $(FLAGS)

doc : $(DEPS) $(OBJ)
	$(DOXY)

.PHONY : clean		
clean :
	rm -f $(OBJ) $(OBJ_GPU) exp-int exp-int-gpu dlink.o
