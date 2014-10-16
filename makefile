SHELL = /bin/sh

.SUFFIXES:
.SUFFIXES: .c .o .cu .cu.o .omp.o

CUDA_PATH = /usr/local/cuda
SDK_PATH = /usr/local/cuda/samples/common/inc/

# Compilers
#CC    = gcc-4.8
CC = gcc
NCC = cc
NCC_BIN = /usr/bin
NVCC = $(CUDA_PATH)/bin/nvcc
LINK   = $(CC) -fPIC
NLINK = $(NCC) -Wl,--no-undefined -fPIC -Xlinker -rpath $(CUDA_PATH)/lib64

# Directories
ODIR = ./obj
SDIR = ./src

#FLAGS, L=0 for testing, L=4 for optimization
ifndef L
  L = 4
endif
#FLAGS, USE_LAPACK, 4 for use MKL in CVODEs, 2 for use the system libraries, 0 for use the serial CVodes version 
ifndef USE_LAPACK
  USE_LAPACK = 4
endif

# Paths
INCLUDES    = -I. -I/usr/local/include/

_DEPS = header.h
DEPS = $(patsubst %,$(SDIR)/%,$(_DEPS))

_OBJ = main.o phiA.o cf.o exp4.o linear-algebra.o complexInverse.o \
       dydt.o fd_jacob.o chem_utils.o mass_mole.o rxn_rates.o spec_rates.o \
       rxn_rates_pres_mod.o mechanism.o
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))

_OBJ_GPU = main.cu.o phiA.cu.o cf.o linear-algebra.o exp4.cu.o complexInverse.cu.o \
           dydt.cu.o fd_jacob.cu.o chem_utils.cu.o mass_mole.o rxn_rates.cu.o \
					 spec_rates.cu.o rxn_rates_pres_mod.cu.o mechanism.o
OBJ_GPU = $(patsubst %,$(ODIR)/%,$(_OBJ_GPU))

_OBJ_CVODES = main_cvodes.o dydt.o chem_utils.o mass_mole.o rxn_rates.o spec_rates.o \
              rxn_rates_pres_mod.o dydt_cvodes.o mechanism.o
OBJ_CVODES = $(patsubst %,$(ODIR)/%,$(_OBJ_CVODES))

_OBJ_KRYLOV = main_krylov.o phiAHessenberg.o cf.o krylov.o linear-algebra.o complexInverse.o \
       dydt.o fd_jacob.o chem_utils.o mass_mole.o rxn_rates.o spec_rates.o \
       rxn_rates_pres_mod.o mechanism.o sparse_multiplier.o
OBJ_KRYLOV = $(patsubst %,$(ODIR)/%,$(_OBJ_KRYLOV))

_OBJ_TEST = unit_tests.o complexInverse.o phiA.o phiAHessenberg.o cf.o linear-algebra.o krylov.o\
            dydt.o fd_jacob.o chem_utils.o mass_mole.o rxn_rates.o spec_rates.o sparse_multiplier.o rxn_rates_pres_mod.o

OBJ_TEST =  $(patsubst %,$(ODIR)/%,$(_OBJ_TEST))


# Paths
INCLUDES = -I. -I$(CUDA_PATH)/include/ -I$(SDK_PATH)
LIBS = -lm -lfftw3 -L$(CUDA_PATH)/lib64 -L/usr/local/lib -lcuda -lcudart -lstdc++ -lsundials_cvodes -lsundials_nvecserial

#flags
#ifeq ("$(CC)", "gcc")
  
ifeq ("$(L)", "0")
  FLAGS = -O0 -g3 -fbounds-check -Wunused-variable -Wunused-parameter \
	        -Wall -ftree-vrp -std=c99 -fopenmp -DDEBUG
  NVCCFLAGS = -g -G -arch=sm_20 -m64 -DDEBUG
else ifeq ("$(L)", "4")
  FLAGS = -O3 -std=c99 -fopenmp -funroll-loops
  NVCCFLAGS = -O3 -arch=sm_20 -m64
endif

ifeq ("$(USE_LAPACK)", "4")
  FLAGS += -DSUNDIALS_USE_LAPACK -I${MKLROOT}/include
  CV_LIBS = -L${MKLROOT}/lib/intel64/ -lmkl_rt -lmkl_intel_lp64 -lmkl_core -lmkl_gnu_thread -ldl -lpthread -lmkl_mc -lmkl_def
else ifeq ("$(USE_LAPACK)", "2")
  FLAGS += -DSUNDIALS_USE_LAPACK
  CV_LIBS = -L/usr/local/lib -llapack -lblas
endif

$(ODIR)/%.o : $(SDIR)/%.c $(DEPS)
	$(CC) $(FLAGS) $(INCLUDES) -c -o $@ $<

$(ODIR)/%.cu.o : $(SDIR)/%.cu $(DEPS)
	$(NVCC) -ccbin=$(NCC_BIN) $(NVCCFLAGS) $(INCLUDES) -dc -o $@ $<

default: $(ODIR) all

$(ODIR):
	mkdir $(ODIR)

all: exp-int exp-int-gpu exp-int-cvodes exp-int-krylov tests

exp-int : $(OBJ)
	$(LINK) $(OBJ) $(LIBS) -llapack $(FLAGS) -o $@

exp-int-krylov : $(OBJ_KRYLOV)
	$(LINK) $(OBJ_KRYLOV) $(LIBS) -llapack $(FLAGS) -o $@

exp-int-gpu : $(OBJ_GPU)
	$(NVCC) -ccbin=$(NCC_BIN) $(OBJ_GPU) $(LIBS) -llapack $(NVCCFLAGS) -dlink -o dlink.o
	$(NLINK) $(OBJ_GPU) dlink.o $(LIBS) -llapack $(FLAGS) -o $@

exp-int-cvodes : $(OBJ_CVODES)
	$(LINK) $(OBJ_CVODES) $(LIBS) $(CV_LIBS) $(FLAGS) -o $@

tests : $(OBJ_TEST)
	$(LINK) $(OBJ_TEST) $(LIBS) $(FLAGS) -o $@

doc : $(DEPS) $(OBJ)
	$(DOXY)

.PHONY : clean		
clean :
	rm -f $(OBJ) $(OBJ_GPU) $(OBJ_CVODES) $(OBJ_KRYLOV) $(OBJ_TEST) exp-int exp-int-gpu exp-int-cvodes exp-int-krylov tests dlink.o
