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
NLINK = $(NCC) -fPIC -fopenmp -Xlinker -rpath $(CUDA_PATH)/lib64
FLAGS = 
NVCCFLAGS = -Xcompiler -fopenmp
NVCCINCLUDES = -I$(CUDA_PATH)/include/ -I$(SDK_PATH)
NVCCLIBS = -L$(CUDA_PATH)/lib64 -L/usr/local/lib -lcuda -lcudart -lstdc++
LIBS = -lm
RA_LIBS = -lfftw3
CV_LIBS = -lsundials_cvodes -lsundials_nvecserial
FAST_MATH = TRUE

# Directories
ODIR := ./obj
SDIR := ./src
LOGDIR := ./log

#Modules
MODULES := rb43 exp4 cvodes prof rates radau2a
MODDIRS := $(patsubst %,$(ODIR)/%/,$(MODULES))

#FLAGS, L=0 for testing, L=4 for optimization
ifndef L
  L = 4
endif
#FLAGS, USE_LAPACK, 4 for use MKL in CVODEs, 2 for use the system libraries, 0 for use the serial CVodes version 
ifndef USE_LAPACK
  USE_LAPACK = 4
endif

# Paths
INCLUDES = -I/usr/local/include/

_DEPS = header.h
DEPS = $(patsubst %,$(SDIR)/%,$(_DEPS))

#generic objects for CPU mechanism
_MECH = dydt.o jacob.o chem_utils.o mass_mole.o rxn_rates.o spec_rates.o rxn_rates_pres_mod.o mechanism.o

#generic objects for GPU mechanism
_MECH_GPU = dydt.cu.o jacob.cu.o chem_utils.cu.o mass_mole.o rxn_rates.cu.o spec_rates.cu.o rxn_rates_pres_mod.cu.o mechanism.cu.o gpu_memory.cu.o

#Generic objects for CPU solver
_OBJ = solver_main.o $(_MECH)

#Generic objects for GPU solver
_OBJ_GPU = solver_main.cu.o $(_MECH_GPU)

#Generic objects for CPU solvers using rational approxmiation / krylov subspaces
_OBJ_RA = cf.o rational_approximant.o phiAHessenberg.o complexInverse.o linear-algebra.o sparse_multiplier.o $(_OBJ)

#Generic objects for GPU solvers using rational approxmiation / krylov subspaces
_OBJ_GPU_RA = cf.o rational_approximant.cu.o phiAHessenberg.cu.o complexInverse.cu.o linear-algebra.o sparse_multiplier.cu.o $(_OBJ_GPU)

#solver specific objects
exprb43-int : FLAGS += -DRB43 -fopenmp
exprb43-int : LIBS += $(RA_LIBS) -fopenmp

exp4-int : FLAGS += -DEXP4 -fopenmp
exp4-int : LIBS += $(RA_LIBS) -fopenmp

exprb43-int-gpu : NVCCFLAGS += $(NVCCINCLUDES) -DRB43
exprb43-int-gpu : LIBS += $(RA_LIBS) $(NVCCLIBS)

exp4-int-gpu : NVCCFLAGS += $(NVCCINCLUDES) -DEXP4
exp4-int-gpu : LIBS +=  $(RA_LIBS) $(NVCCLIBS)

radau2a-int-gpu : NVCCFLAGS += $(NVCCINCLUDES) -DRADAU2A
radau2a-int-gpu : LIBS += $(NVCCLIBS)

cvodes-int : FLAGS += -DCVODES -fopenmp
cvodes-int : LIBS += $(CV_LIBS) -fopenmp

cvodes-analytical-int : FLAGS += -DCVODES -DSUNDIALS_ANALYTIC_JACOBIAN -fopenmp
cvodes-analytical-int : LIBS += $(CV_LIBS) -fopenmp

radau2a-int : FLAGS += -DRADAU2A -fopenmp
radau2a-int : LIBS += -fopenmp

profiler : L = 4
profiler : FLAGS += -pg -DPROFILER -fopenmp
profiler : LIBS += -fopenmp

gpuprofiler : L = 4
gpuprofiler : NVCCFLAGS += -DPROFILER -Xnvlink -v --ptxas-options=-v -lineinfo
gpuprofiler : LIBS += $(NVCCLIBS)

rb43profiler : L = 4
rb43profiler : NVCCFLAGS += -DRB43 -Xnvlink -v --ptxas-options=-v -lineinfo
rb43profiler : LIBS += $(NVCCLIBS) $(RA_LIBS)

ratestest : FLAGS += -DRATES_TEST -fopenmp
ratestest : LIBS += -fopenmp

gpuratestest : NVCCFLAGS += -DRATES_TEST
gpuratestest : LIBS +=  $(NVCCLIBS)

#standard flags
FLAGS += -std=c99
NVCCFLAGS += -arch=sm_20 -m64

#fast math
ifeq ("$(FAST_MATH)", "TRUE")
	NVCCFLAGS += --use_fast_math
else
	NVCCFLAGS += --ftz=false --prec-div=true --prec-sqrt=true --fmad=false
endif

#flags for various debug levels
ifeq ($(L), 0)
  FLAGS += -O0 -g -fbounds-check -Wunused-variable -Wunused-parameter \
	  -Wall -ftree-vrp -DDEBUG
  NVCCFLAGS += -g -G -DDEBUG
else ifeq ($(L), 4)
  FLAGS += -O3 -funroll-loops
  NVCCFLAGS += -O3
endif

#GCC tuning
ifeq ($(L), 4)
  ifeq ("$(CC)", "gcc")
	FLAGS += -mtune=native
  endif
endif

#LAPACK levels
ifeq ($(USE_LAPACK), 4)
  FLAGS += -DSUNDIALS_USE_LAPACK -I${MKLROOT}/include
  LIBS += -L${MKLROOT}/lib/intel64/ -lmkl_rt -lmkl_intel_lp64 -lmkl_core -lmkl_gnu_thread -ldl -lpthread -lmkl_mc -lmkl_def
else ifeq ($(USE_LAPACK), 2)
  FLAGS += -DSUNDIALS_USE_LAPACK
  LIBS += -L/usr/local/lib -llapack -lblas
endif

_OBJ_RB43 = exprb43_init.o exprb43.o solver_generic.o $(_OBJ_RA)
OBJ_RB43 = $(patsubst %,$(ODIR)/rb43/%,$(_OBJ_RB43))

_OBJ_EXP4 = exp4_init.o exp4.o solver_generic.o $(_OBJ_RA)
OBJ_EXP4 = $(patsubst %,$(ODIR)/exp4/%,$(_OBJ_EXP4))

_OBJ_RB43_GPU = exprb43_init.cu.o exprb43.cu.o solver_generic.cu.o $(_OBJ_GPU_RA)
OBJ_RB43_GPU = $(patsubst %,$(ODIR)/rb43/%,$(_OBJ_RB43_GPU))

_OBJ_EXP4_GPU = exp4_init.cu.o exp4.cu.o solver_generic.cu.o $(_OBJ_GPU_RA)
OBJ_EXP4_GPU = $(patsubst %,$(ODIR)/exp4/%,$(_OBJ_EXP4_GPU))

_OBJ_CVODES = cvodes_dydt.o cvodes_init.o solver_cvodes.o $(filter-out jacob.o,$(_OBJ))
OBJ_CVODES = $(patsubst %,$(ODIR)/cvodes/%,$(_OBJ_CVODES))

_OBJ_CVODES_ANALYTICAL = cvodes_dydt.o cvodes_jac.o cvodes_init.o solver_cvodes.o $(_OBJ)
OBJ_CVODES_ANALYTICAL = $(patsubst %,$(ODIR)/cvodes/%,$(_OBJ_CVODES_ANALYTICAL))

_OBJ_PROFILER = rateOutputTest.o $(_MECH)
OBJ_PROFILER = $(patsubst %,$(ODIR)/prof/%,$(_OBJ_PROFILER))

_OBJ_GPU_PROFILER = rateOutputTest.cu.o $(_MECH_GPU)
OBJ_GPU_PROFILER = $(patsubst %,$(ODIR)/prof/%,$(_OBJ_GPU_PROFILER))

_OBJ_RB43_GPU_PROFILER = solver_profiler.cu.o exprb43_init.cu.o $(filter-out solver_main.cu.o,$(_OBJ_GPU_RA))
OBJ_RB43_GPU_PROFILER = $(patsubst %,$(ODIR)/prof/%,$(_OBJ_RB43_GPU_PROFILER))

_OBJ_RATES_TEST = rateOutputTest.o $(_MECH) 
OBJ_RATES_TEST = $(patsubst %,$(ODIR)/rates/%,$(_OBJ_RATES_TEST))

_OBJ_GPU_RATES_TEST = rateOutputTest.cu.o $(_MECH_GPU)
OBJ_GPU_RATES_TEST = $(patsubst %,$(ODIR)/rates/%,$(_OBJ_GPU_RATES_TEST))

_OBJ_GPU_RADAU2A = radau2a.cu.o radau2a_init.cu.o  inverse.cu.o complexInverse_NN.cu.o solver_generic.cu.o $(_OBJ_GPU)
OBJ_GPU_RADAU2A = $(patsubst %,$(ODIR)/radau2a/%,$(_OBJ_GPU_RADAU2A))

_OBJ_RADAU2A = radau2a.o radau2a_init.o solver_generic.o $(_OBJ)
OBJ_RADAU2A = $(patsubst %,$(ODIR)/radau2a/%,$(_OBJ_RADAU2A))

define module_rules
$(ODIR)/$1/%.o : $(SDIR)/%.c $(DEPS)
	$(shell test -d $(ODIR)/$1 || mkdir -p $(ODIR)/$1)
	$(CC) $$(FLAGS) $$(INCLUDES) -c -o $$@ $$<

$(ODIR)/$1/%.cu.o : $(SDIR)/%.cu $(DEPS)
	$(shell test -d $(ODIR)/$1 || mkdir -p $(ODIR)/$1)
	$(NVCC) -ccbin=$$(NCC_BIN) $$(NVCCFLAGS) $$(INCLUDES) $$(NVCCINCLUDES) -dc -o $$@ $$<
endef

default: all

all : $(ODIR) $(LOGDIR) exprb43-int exp4-int exprb43-int-gpu exp4-int-gpu cvodes-int ratestest gpuratestest radau2a-int-gpu radau2a-int
$(ODIR):
	mkdir -p $(ODIR)
$(LOGDIR):
	mkdir -p $(LOGDIR)
.PHONY: clean all $(ODIR) $(LOGDIR)

log_maker := $(shell test -d $(LOGDIR) || mkdir -p $(LOGDIR))

print-%  : ; @echo $* = $($*)

exprb43-int : $(OBJ_RB43)
	$(LINK) $(OBJ_RB43) $(LIBS) -o $@

exp4-int : $(OBJ_EXP4)
	$(LINK) $(OBJ_EXP4) $(LIBS) -o $@

exprb43-int-gpu : $(OBJ_RB43_GPU)
	$(NVCC) -ccbin=$(NCC_BIN) $(OBJ_RB43_GPU) $(LIBS) -dlink -o dlink.o
	$(NLINK) $(OBJ_RB43_GPU) dlink.o $(LIBS) -o $@

exp4-int-gpu : $(OBJ_EXP4_GPU)
	$(NVCC) -ccbin=$(NCC_BIN) $(OBJ_EXP4_GPU) $(LIBS) -dlink -o dlink.o
	$(NLINK) $(OBJ_EXP4_GPU) dlink.o $(LIBS) -o $@

cvodes-int : $(OBJ_CVODES)
	$(LINK) $(OBJ_CVODES) $(LIBS) -o $@

cvodes-analytical-int : $(OBJ_CVODES_ANALYTICAL)
	$(LINK) $(OBJ_CVODES_ANALYTICAL) $(LIBS) -o $@

radau2a-int : $(OBJ_RADAU2A)
	$(LINK) $(OBJ_RADAU2A) $(LIBS) -o $@

radau2a-int-gpu : $(OBJ_GPU_RADAU2A)
	$(NVCC) -ccbin=$(NCC_BIN) $(OBJ_GPU_RADAU2A) $(LIBS) -dlink -o dlink.o
	$(NLINK) $(OBJ_GPU_RADAU2A) dlink.o $(LIBS) -o $@

profiler : $(OBJ_PROFILER)
	$(LINK) $(OBJ_PROFILER) $(LIBS) -o $@

gpuprofiler : $(OBJ_GPU_PROFILER)
	$(NVCC) -ccbin=$(NCC_BIN) $(OBJ_GPU_PROFILER) $(LIBS) -dlink -o dlink.o
	$(NLINK) $(OBJ_GPU_PROFILER) dlink.o $(LIBS) -o $@

ratestest : $(OBJ_RATES_TEST)
	$(LINK) $(OBJ_RATES_TEST) $(LIBS) -o $@

gpuratestest : $(OBJ_GPU_RATES_TEST)
	$(NVCC) -ccbin=$(NCC_BIN) $(OBJ_GPU_RATES_TEST) $(LIBS) -dlink -o dlink.o
	$(NLINK) $(OBJ_GPU_RATES_TEST) dlink.o $(LIBS) -o $@

rb43profiler : $(OBJ_RB43_GPU_PROFILER)
	$(NVCC) -ccbin=$(NCC_BIN) $(OBJ_RB43_GPU_PROFILER) $(LIBS) -dlink -o dlink.o
	$(NLINK) $(OBJ_RB43_GPU_PROFILER) dlink.o $(LIBS) -o $@

doc : $(DEPS) $(OBJ)
	$(DOXY)

clean :
	rm -f $(OBJ_EXP4) $(OBJ_RB43) $(OBJ_CVODES) $(OBJ_RB43_GPU) $(OBJ_EXP4_GPU) $(OBJ_PROFILER) $(OBJ_GPU_PROFILER) $(OBJ_RATES_TEST) $(OBJ_GPU_RATES_TEST) $(OBJ_RB43_GPU_PROFILER) $(OBJ_RADAU2A) $(OBJ_GPU_RADAU2A) \
		exprb43-int exp4-int exprb43-int-gpu exp4-int-gpu cvodes-int profiler gpuprofiler ratestest gpuratestest rb43profiler radau2a-int-gpu radau2a-int doc \
		dlink.o

$(foreach mod,$(MODULES),$(eval $(call module_rules,$(mod))))