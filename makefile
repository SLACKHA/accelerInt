SHELL = /bin/sh

.SUFFIXES:
.SUFFIXES: .c .o .cu .cu.o .omp.o

CUDA_PATH := /usr/local/cuda
SDK_PATH := /usr/local/cuda/samples/common/inc/
# Directories
ODIR := ./obj
SDIR := ./src
JDIR := ./src/jacobs
LOGDIR := ./log
OUTDIR := ./output
#create
log_maker := $(shell test -d $(LOGDIR) || mkdir -p $(LOGDIR))
obj_maker := $(shell test -d $(ODIR) || mkdir -p $(ODIR))
out_maker := $(shell test -d $(OUTDIR) || mkdir -p $(OUTDIR))
j_maker := $(shell test -d $(JDIR) || mkdir -p $(JDIR))
j_maker2 := $(shell test -f $(JDIR)/jac_list || touch $(JDIR)/jac_list)
#Modules
MODULES := rb43 exp4 cvodes prof rates radau2a cvodes-analytical unittest specrad
MODDIRS := $(patsubst %,$(ODIR)/%/,$(MODULES))

# Paths
INCLUDES = -I/usr/local/include/


#user specifiable variables, control output, optimization level, initial conditions etc.
DEBUG = FALSE
IGN = TRUE
SAME_IC = TRUE
PRINT = FALSE
LOG_OUTPUT = FALSE
SHUFFLE = FALSE
LOW_TOL = FALSE
LARGE_STEP = FALSE
#valid options are MKL, SYS, NONE
USE_LAPACK = MKL
FAST_MATH = FALSE

#used to force a build on different flags
CPU_FILE = CPU_FLAGS
GPU_FILE = GPU_FLAGS
DEF_FILE = DEF_FLAGS

DEFINES = 

#dependencies
_DEPS = $(SDIR)/header.h
CPU_DEPS = $(_DEPS)
GPU_DEPS = $(_DEPS) $(SDIR)/launch_bounds.cuh $(SDIR)/jacobs/jac_list

#turn this into the control flags
ifeq ("$(DEBUG)", "TRUE")
    DEFINES += -DDEBUG
else
	DEFINES += -DNDEBUG
endif
ifeq ("$(USE_LAPACK)", "MKL")
    DEFINES += -DUSE_MKL
else ifeq ("$(USE_LAPACK)", "SYS")
    DEFINES += -DUSE_SYSTEM_LAPACK
else
    DEFINES += -DNO_LAPACK
endif
ifeq ("$(FAST_MATH)", "TRUE")
    DEFINES += -DUSE_FAST_MATH
endif
ifeq ("$(LOW_TOL)", "TRUE")
	DEFINES += -DLOW_TOL
endif
ifeq ("$(LARGE_STEP)", "TRUE")
	DEFINES += -DLARGE_STEP
endif
ifeq ("$(SHUFFLE)", "TRUE")
    DEFINES += -DSHUFFLE
else ifeq ("$(SAME_IC)", "TRUE")
    DEFINES += -DSAME_IC
endif
ifeq ("$(PRINT)", "TRUE")
    DEFINES += -DPRINT
endif
ifeq ("$(IGN)", "TRUE")
    DEFINES += -DIGN
endif
ifeq ("$(LOG_OUTPUT)", "TRUE")
    DEFINES += -DLOG_OUTPUT
endif

#get the stored register count
reg_count := $(shell cat $(SDIR)/regcount)

#compilers
CC = gcc
NCC = cc
NCC_BIN = /usr/bin
NVCC = $(CUDA_PATH)/bin/nvcc
LINK   = $(CC) -fPIC
NLINK = $(NCC) -fPIC -fopenmp -Xlinker -rpath $(CUDA_PATH)/lib64

NVCCFLAGS := -arch=sm_20 -m64 -maxrregcount $(reg_count) -Xcompiler -fopenmp
FLAGS = -std=c99
NVCCINCLUDES = -I$(CUDA_PATH)/include/ -I$(SDK_PATH)
NVCCLIBS = -L$(CUDA_PATH)/lib64 -L/usr/local/lib -lcuda -lcudart -lstdc++
LIBS = -lm
RA_LIBS = -lfftw3
CV_LIBS = -lsundials_cvodes -lsundials_nvecserial

#generic objects for CPU mechanism
_BASE_MECH = dydt.o jacob.o chem_utils.o mass_mole.o rxn_rates.o spec_rates.o rxn_rates_pres_mod.o
_MECH = $(_BASE_MECH) mechanism.o
BASE_MECH = $(patsubst %,$(ODIR)/mech/%,$(_BASE_MECH))
MECH = $(patsubst %,$(ODIR)/mech/%,$(_MECH))

#generic objects for GPU mechanism
_GPU_JACOB_FILES = $(patsubst %.cu,%.jac.cu.o,$(shell cat $(SDIR)/jacobs/jac_list))
_BASE_MECH_GPU = dydt.cu.o jacob.cu.o $(_GPU_JACOB_FILES) chem_utils.cu.o mass_mole.o rxn_rates.cu.o spec_rates.cu.o rxn_rates_pres_mod.cu.o gpu_memory.cu.o
_MECH_GPU = $(_BASE_MECH_GPU) mechanism.cu.o
BASE_MECH_GPU = $(patsubst %,$(ODIR)/mech/%,$(_BASE_MECH_GPU))
MECH_GPU = $(patsubst %,$(ODIR)/mech/%,$(_MECH_GPU))

#Generic objects for CPU solver
_OBJ = solver_main.o read_initial_conditions.o

#Generic objects for GPU solver
_OBJ_GPU = solver_main.cu.o read_initial_conditions.cu.o

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
profiler : FLAGS += -DPROFILER -fopenmp -pg
profiler : LIBS += -fopenmp -pg

gpuprofiler : L = 4
gpuprofiler : NVCCFLAGS += -DPROFILER -Xnvlink -v --ptxas-options=-v -lineinfo
gpuprofiler : LIBS += $(NVCCLIBS)

ratestest : FLAGS += -DRATES_TEST -fopenmp
ratestest : LIBS += -fopenmp

gpuratestest : NVCCFLAGS += -DRATES_TEST
gpuratestest : LIBS +=  $(NVCCLIBS)

gpuunittest : NVCCFLAGS += $(NVCCINCLUDES) -DRB43
gpuunittest : LIBS +=  $(NVCCLIBS)

specradius : FLAGS += -fopenmp
specradius : LIBS += -fopenmp

#add specific flags based on control options, updating will be handled by the files above in conjuction with the control options
ifeq ("$(DEBUG)", "FALSE")
ifeq ("$(FAST_MATH)", "TRUE")
    NVCCFLAGS += --use_fast_math
endif
endif
ifeq ("$(DEBUG)", "FALSE")
ifeq ("$(FAST_MATH)", "TRUE")
ifeq ("$(CC)", "gcc")
    FLAGS += -ffast-math
else ifeq ("$(CC)", "icc")
    FLAGS += -fp-model fast=2
endif
endif
endif

ifeq ("$(FAST_MATH)", "FALSE")
    NVCCFLAGS += --ftz=false --prec-div=true --prec-sqrt=true --fmad=false
endif
ifeq ("$(FAST_MATH)", "FALSE")
ifeq ("$(CC)", "icc")
    FLAGS += -fp-model precise
endif
endif

ifeq ("$(DEBUG)", "FALSE")
    FLAGS += -O3 -funroll-loops
    NVCCFLAGS += -O3
endif
ifeq ("$(DEBUG)", "FALSE")
#tuning
ifeq ("$(CC)", "gcc")
    FLAGS += -mtune=native
else ifeq ("$(CC)", "icc")
    FLAGS += -xhost -ipo
endif
endif

ifeq ("$(DEBUG)", "TRUE")
    FLAGS += -O0 -g -fbounds-check -Wunused-variable -Wunused-parameter \
        -Wall -ftree-vrp
    NVCCFLAGS += -g -G
endif

#LAPACK levels
ifeq ("$(USE_LAPACK)", "MKL")
    FLAGS += -I${MKLROOT}/include
    LIBS += -L${MKLROOT}/lib/intel64/ -lmkl_rt -lmkl_intel_lp64 -lmkl_core -lmkl_gnu_thread -ldl -lpthread -lmkl_mc -lmkl_def
else #still need lapack libraries for CF, but they will not be used by CVODEs or any of the other solvers
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
OBJ_CVODES_ANALYTICAL = $(patsubst %,$(ODIR)/cvodes-analytical/%,$(_OBJ_CVODES_ANALYTICAL))

_OBJ_PROFILER = rateOutputTest.o $(_MECH)
OBJ_PROFILER = $(patsubst %,$(ODIR)/prof/%,$(_OBJ_PROFILER))

_OBJ_GPU_PROFILER = rateOutputTest.cu.o $(_MECH_GPU)
OBJ_GPU_PROFILER = $(patsubst %,$(ODIR)/prof/%,$(_OBJ_GPU_PROFILER))

_OBJ_RATES_TEST = rateOutputTest.o $(_MECH)
OBJ_RATES_TEST = $(patsubst %,$(ODIR)/rates/%,$(_OBJ_RATES_TEST))

_OBJ_GPU_RATES_TEST = rateOutputTest.cu.o $(_MECH_GPU)
OBJ_GPU_RATES_TEST = $(patsubst %,$(ODIR)/rates/%,$(_OBJ_GPU_RATES_TEST))

_OBJ_GPU_RADAU2A = radau2a.cu.o radau2a_init.cu.o  inverse.cu.o complexInverse_NN.cu.o solver_generic.cu.o $(_OBJ_GPU)
OBJ_GPU_RADAU2A = $(patsubst %,$(ODIR)/radau2a/%,$(_OBJ_GPU_RADAU2A))

_OBJ_RADAU2A = radau2a.o radau2a_init.o solver_generic.o $(_OBJ)
OBJ_RADAU2A = $(patsubst %,$(ODIR)/radau2a/%,$(_OBJ_RADAU2A))

_OBJ_GPU_UNITTEST = unit_tests.cu.o complexInverse.cu.o old_complexInverse.cu.o
OBJ_GPU_UNITTEST = $(patsubst %,$(ODIR)/unittest/%,$(_OBJ_GPU_UNITTEST))

_OBJ_SPEC_RAD = spectral_radius_generator.o read_initial_conditions.o
OBJ_SPEC_RAD = $(patsubst %,$(ODIR)/specrad/%,$(_OBJ_SPEC_RAD))

#mechanism files only depend on the actual flags
mech_folder_maker := $(shell test -d $(ODIR)/mech || mkdir -p $(ODIR)/mech)
mech_cpu_maker := $(shell test -f $(ODIR)/mech/$(CPU_FILE) || touch $(ODIR)/mech/$(CPU_FILE))
mech_gpu_maker := $(shell test -f $(ODIR)/mech/$(GPU_FILE) || touch $(ODIR)/mech/$(GPU_FILE))
mech_tmp1 := $(shell grep -Fx -e "$(FLAGS)" $(ODIR)/mech/$(CPU_FILE) || echo "$(FLAGS)" > $(ODIR)/mech/$(CPU_FILE))
mech_tmp2 := $(shell grep -Fx -e "$(NVCCFLAGS)" $(ODIR)/mech/$(GPU_FILE) || echo "$(NVCCFLAGS)" > $(ODIR)/mech/$(GPU_FILE))
#mechanism files
$(ODIR)/mech/%.o : $(SDIR)/%.c $(CPU_DEPS) $(ODIR)/mech/$(CPU_FILE)
	$(CC) $(FLAGS) $(INCLUDES) -c -o $@ $<
$(ODIR)/mech/%.cu.o : $(SDIR)/%.cu $(CPU_DEPS) $(ODIR)/mech/$(GPU_FILE)
	$(NVCC) -ccbin=$(NCC_BIN) $(NVCCFLAGS) $(INCLUDES) $(NVCCINCLUDES) -dc -o $@ $<
#gpu jacob files
$(ODIR)/mech/%.jac.cu.o : $(SDIR)/jacobs/%.cu $(GPU_DEPS $(ODIR)/mech/$(GPU_FILE)
	$(NVCC) -ccbin=$(NCC_BIN) $(NVCCFLAGS) $(INCLUDES) $(NVCCINCLUDES) -dc -o $@ $<
#rates and prof GPU
rate_gpu_maker := $(shell test -f $(ODIR)/rates/$(GPU_FILE) || touch $(ODIR)/rates/$(GPU_FILE))
rate_tmp := $(shell grep -Fx -e "$(NVCCFLAGS)" $(ODIR)/rates/$(GPU_FILE) || echo "$(NVCCFLAGS)" > $(ODIR)/rates/$(GPU_FILE))
$(ODIR)/rates/%.jac.cu.o : $(SDIR)/jacobs/%.cu $(GPU_DEPS $(ODIR)/rates/$(GPU_FILE)
	$(NVCC) -ccbin=$(NCC_BIN) $(NVCCFLAGS) $(INCLUDES) $(NVCCINCLUDES) -dc -o $@ $<
prof_gpu_maker := $(shell test -f $(ODIR)/prof/$(GPU_FILE) || touch $(ODIR)/prof/$(GPU_FILE))
prof_tmp := $(shell grep -Fx -e "$(NVCCFLAGS)" $(ODIR)/prof/$(GPU_FILE) || echo "$(NVCCFLAGS)" > $(ODIR)/prof/$(GPU_FILE))
$(ODIR)/prof/%.jac.cu.o : $(SDIR)/jacobs/%.cu $(GPU_DEPS $(ODIR)/prof/$(GPU_FILE)
	$(NVCC) -ccbin=$(NCC_BIN) $(NVCCFLAGS) $(INCLUDES) $(NVCCINCLUDES) -dc -o $@ $<
#all the various targets
define module_rules
#make folder
folder_maker := $(shell test -d $(ODIR)/$1 || mkdir -p $(ODIR)/$1)
#test for and update (if needed) the defines file
def_maker := $(shell test -f $(ODIR)/$1/$(DEF_FILE) || touch $(ODIR)/$1/$(DEF_FILE))
cpu_maker := $(shell test -f $(ODIR)/$1/$(CPU_FILE) || touch $(ODIR)/$1/$(CPU_FILE))
gpu_maker := $(shell test -f $(ODIR)/$1/$(GPU_FILE) || touch $(ODIR)/$1/$(GPU_FILE))
tmp1 := $(shell grep -Fx -e "$(DEFINES)" $(ODIR)/$1/$(DEF_FILE) || echo "$(DEFINES)" > $(ODIR)/$1/$(DEF_FILE))
tmp2 := $(shell grep -Fx -e "$(FLAGS)" $(ODIR)/$1/$(CPU_FILE) || echo "$(FLAGS)" > $(ODIR)/$1/$(CPU_FILE))
tmp3 := $(shell grep -Fx -e "$(NVCCFLAGS)" $(ODIR)/$1/$(GPU_FILE) || echo "$(NVCCFLAGS)" > $(ODIR)/$1/$(GPU_FILE))

$(ODIR)/$1/%.o : $(SDIR)/%.c $(CPU_DEPS) $(ODIR)/$1/$(DEF_FILE) $(ODIR)/$1/$(CPU_FILE)
	$(CC) $$(FLAGS) $$(DEFINES) $$(INCLUDES) -c -o $$@ $$<

$(ODIR)/$1/%.cu.o : $(SDIR)/%.cu $(GPU_DEPS) $(ODIR)/$1/$(DEF_FILE) $(ODIR)/$1/$(GPU_FILE)
	$(NVCC) -ccbin=$$(NCC_BIN) $$(NVCCFLAGS) $$(DEFINES)  $$(INCLUDES) $$(NVCCINCLUDES) -dc -o $$@ $$<
endef

default: all

all : exprb43-int exp4-int exprb43-int-gpu exp4-int-gpu cvodes-int cvodes-analytical-int radau2a-int-gpu radau2a-int
special : ratestest gpuratestest profiler gpuprofiler
.PHONY: clean all

print-%  : ; @echo $* = $($*)

exprb43-int : $(OBJ_RB43) $(MECH)
	$(LINK) $^ $(LIBS) -o $@

exp4-int : $(OBJ_EXP4) $(MECH)
	$(LINK) $^ $(LIBS) -o $@

exprb43-int-gpu : $(OBJ_RB43_GPU) $(MECH_GPU)
	$(NVCC) -ccbin=$(NCC_BIN) $^ $(LIBS) -dlink -o $(ODIR)/rb43/dlink.o
	$(NLINK) $^ $(ODIR)/rb43/dlink.o $(LIBS) -o $@

exp4-int-gpu : $(OBJ_EXP4_GPU) $(MECH_GPU)
	$(NVCC) -ccbin=$(NCC_BIN) $^ $(LIBS) -dlink -o $(ODIR)/exp4/dlink.o
	$(NLINK) $^ $(ODIR)/exp4/dlink.o $(LIBS) -o $@

cvodes-int : $(OBJ_CVODES) $(MECH)
	$(LINK) $^ $(LIBS) -o $@

cvodes-analytical-int : $(OBJ_CVODES_ANALYTICAL) $(MECH)
	$(LINK) $^ $(LIBS) -o $@

radau2a-int : $(OBJ_RADAU2A) $(MECH)
	$(LINK) $^ $(LIBS) -o $@

radau2a-int-gpu : $(OBJ_GPU_RADAU2A) $(MECH_GPU)
	$(NVCC) -ccbin=$(NCC_BIN) $^ $(LIBS) -dlink -o $(ODIR)/radau2a/dlink.o
	$(NLINK) $^ $(ODIR)/radau2a/dlink.o $(LIBS) -o $@

profiler : $(OBJ_PROFILER)
	$(LINK) $^ $(LIBS) -o $@

gpuprofiler : $(OBJ_GPU_PROFILER)
	$(NVCC) -ccbin=$(NCC_BIN) $^ $(LIBS) -dlink -o $(ODIR)/prof/dlink.o
	$(NLINK) $^ $(ODIR)/prof/dlink.o $(LIBS) -o $@

ratestest : $(OBJ_RATES_TEST)
	$(LINK) $^ $(LIBS) -o $@

gpuratestest : $(OBJ_GPU_RATES_TEST)
	$(NVCC) -ccbin=$(NCC_BIN) $^ $(LIBS) -dlink -o $(ODIR)/rates/dlink.o
	$(NLINK) $^ $(ODIR)/rates/dlink.o $(LIBS) -o $@

gpuunittest : $(OBJ_GPU_UNITTEST)
	$(NVCC) -ccbin=$(NCC_BIN) $^ $(LIBS) -dlink -o $(ODIR)/unittest/dlink.o
	$(NLINK) $^ $(ODIR)/unittest/dlink.o $(LIBS) -o $@

specradius : $(OBJ_SPEC_RAD) $(MECH)
	$(LINK) $^ $(LIBS) -o $@

doc : $(DEPS) $(OBJ)
	$(DOXY)

clean :
	rm -rf $(ODIR) \
		exprb43-int exp4-int exprb43-int-gpu exp4-int-gpu cvodes-int profiler gpuprofiler ratestest gpuratestest radau2a-int-gpu radau2a-int cvodes-analytical-int gpuunittest specradius doc

$(foreach mod,$(MODULES),$(eval $(call module_rules,$(mod))))