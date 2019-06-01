SUNDIALS_ROOT=$(HOME)/kaust/sundials/2.5.0
TCHEM_ROOT=$(HOME)/kaust/tchem/TChem_v1.0
#SUNDIALS_ROOT=$(HOME)/sundials/

#CLINCPATH= $(AMDAPPSDKROOT)/include
#CLLIBPATH= $(AMDAPPSDKROOT)/lib/x86_64
CLINCPATH= /usr/include
#CLINCPATH= $(TACC_CUDA_INC)
CLLIBPATH= /usr/lib64
##CLLIBPATH= /opt/apps/intel/opencl/libmic

USE_TCHEM= 0
ifneq ($(tchem),)
ifneq ($(tchem),0)
   USE_TCHEM = 1
endif
endif

CV= 0
ifneq ($(cv),)
ifneq ($(cv),0)
   CV= 1
endif
endif

CL= 0
ifneq ($(cl),)
ifneq ($(cl),0)
   CL= 1
endif
endif

USE_OMP= 0
ifneq ($(omp),)
ifneq ($(omp),0)
   USE_OMP = 1
endif
endif

CFLAGS  = -I. #-std=c99

ifeq ($(intel),1)
   CC	= icc -std=c99
   CXX	= icpc
   OMP	= -openmp -openmp-report
   #CFLAGS += -vec-report2
else
   CC	= gcc -std=c99
   CXX	= g++
#  CFLAGS += -Wno-enum-compare
   CFLAGS += -mtune=native
   #CFLAGS += -ftree-vectorizer-verbose=2
   OMP	= -fopenmp
endif

   CFLAGS += -O3
#  CFLAGS += -no-vec
ifeq ($(opt),3)
   CFLAGS += -DFAST_MATH
#  ifeq ($(intel),1)
#     CFLAGS += -xHost
#  endif
   CFLAGS += -O3
   CFLAGS += -ffast-math
endif
ifeq ($(opt),0)
endif

ifeq ($(debug),1)
   CFLAGS += -DDEBUG
   CFLAGS += -g
endif

ifneq ($(USE_OMP),0)
   CFLAGS += $(OMP)
endif

CXXFLAGS= ${CFLAGS} -std=c++0x -Wno-enum-compare

FC	= ifort
FFLAGS	= -O2 -r8
FFLAGS += -g

LDFLAGS = -lstdc++
#ifeq ($(intel),1)
#LDFLAGS+= -mkl=sequential
#endif

ifneq ($(CL),0)
   CFLAGS += -I$(CLINCPATH) -D__ENABLE_OPENCL
   LDFLAGS += -L$(CLLIBPATH) -lOpenCL
endif

ifneq ($(CV),0)
   CFLAGS += -I${SUNDIALS_ROOT}/include -DUSE_SUNDIALS
   LDFLAGS+= -lsundials_cvodes -lsundials_nvecserial -L${SUNDIALS_ROOT}/lib
endif
ifneq ($(USE_TCHEM),0)
   CFLAGS += -I$(TCHEM_ROOT)/include -D__ENABLE_TCHEM=1
   LDFLAGS += -ltchem -L$(TCHEM_ROOT)/lib
endif

#LDFLAGS+= -llapack
#LDFLAGS+= -lfatode -L${SUNDIALS_ROOT}/lib

#LD_PRELOAD=libtbbmalloc_proxy.so.2
#LDFLAGS += -ltbbmalloc_proxy -ltbbmalloc -L/opt/apps/intel/13/composer_xe_2013.2.146/tbb/lib/mic

#CFLAGS += -D__LEVEL1_DCACHE_LINESIZE=`getconf LEVEL1_DCACHE_LINESIZE`
#CFLAGS += -D__ALIGNMENT=`getconf LEVEL1_DCACHE_LINESIZE`

ifeq ($(align),)
   CFLAGS += -D__ALIGNMENT=32
else
   CFLAGS += -D__ALIGNMENT=$(align)
endif

ifeq ($(intel),1)
   ifneq ($(verbose),)
      CFLAGS += -vec-report=$(verbose)
   endif

   MIC_ = 0
   OFFLOAD_ = 0

   CFLAGS += -restrict
   CFLAGS += -wd161 -wd181

   #CFLAGS += -simd

   #ifeq ($(omp),1)
   #   CFLAGS += -openmp -openmp-report
   #endif

   ifeq ($(MIC_),0)
      CFLAGS += -xHost
   endif
endif

AR	= ar rv

EXEC	= test_ck.exe
OBJS	= cklib.o clock.o rk.o ros.o sdirk.o
ifneq ($(CL),0)
   OBJS	+= cl_driver.o
endif

all: exec

# Object rules

%.o: %.f
	${FC} ${FFLAGS} -c $<
%.o: %.F
	${FC} ${FFLAGS} -c $<
%.o: %.c
	${CC} ${CFLAGS} -c $<
%.o: %.cxx
	${CXX} ${CXXFLAGS} -c $<

$(EXEC) : test_ck.cxx ${OBJS}
#%.exe : test_ck.cxx ${OBJS}
	@echo "Building executable $@"
	${CXX} ${CXXFLAGS} -o $@ $< ${OBJS} ${LDFLAGS}

exec:
	make ${EXEC}
objs:
	make ${OBJS}

clean:
	@echo "Cleaning objects ..."
	@/bin/rm -fv ${OBJS} *.o __kernel_source.cl

veryclean: clean
	@echo "Cleaning executables ..."
	@/bin/rm -fv ${EXEC}
