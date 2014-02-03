SHELL = /bin/sh

.SUFFIXES:
.SUFFIXES: .c .o

# Compilers
CC    = gcc-4.8
#CC	= icc
LINK   = $(CC)

# Directories
ODIR = ./obj
SDIR = ./src

#FLAGS, L=0 for testing, L=4 for optimization
ifndef L
  L = 4
endif

# Paths
INCLUDES    = -I.
LIBS        = -llapack -lm -lfftw3

_OBJ = main.o phiA.o cf.o exp4.o linear-algebra.o complexInverse.o \
      dydt.o jacob.o chem_utils.o mass_mole.o rxn_rates.o spec_rates.o rxn_rates_pres_mod.o
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))

_DEPS = head.h
DEPS = $(patsubst %,$(SDIR)/%,$(_DEPS))

#flags
#ifeq ("$(CC)", "gcc")
  
ifeq ("$(L)", "0")
  FLAGS = -O0 -g3 -fbounds-check -Wunused-variable -Wunused-parameter \
	        -Wall -ftree-vrp -std=c99 \
					#-fsanitize=address -fno-omit-frame-pointer -fno-common
else ifeq ("$(L)", "4")
  FLAGS = -O3 -std=c99
#		FLAGS += -ffast-math
endif
#endif

$(ODIR)/%.o : $(SDIR)/%.c $(DEPS)
	$(CC) -c -o $@ $< $(FLAGS) $(INCLUDES)

all: exp-int

exp-int : $(OBJ)
	$(LINK) -o $@ $(OBJ) $(LIBS) $(FLAGS)
#	strip $@

doc : $(DEPS) $(OBJ)
	$(DOXY)

.PHONY : clean		
clean :
	rm -f $(OBJ) exp-int
