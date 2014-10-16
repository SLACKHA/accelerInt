#ifndef SPARSE_HEAD
#define SPARSE_HEAD

#define N_A 126
#include "header.h"

void sparse_multiplier (const Real *, const Real *, Real*);

#ifdef COMPILE_TESTING_METHODS
  int test_sparse_multiplier();
#endif

#endif
