#ifndef __clock_h
#define __clock_h

#include <sys/time.h>

#ifdef __cplusplus
extern "C"
{
#endif

double WallClock (void);

void wallclock (double *t);
void wallclock_ (double *t);
void wallclock__ (double *t);

#ifdef __cplusplus
}
#endif

#endif
